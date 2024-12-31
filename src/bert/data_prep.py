from audioop import reverse

import torch
import numpy as np
import scanpy as sc
from typing import Union, Optional, List, Literal
from pathlib import Path
from anndata import AnnData
from rich.console import Console
from rich.pretty import pprint
from rich.progress import track
from pydantic import BaseModel
from scripts.regsetup import description
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances

console = Console()


class NeighborInteractivity(BaseModel):
    interactivity_score: float
    location: List[float]
    distance: float
    angle: float


class VariableGene(BaseModel):
    gene_name: str
    expression_level: float
    dispersion_level: float


class ModelFeatures(BaseModel):
    tissue_type: str
    spatial_coords: List[float]
    cancer_score: float
    top_var_genes: List[VariableGene]
    cell_reactivity: float
    neighbor_interactivities: Optional[List[NeighborInteractivity]]
    mito_activity: float
    is_border: bool


class DataPrep:
    """Data preparation for transformer model with biological focus"""

    def __init__(self, adata: Union[str, AnnData, Path], device: str = 'cuda'):
        """Initialize data preparation with biological focus

        :param adata: AnnData object or path to h5ad file

        """
        print('\n\n------------ Starting Data preparation pipeline --\n')
        # Load data if path is provided
        if isinstance(adata, (str, Path)):
            console.print("Loading data from file...")
            self.adata = sc.read_h5ad(adata)
        else:
            self.adata = adata

        self.scaler = MinMaxScaler()
        # Normalize the gene expressions to [0, 1] - MinMax
        console.print('--> Normalizing gene expression to range [0, 1]')
        gene_expr_dense = self.adata.X.toarray() if hasattr(self.adata.X, 'toarray') else self.adata.X
        self.adata.X = self.scaler.fit_transform(gene_expr_dense)

        # Scale spatial coords to a range of [[-100: 100], [-100, 100]]
        spatial_coords = self.adata.obsm['spatial']
        target_range = (-100, 100)
        a, b = target_range
        # Calculate min an max value for eacg dim
        min_vals = spatial_coords.min(axis=0)
        max_vals = spatial_coords.max(axis=0)

        scaled_spatial = a + (spatial_coords - min_vals) * (b - a) / (max_vals - min_vals)
        self.spatial_coords = scaled_spatial

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self._cell_to_cell_interactivity(30, 70, 0.5)

    def prepare_data(self) -> List[ModelFeatures]:
        """Prepare data for the transformer model"""

        features: List[ModelFeatures] = []
        reactivities = self._cell_reactivity(30, 70)
        border_cells = self._is_border_cell(90, 10)

        for i in track(range(len(self.adata.var['dispersions_norm'])), description='--> Building features... '):
            # Calculate top variable genee
            top_genes = self._get_hvg_per_cell(i, 5)
            neighbors = self._get_cell_neighbors(i, 5)

            features.append(
                ModelFeatures(
                    tissue_type=self.adata.obs['tissue_type'][i],
                    spatial_coords=self.spatial_coords[i],
                    cancer_score=self.adata.obs['cancer_score'][i],
                    top_var_genes=[VariableGene(gene_name=hvgene[0],
                                                expression_level=hvgene[1],
                                                dispersion_level=hvgene[2])
                                   for hvgene in top_genes],
                    cell_reactivity=reactivities[i],
                    is_border=border_cells[i],
                    neighbor_interactivities=neighbors,
                    mito_activity=self.adata.obs['pct_counts_mito'][i],
                )
            )

        return features

    def _cell_reactivity(self, graph_weight: float, spatial_weight: float):
        # Extract the connectivity matrix from the AnnData object
        connectivities = self.adata.obsp['connectivities']

        # Sum the connectivities for each cell and flatten the result to a 1D array
        G = np.array(connectivities.sum(axis=1)).flatten()

        # Compute the pairwise Euclidean distances between spatial coordinates
        spatial_distances = euclidean_distances(self.spatial_coords)

        # Set the diagonal of the distance matrix to infinity to avoid division by zero
        np.fill_diagonal(spatial_distances, np.inf)

        # Compute the inverse of the spatial distances
        inverse_distances = 1 / spatial_distances

        # Sum the inverse distances for each cell
        p = inverse_distances.sum(axis=1)

        # Normalize the connectivity and inverse distance sums
        G_norm = self.scaler.fit_transform(G.reshape(-1, 1)).flatten()
        p_norm = self.scaler.fit_transform(p.reshape(-1, 1)).flatten()

        # Return the weighted product of the normalized connectivity and inverse distance sums
        return (graph_weight * G_norm) + (spatial_weight * p_norm)

    def _cell_to_cell_interactivity(self, graph_weight: float, spatial_weight: float, distance_percentile: float):
        console.print("--> Calculating cell-cell interactivity")
        # Extract connectivities and spatial coordinates
        connectivities = self.adata.obsp['connectivities']

        # Compute pairwise Euclidean distances
        spatial_distances = euclidean_distances(self.spatial_coords)

        # Avoid division by zero and exclude self-distances
        np.fill_diagonal(spatial_distances, np.inf)

        # Determine the distance threshold based on the specified percentile
        distance_threshold = np.percentile(spatial_distances[spatial_distances != np.inf], distance_percentile)
        console.print(f"--> Distance threshold (percentile {distance_percentile}): {distance_threshold}")

        # Apply distance filter: mask distances exceeding the threshold
        distance_mask = spatial_distances < distance_threshold
        spatial_distances[~distance_mask] = np.inf  # Set excluded distances to inf for inverse calculation

        # Compute inverse spatial distances (for proximity-based interaction)
        P = 1 / spatial_distances
        P[~distance_mask] = 0  # Set contributions from excluded neighbors to zero

        # Convert connectivity matrix to dense format
        G = connectivities.toarray()

        # Normalize G and P row-wise
        G_norm = self.scaler.fit_transform(G)
        P_norm = self.scaler.fit_transform(P)

        # Compute the weighted combination
        interactivity = (graph_weight * G_norm) + (spatial_weight * P_norm)
        self.cell_interactivities = interactivity

        return interactivity

    def _is_border_cell(self, limit_percentile: int, distance_percentile: float):
        # Extract spatial coordinates, clusters and connectivities
        clusters = self.adata.obs['clusters'].to_numpy()
        connectivities = self.adata.obsp['connectivities']

        # Compute pairwise spatial distances
        distances = euclidean_distances(self.spatial_coords)

        # Determine the dynamic distance threshold based on the top 20th percentile of distances
        distance_thresholds = np.percentile(distances, distance_percentile, axis=1)

        # Initialize an array to store border cell status
        diff_cluster_count = np.zeros(len(clusters))

        # Iterate over each cell to compute diff_cluster_count
        for i in track(range(len(clusters)), description="Checking for border cells..."):
            # Find neighbors within the dynamic distance threshold for each cell
            spatial_neighbors = np.where(distances[i] < distance_thresholds[i])[0]
            graph_neighbors = connectivities[i].nonzero()[1]
            neighbors = np.intersect1d(spatial_neighbors, graph_neighbors)
            neighbor_clusters = clusters[neighbors]
            current_cluster = clusters[i]

            # Count neighbors in different clusters
            diff_cluster_count[i] = np.sum(neighbor_clusters != current_cluster)

        # Calculate limit threshold as specified percentile
        limit_threshold = np.percentile(diff_cluster_count, limit_percentile)

        # Mark cells as border cells if their diff_cluster_count exceeds the threshold
        is_border = diff_cluster_count > limit_threshold

        return is_border

    def _get_hvg_per_cell(self, cell_index: int, top_n: int = 5):
        """
        Get the most expressed highly variable genes in single cell
        :param cell_index: index of the cell
        :param top_n: NUmber of genes to return
        :return:
        """
        if 'dispersions_norm' not in self.adata.var.columns:
            raise ValueError('Gene variability metrics are missing. Run sc.pp.highly_variable_genes on your adata.')

        # Subset to highly variable genes
        self.adata = self.adata[:, self.adata.var['highly_variable']]

        # Extract expression values for the specific cell
        expression_values = self.adata.X[cell_index].toarray().flatten() \
            if hasattr(self.adata.X[cell_index], 'toarray') \
            else self.adata.X

        # Combine expression values for the cell
        genes_info = [
            (gene, round(expression_values[i], 2), round(self.adata.var.loc[gene, 'dispersions_norm'], 2))
            for i, gene in enumerate(self.adata.var_names)
        ]  # gene_id, expression_value, dispersion

        # Sort by expression values in the specific cell and by variability
        sorted_genes = sorted(genes_info, key=lambda x: (x[1], x[2]), reverse=True)

        # select top n genes
        top_genes = sorted_genes[:top_n]
        return top_genes

    def _get_cell_neighbors(self, cell_index: int, n_neighbors: int = 10):
        """
        Construct neighborhood data for a single cell
        :param cell_index: index of the cell to be processed
        :return:
        """
        # Calculate the distances between the cell and its neighbors
        cell_location = self.spatial_coords[cell_index]
        neighbors = np.nonzero(self.cell_interactivities[cell_index])[0]
        scores = self.cell_interactivities[cell_index, neighbors]
        sorted_neighbors = sorted(
            zip(neighbors, scores),
            key=lambda x: x[1],
            reverse=True
        )
        neighbors = [neighbor for neighbor, score in sorted_neighbors[:n_neighbors]]

        neighbor_data = []
        for neighbor in neighbors:
            neighbor_location = self.spatial_coords[neighbor]
            distance = np.linalg.norm(cell_location - neighbor_location)

            x1, y1 = cell_location
            x2, y2 = neighbor_location

            # Compute vector
            dx = x2 - x1
            dy = y2 - y1

            # Compute the angle radians
            angle_radians = np.arctan2(dy, dx)

            neighbor_data.append(
                NeighborInteractivity(
                    distance=distance,
                    angle=angle_radians,
                    interactivity_score=self.cell_interactivities[cell_index, neighbor],
                    location=neighbor_location
                )
            )

        return neighbor_data

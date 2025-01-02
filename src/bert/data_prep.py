import json
import torch
import numpy as np
import scanpy as sc
from typing import Union, Optional, List, Literal, Dict, Generator, Tuple
from pathlib import Path
from anndata import AnnData
from rich.console import Console
from rich.pretty import pprint
from rich.progress import track
from pydantic import BaseModel
from scripts.regsetup import description
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances
from functools import lru_cache
import scipy.sparse as sp
import ijson

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


class EfficientDataLoader:
    """Efficient data loading with batch processing and memory optimization"""
    
    def __init__(self, batch_size: int = 1000):
        self.batch_size = batch_size
    
    def load_features_batch(self, json_path: str) -> Generator[List[ModelFeatures], None, None]:
        """Load features in batches to minimize memory usage"""
        with open(json_path, 'rb') as file:
            # Create a parser for the JSON array
            parser = ijson.items(file, 'item')
            
            current_batch: List[ModelFeatures] = []
            for feature in parser:
                # Convert feature dict to ModelFeatures object efficiently
                model_feature = self._convert_feature(feature)
                current_batch.append(model_feature)
                
                if len(current_batch) >= self.batch_size:
                    yield current_batch
                    current_batch = []
            
            # Yield remaining features
            if current_batch:
                yield current_batch
    
    @staticmethod
    def _convert_feature(feature: Dict) -> ModelFeatures:
        """Efficiently convert a feature dictionary to ModelFeatures object"""
        # Pre-process nested objects to avoid multiple conversions
        var_genes = [
            VariableGene(
                gene_name=gene['gene_name'],
                expression_level=gene['expression_level'],
                dispersion_level=gene['dispersion_level']
            )
            for gene in feature['top_var_genes']
        ]
        
        neighbor_interactivities = None
        if feature['neighbor_interactivities']:
            neighbor_interactivities = [
                NeighborInteractivity(
                    interactivity_score=n['interactivity_score'],
                    location=n['location'],
                    distance=n['distance'],
                    angle=n['angle']
                )
                for n in feature['neighbor_interactivities']
            ]
        
        return ModelFeatures(
            tissue_type=feature['tissue_type'],
            spatial_coords=feature['spatial_coords'],
            cancer_score=feature['cancer_score'],
            top_var_genes=var_genes,
            cell_reactivity=feature['cell_reactivity'],
            neighbor_interactivities=neighbor_interactivities,
            mito_activity=feature['mito_activity'],
            is_border=feature['is_border']
        )

    def load_all_features(self, json_path: str) -> List[ModelFeatures]:
        """Load all features with progress tracking"""
        all_features = []
        with console.status("[bold green]Loading features efficiently...") as status:
            for batch in self.load_features_batch(json_path):
                all_features.extend(batch)
                status.update(f"[bold green]Loaded {len(all_features)} features...")
        
        console.print(f"[bold green]✓[/] Loaded {len(all_features)} features successfully")
        return all_features


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
        self.device = 'cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        
        # Initialize cache for computed values
        self._cache: Dict = {}
        
        # Preprocess data
        self._preprocess_data()
        
    def _preprocess_data(self):
        """Preprocess data and cache common computations"""
        console.print('--> Preprocessing and caching data...')
        
        # Normalize gene expressions to [0, 1] - MinMax
        gene_expr_dense = self.adata.X.toarray() if sp.issparse(self.adata.X) else self.adata.X
        self.adata.X = self.scaler.fit_transform(gene_expr_dense)

        # Scale spatial coordinates
        spatial_coords = self.adata.obsm['spatial']
        target_range = (-100, 100)
        a, b = target_range
        min_vals = spatial_coords.min(axis=0)
        max_vals = spatial_coords.max(axis=0)
        self.spatial_coords = a + (spatial_coords - min_vals) * (b - a) / (max_vals - min_vals)

        # Pre-compute and cache distance matrix
        self._cache['spatial_distances'] = euclidean_distances(self.spatial_coords)
        np.fill_diagonal(self._cache['spatial_distances'], np.inf)
        
        # Pre-compute connectivities in CSR format
        self._cache['connectivities'] = self.adata.obsp['connectivities'].tocsr()
        
        # Compute cell interactivity
        self._cell_to_cell_interactivity(30, 70, 0.5)

    @lru_cache(maxsize=1)
    def _get_hvg_batch(self, top_n: int = 5) -> np.ndarray:
        """Vectorized version to get top expressed genes for all cells at once"""
        if 'dispersions_norm' not in self.adata.var.columns:
            raise ValueError('Gene variability metrics missing')
        
        # Get expression matrix
        expr_matrix = self.adata.X
        if sp.issparse(expr_matrix):
            expr_matrix = expr_matrix.toarray()
            
        # Get dispersions
        dispersions = self.adata.var['dispersions_norm'].values
        
        # Combine expression and dispersion scores
        combined_scores = expr_matrix * dispersions[None, :]
        
        # Get indices of top N genes per cell
        top_indices = np.argpartition(-combined_scores, top_n, axis=1)[:, :top_n]
        
        return top_indices

    def _cell_reactivity(self, graph_weight: float, spatial_weight: float) -> np.ndarray:
        """Compute cell reactivity using cached matrices"""
        # Use cached connectivities
        G = np.asarray(self._cache['connectivities'].sum(axis=1)).flatten()
        
        # Use cached distances
        inverse_distances = 1 / self._cache['spatial_distances']
        p = inverse_distances.sum(axis=1)
        
        # Normalize scores
        G_norm = self.scaler.fit_transform(G.reshape(-1, 1)).flatten()
        p_norm = self.scaler.fit_transform(p.reshape(-1, 1)).flatten()
        
        return (graph_weight * G_norm) + (spatial_weight * p_norm)

    def _cell_to_cell_interactivity(self, graph_weight: float, spatial_weight: float, distance_percentile: float):
        """Compute cell-to-cell interactivity using cached matrices"""
        console.print("--> Calculating cell-cell interactivity")
        
        spatial_distances = self._cache['spatial_distances']
        distance_threshold = np.percentile(spatial_distances[spatial_distances != np.inf], distance_percentile)
        
        # Create distance mask
        distance_mask = spatial_distances < distance_threshold
        masked_distances = np.where(distance_mask, spatial_distances, np.inf)
        
        # Compute proximity scores
        P = np.where(masked_distances != np.inf, 1 / masked_distances, 0)
        
        # Get normalized connectivity scores
        G = self._cache['connectivities'].toarray()
        G_norm = self.scaler.fit_transform(G)
        P_norm = self.scaler.fit_transform(P)
        
        # Compute weighted combination
        self.cell_interactivities = (graph_weight * G_norm) + (spatial_weight * P_norm)
        return self.cell_interactivities

    def _batch_process_neighbors(self, cell_indices: np.ndarray, n_neighbors: int = 10) -> List[List[NeighborInteractivity]]:
        """Process multiple cells' neighbors at once"""
        # Get locations for all cells
        locations = self.spatial_coords[cell_indices]
        
        # Compute distances and angles using broadcasting
        dx = self.spatial_coords[:, 0][None, :] - locations[:, 0][:, None]
        dy = self.spatial_coords[:, 1][None, :] - locations[:, 1][:, None]
        
        distances = np.sqrt(dx**2 + dy**2)
        angles = np.arctan2(dy, dx)
        
        # Get interactivity scores for all cells
        scores = self.cell_interactivities[cell_indices]
        
        # Process each cell's neighbors
        all_neighbors = []
        for i, cell_idx in enumerate(cell_indices):
            # Get top n_neighbors based on interactivity scores
            neighbor_indices = np.argpartition(-scores[i], n_neighbors)[:n_neighbors]
            
            cell_neighbors = []
            for n_idx in neighbor_indices:
                cell_neighbors.append(
                    NeighborInteractivity(
                        distance=round(float(distances[i, n_idx]), 2),
                        angle=round(float(angles[i, n_idx]), 2),
                        interactivity_score=round(float(scores[i, n_idx]), 2),
                        location=[round(x, 2) for x in self.spatial_coords[n_idx].tolist()]
                    )
                )
            all_neighbors.append(cell_neighbors)
            
        return all_neighbors

    def prepare_data(self) -> List[ModelFeatures]:
        """Prepare data for the transformer model using vectorized operations"""
        console.print("--> Preparing data with vectorized operations")
        
        # Get all required data in batch
        reactivities = self._cell_reactivity(30, 70)
        border_cells = self._is_border_cell(90, 10)
        
        # Process cells in batches for memory efficiency
        batch_size = 1000
        features = []
        
        for start_idx in track(range(0, len(self.adata), batch_size), description="Processing cells in batches"):
            end_idx = min(start_idx + batch_size, len(self.adata))
            batch_indices = np.arange(start_idx, end_idx)
            
            # Get neighbors for batch
            batch_neighbors = self._batch_process_neighbors(batch_indices, 5)
            
            # Get top genes for batch
            batch_top_genes = self._get_hvg_batch(5)[batch_indices]
            
            # Create features for batch
            for i, cell_idx in enumerate(batch_indices):
                top_genes = [
                    VariableGene(
                        gene_name=self.adata.var_names[gene_idx],
                        expression_level=round(float(self.adata.X[cell_idx, gene_idx]), 2),
                        dispersion_level=round(float(self.adata.var['dispersions_norm'].iloc[gene_idx]), 2)
                    )
                    for gene_idx in batch_top_genes[i]
                ]
                
                features.append(
                    ModelFeatures(
                        tissue_type=self.adata.obs['tissue_type'][cell_idx],
                        spatial_coords=[round(x, 2) for x in self.spatial_coords[cell_idx].tolist()],
                        cancer_score=round(float(self.adata.obs['cancer_score'][cell_idx]), 2),
                        top_var_genes=top_genes,
                        cell_reactivity=round(float(reactivities[cell_idx]), 2),
                        is_border=bool(border_cells[cell_idx]),
                        neighbor_interactivities=batch_neighbors[i],
                        mito_activity=round(float(self.adata.obs['pct_counts_mito'][cell_idx]), 2)
                    )
                )
        
        # Write results to file
        with open('tokenizer/data_prep.json', 'w') as f:
            json.dump([feature.model_dump() for feature in features], f, indent=4)
            
        return features

    def _is_border_cell(self, limit_percentile: int, distance_percentile: float) -> np.ndarray:
        """Vectorized implementation of border cell detection"""
        clusters = self.adata.obs['clusters'].to_numpy()
        distances = self._cache['spatial_distances']
        
        # Compute distance thresholds for all cells at once
        distance_thresholds = np.percentile(distances, distance_percentile, axis=1)
        
        # Create mask for valid neighbors
        distance_mask = distances < distance_thresholds[:, None]
        connectivity_mask = self._cache['connectivities'].toarray().astype(bool)
        neighbor_mask = distance_mask & connectivity_mask
        
        # Broadcast clusters for comparison
        cell_clusters = clusters[:, None]
        neighbor_clusters = clusters[None, :]
        
        # Count different cluster neighbors
        diff_cluster_count = np.sum((cell_clusters != neighbor_clusters) & neighbor_mask, axis=1)
        
        # Calculate threshold and return border cell mask
        limit_threshold = np.percentile(diff_cluster_count, limit_percentile)
        return diff_cluster_count > limit_threshold

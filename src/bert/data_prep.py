import torch
import scanpy as sc
import numpy as np
from typing import Tuple
from pathlib import Path
from rich.console import Console
from typing import Tuple, Dict, Optional
import pandas as pd
from scipy.sparse import issparse
from sklearn.preprocessing import StandardScaler

console = Console()


class TransformerDataPrep:
    """Prepares integrated data for transformer model"""

    def __init__(self,
                 adata_path: str,
                 use_pca: bool = True,
                 max_dist: float = None,
                 device='cuda' if torch.cuda.is_available() else 'cpu'
                 ):
        """Initialise data preparation"""
        self.device = device
        self.use_pca = use_pca
        self.max_dist = max_dist

        # Load data
        console.print("Loading data...")
        self.adata = sc.read_h5ad(adata_path)

        console.print(self.adata)

        # Load cancer genes
        if 'present_cancer_genes' in self.adata.uns:
            self.cancer_genes = self.adata.uns['present_cancer_genes']
            console.print(f'Found {len(self.cancer_genes)} cancer genes in AnnData Object')
        else:
            raise ValueError(
                'Cancer genes present in data not detected. Check that your Anndata.uns[present_cancer_genes] is available')

        # Process distances and features
        self._create_feature_matrix()
        self._process_distances()

    def _process_distances(self):
        """Process distances and connectivity information from AnnData"""
        console.print("Processing distance information...")

        # Get distance and connectivities
        distances = self.adata.obsp['distances']
        connectivities = self.adata.obsp['connectivities']

        # Convert to dense matrix if sparse
        if issparse(distances):
            distances = distances.toarray()
        if issparse(connectivities):
            connectivities = connectivities.toarray()

        # Apply distance threshold
        if self.max_dist is not None:
            distances[distances > self.max_dist] = 0

        # Create distance masks (1 where distance is non-zero)
        distance_mask = (distances > 0).astype(np.float32)

        # Normalise distances to [0, 1] range
        max_dist = distances.max()
        distances_norm = distances / max_dist

        # Store processed matrices
        self.distances = torch.tensor(distances_norm, dtype=torch.float32)
        self.distance_mask = torch.tensor(distance_mask, dtype=torch.float32)
        self.connectivities = torch.tensor(connectivities, dtype=torch.float32)

        console.print(f'Distances matrix shape: {self.distances.shape}.')
        console.print(f'Average connections per spot: {self.distance_mask.sum(1).mean():.2f}')

    def _create_feature_matrix(self):
        """Create combined feature matrix with comprehensive features"""
        self.features = pd.DataFrame({
            "cancer_score": self.adata.obs['cancer_score'],
            'total_counts': self.adata.obs['total_counts'],
            'n_genes': self.adata.obs['n_genes_by_counts'],
            'pct_counts_mito': self.adata.obs['pct_counts_mito'],
            'pct_counts_ribo': self.adata.obs['pct_counts_ribo'],
            'total_counts_mito': self.adata.obs['total_counts_mito'],
            'total_counts_ribo': self.adata.obs['total_counts_ribo']
        })

        # Scale the features
        scaler = StandardScaler()
        self.features_scaled = scaler.fit_transform(self.features)

        # Store the feature names for reference
        self.feature_names = list(self.features.columns)

    def prepare_data(self):
        """Prepare integrated data for transformer model"""
        features = []
        feature_info = {}
        current_dim = 0

        # Add PCA embeddings if true
        if self.use_pca:
            pca = torch.tensor(self.adata.obsm['X_pca'], dtype=torch.float32)
            features.append(pca)
            feature_info['pca'] = {
                "start_idx": current_dim,
                "end_idx": current_dim + pca.shape[1]
            }
            current_dim += pca.shape[1]

        # Add cancer features
        cancer_features = torch.tensor(self.features_scaled, dtype=torch.float32)
        features.append(cancer_features)
        feature_info['cancer_features'] = {
            "start_idx": current_dim,
            "end_idx": current_dim + cancer_features.shape[1]
        }
        current_dim += cancer_features.shape[1]

        # Add spatial coordinates
        spatial_coords = torch.tensor(self.adata.obsm['spatial'], dtype=torch.float32)
        features.append(spatial_coords)
        feature_info['spatial'] = {
            "start_idx": current_dim,
            "end_idx": current_dim + spatial_coords.shape[1]
        }

        # Combine all features and move to device
        combined_tensor = torch.cat(features, dim=1).to(self.device)

        distance_info = {
            "distances": self.distances.to(self.device),
            "mask": self.distance_mask.to(self.device),
            "connectivities": self.connectivities.to(self.device)
        }

        # Update feature info
        feature_info = {
            'total_dim': combined_tensor.shape[1],
            'n_spots': combined_tensor.shape[0],
            'feature_names': {
                'cancer_features': list(self.features.columns),
                'spatial': ['x_coord', 'y_coord'],
                'pca': [f'PC_{i}' for i in range(pca.shape[1])] if self.use_pca else [],
            },
            "cancer_genes": self.cancer_genes
        }

        return combined_tensor, distance_info, feature_info

    def get_neighbor_info(self, spot_idx: int) -> Dict:
        """Get detailed information about neighbors for a specific spot"""
        # Get neighbor indices
        neighbors_mask = self.distance_mask[spot_idx] > 0
        neighbor_indices = torch.where(neighbors_mask)[0]

        # Get distances to neighbors
        neighbor_distances = self.distances[spot_idx][neighbors_mask]

        # Get features for spot and neighbors
        spot_features = self.features_scaled.iloc[spot_idx]
        neighbor_features = self.features_scaled.iloc[neighbor_indices]

        return {
            'n_neighbors': len(neighbor_indices),
            'neighbor_indices': neighbor_indices.tolist(),
            'neighbor_distances': neighbor_distances.tolist(),
            'spot_features': spot_features.to_dict(),
            'neighbor_features': neighbor_features.to_dict('records')
        }

    @staticmethod
    def save_processed_data(self, output_dir: str,
                            combined_tensor: torch.Tensor,
                            distance_info: Dict,
                            feature_info: Dict,
                            ):
        """Save processed data and metadata"""

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "combined_features": combined_tensor,
                "distance_info": distance_info,
                "feature_info": feature_info
            },
            output_path / "processed_data.pt"
        )

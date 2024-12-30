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
    """Prepares integrated data for transformer model with biological focus"""

    def __init__(self,
                 adata_path: str,
                 use_pca: bool = True,
                 max_dist: float = None,
                 max_neighbors: int = 12,
                 device='cuda' if torch.cuda.is_available() else 'cpu'
                 ):
        """Initialize data preparation with biological focus"""
        self.device = device
        self.use_pca = use_pca
        self.max_dist = max_dist
        self.max_neighbors = max_neighbors

        # Load data
        console.print("Loading data...")
        self.adata = sc.read_h5ad(adata_path)

        # Load cancer genes
        if 'present_cancer_genes' in self.adata.uns:
            self.cancer_genes = self.adata.uns['present_cancer_genes']
            console.print(f'Found {len(self.cancer_genes)} cancer genes in AnnData Object')
        else:
            self.cancer_genes = []
            console.print('[yellow]Warning: No cancer genes found in AnnData Object[/]')

        # Process data in biological context
        self._create_feature_matrix()  # Basic features
        self._process_distances()      # Spatial relationships
        self._process_spatial_context()  # Advanced spatial features
        self._process_gene_context()     # Gene expression context
        self._process_cell_state()       # Cell state features
        self._scale_features()           # Final scaling

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

        # For each spot, get indices of neighbors sorted by distance
        neighbor_info = []
        for i in range(len(distances)):
            # Get valid neighbors (within distance threshold)
            valid_neighbors = np.where(distances[i] > 0)[0]
            neighbor_distances = distances[i][valid_neighbors]
            
            # Sort by distance
            sorted_indices = np.argsort(neighbor_distances)
            sorted_neighbors = valid_neighbors[sorted_indices]
            sorted_distances = neighbor_distances[sorted_indices]
            
            # Limit to max_neighbors if needed
            if len(sorted_neighbors) > self.max_neighbors:
                sorted_neighbors = sorted_neighbors[:self.max_neighbors]
                sorted_distances = sorted_distances[:self.max_neighbors]
            
            # Store neighbor information
            neighbor_info.append({
                'indices': sorted_neighbors,
                'distances': sorted_distances,
                'n_neighbors': len(sorted_neighbors)
            })

        # Store processed information
        self.neighbor_info = neighbor_info
        self.distances = torch.tensor(distances, dtype=torch.float32)
        self.connectivities = torch.tensor(connectivities, dtype=torch.float32)

        # Log statistics
        n_neighbors = [info['n_neighbors'] for info in neighbor_info]
        console.print(f'Average neighbors per spot: {np.mean(n_neighbors):.2f}')
        console.print(f'Max neighbors in dataset: {max(n_neighbors)}')
        console.print(f'Min neighbors in dataset: {min(n_neighbors)}')

    def _create_feature_matrix(self):
        """Create combined feature matrix with comprehensive features"""
        # Basic features that should always be present
        basic_features = {
            'cancer_score': self.adata.obs['cancer_score'],
            'total_counts': self.adata.obs['total_counts'],
            'n_genes': self.adata.obs['n_genes_by_counts'],
            'pct_counts_mito': self.adata.obs['pct_counts_mito'],
            'pct_counts_ribo': self.adata.obs['pct_counts_ribo']
        }

        # Additional features if available
        additional_features = {}
        for feature in ['total_counts_mito', 'total_counts_ribo']:
            if feature in self.adata.obs:
                additional_features[feature] = self.adata.obs[feature]

        # Initialize features DataFrame
        self.features = pd.DataFrame({**basic_features, **additional_features})
        
    def _scale_features(self):
        """Scale all features after they've been collected"""
        console.print("Scaling features...")
        
        # Create scaler
        scaler = StandardScaler()
        
        # Scale features
        self.features_scaled = scaler.fit_transform(self.features)
        
        # Store feature statistics for tokenizer
        self.feature_stats = {
            name: {'mean': scaler.mean_[i], 'std': scaler.scale_[i]}
            for i, name in enumerate(self.features.columns)
        }
        
        # Store the feature names for reference
        self.feature_names = list(self.features.columns)
        
        # Log feature information
        console.print(f"Processed {len(self.feature_names)} features:")
        for name in self.feature_names:
            console.print(f"  - {name}")

    def _process_spatial_context(self):
        """Process essential spatial context features"""
        console.print("Processing spatial context...")
        spatial_coords = self.adata.obsm['spatial']
        
        # Compute local density features
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=15).fit(spatial_coords)
        distances, indices = nbrs.kneighbors(spatial_coords)
        
        # Local density (cells per unit area)
        local_density = 1 / (distances.mean(axis=1) + 1e-6)
        
        # Local heterogeneity (variance in neighbor features)
        local_heterogeneity = np.array([
            self.features.iloc[indices[i]].std(axis=0).mean() 
            for i in range(len(indices))
        ])
        
        # Border probability (proportion of distant neighbors)
        border_prob = np.array([
            len(np.where(distances[i] > np.median(distances))[0]) / len(distances[i])
            for i in range(len(distances))
        ])
        
        # Add to features
        self.features['local_density'] = local_density
        self.features['local_heterogeneity'] = local_heterogeneity
        self.features['border_probability'] = border_prob
        
        # Log the features
        console.print("[green]Added spatial context features:[/]")
        console.print("  - local_density")
        console.print("  - local_heterogeneity")
        console.print("  - border_probability")

    def _process_gene_context(self):
        """Process gene expression context"""
        console.print("Processing gene expression context...")
        
        # Identify highly variable genes if not already done
        if 'highly_variable' not in self.adata.var:
            sc.pp.highly_variable_genes(self.adata, n_top_genes=1000, flavor='seurat_v3')
        
        # Get indices of highly variable genes
        hvg_indices = np.where(self.adata.var['highly_variable'])[0]
        
        # Calculate mean expression of neighboring cells
        neighbor_expr = np.zeros((len(self.adata), len(hvg_indices)))
        for i, info in enumerate(self.neighbor_info):
            if len(info['indices']) > 0:
                if issparse(self.adata.X):
                    neighbor_expr[i] = self.adata.X[info['indices']][:, hvg_indices].mean(axis=0).A1
                else:
                    neighbor_expr[i] = self.adata.X[info['indices']][:, hvg_indices].mean(axis=0)
        
        # Store HVG information
        self.hvg_indices = hvg_indices
        self.hvg_names = self.adata.var_names[hvg_indices].tolist()
        
        # Calculate differential expression from neighbors
        diff_expr = np.zeros_like(neighbor_expr)
        for i in range(len(self.adata)):
            if issparse(self.adata.X):
                cell_expr = self.adata.X[i, hvg_indices].A1
            else:
                cell_expr = self.adata.X[i, hvg_indices]
            diff_expr[i] = cell_expr - neighbor_expr[i]
        
        # Add top differential genes as features
        top_diff_genes = np.argsort(np.abs(diff_expr).mean(axis=0))[-10:]
        for idx in top_diff_genes:
            gene_name = self.adata.var_names[hvg_indices[idx]]
            self.features[f'diff_expr_{gene_name}'] = diff_expr[:, idx]
        
        # Add mean neighbor expression as feature
        self.features['mean_neighbor_expr'] = neighbor_expr.mean(axis=1)

    def _process_cell_state(self):
        """Process cell state features"""
        console.print("Processing cell state information...")
        
        # Cell cycle scoring if not present
        if 'cell_cycle_score' not in self.adata.obs:
            try:
                sc.tl.score_genes_cell_cycle(self.adata)
                self.features['cell_cycle_score'] = self.adata.obs['cell_cycle_score']
                self.features['s_score'] = self.adata.obs['S_score']
                self.features['g2m_score'] = self.adata.obs['G2M_score']
            except Exception as e:
                console.print(f"[yellow]Warning: Cell cycle scoring failed: {str(e)}[/]")
        
        # Stress response genes
        stress_genes = ['HSP90AA1', 'HSPA1A', 'HSPA5', 'ATF4', 'XBP1']
        present_stress = [g for g in stress_genes if g in self.adata.var_names]
        if present_stress:
            if issparse(self.adata.X):
                stress_score = self.adata.X[:, [self.adata.var_names.get_loc(g) 
                                          for g in present_stress]].mean(axis=1).A1
            else:
                stress_score = self.adata.X[:, [self.adata.var_names.get_loc(g) 
                                          for g in present_stress]].mean(axis=1)
            self.features['stress_score'] = stress_score
        
        # Add metabolic scores if available
        metabolic_genes = {
            'glycolysis': ['HK2', 'PFKP', 'LDHA', 'PKM'],
            'oxidative_phos': ['ATP5A1', 'COX5A', 'NDUFB8', 'SDHB']
        }
        
        for pathway, genes in metabolic_genes.items():
            present_genes = [g for g in genes if g in self.adata.var_names]
            if present_genes:
                if issparse(self.adata.X):
                    score = self.adata.X[:, [self.adata.var_names.get_loc(g) 
                                      for g in present_genes]].mean(axis=1).A1
                else:
                    score = self.adata.X[:, [self.adata.var_names.get_loc(g) 
                                      for g in present_genes]].mean(axis=1)
                self.features[f'{pathway}_score'] = score

    def prepare_data(self):
        """Prepare data for transformer model"""
        # Create combined feature tensor (keep on CPU)
        combined_features = torch.tensor(self.features_scaled, dtype=torch.float32)

        # Create distance information dictionary (keep on CPU)
        distance_info = {
            "neighbor_info": self.neighbor_info,
            "distances": torch.tensor(self.distances.cpu().numpy(), dtype=torch.float32),
            "connectivities": torch.tensor(self.connectivities.cpu().numpy(), dtype=torch.float32)
        }

        # Create feature information dictionary with biological context
        feature_info = {
            "feature_names": {
                "features": self.feature_names,
                "hvg_names": self.hvg_names
            },
            "feature_stats": self.feature_stats,
            "cancer_genes": self.cancer_genes,
            "spatial_context": {
                "local_density_stats": {
                    "mean": float(self.features['local_density'].mean()),
                    "std": float(self.features['local_density'].std())
                },
                "heterogeneity_stats": {
                    "mean": float(self.features['local_heterogeneity'].mean()),
                    "std": float(self.features['local_heterogeneity'].std())
                }
            }
        }

        return combined_features, distance_info, feature_info

    def get_neighbor_info(self, spot_idx: int) -> Dict:
        """Get detailed information about neighbors for a specific spot"""
        # Get neighbor indices
        neighbors_mask = self.distance_mask[spot_idx] > 0
        neighbor_indices = torch.where(neighbors_mask)[0]

        # Get distances to neighbors
        neighbor_distances = self.distances[spot_idx][neighbors_mask]

        # Get features for spot and neighbors
        spot_features = self.features_scaled[spot_idx]
        neighbor_features = self.features_scaled[neighbor_indices]

        return {
            'n_neighbors': len(neighbor_indices),
            'neighbor_indices': neighbor_indices.tolist(),
            'neighbor_distances': neighbor_distances.tolist(),
            'spot_features': dict(zip(self.feature_names, spot_features)),
            'neighbor_features': [
                dict(zip(self.feature_names, features))
                for features in neighbor_features
            ]
        }

    @staticmethod
    def save_processed_data(output_dir: str,
                          combined_tensor: torch.Tensor,
                          distance_info: Dict,
                          feature_info: Dict):
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

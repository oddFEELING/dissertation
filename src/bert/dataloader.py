import torch
from anndata import AnnData
from torch.utils.data import Dataset, DataLoader
import numpy as np
from rich.progress import track
from rich.console import Console
from typing import Dict, List, Optional, Tuple

from src.bert.new_tokeniser import IncrementalTissueTokenizer

console = Console()


class SpatialTranscriptomicsDataset(Dataset):
    """Dataset for spatial transcriptomics data"""

    def __init__(self, adata, processed_data: Dict, tokenizer, indices: Optional[List[int]] = None):
        """
        Initialize dataset with processed spatial transcriptomics data
        
        :param adata: AnnData object
        :param processed_data: Dictionary containing processed features and distances
        :param tokenizer: IncrementalTissueTokenizer instance
        :param indices: Optional indices for train/val split
        """
        self.adata = adata
        self.combined_features = processed_data['combined_features']
        self.neighbor_info = processed_data['distance_info']['neighbor_info']
        self.tokenizer = tokenizer.tokenizer
        self.indices = indices if indices is not None else range(len(adata))

        # Extract feature information
        self.feature_info = processed_data['feature_info']
        self.feature_stats = self.feature_info['feature_stats']
        self.feature_names = self.feature_info['feature_names']['features']

        # Create text tokens for each spot
        console.print("[yellow]Creating text tokens for dataset...[/]")
        self.text_tokens = []

        for idx in track(self.indices, description="Processing spots"):
            spot_tokens = ["[CLS]", "[SPOT]"]

            # Add spatial location
            spatial_x = int(np.floor(20 * (adata.obsm['spatial'][idx, 0] - adata.obsm['spatial'][:, 0].min()) /
                                     (adata.obsm['spatial'][:, 0].max() - adata.obsm['spatial'][:, 0].min())))
            spatial_y = int(np.floor(20 * (adata.obsm['spatial'][idx, 1] - adata.obsm['spatial'][:, 1].min()) /
                                     (adata.obsm['spatial'][:, 1].max() - adata.obsm['spatial'][:, 1].min())))
            spot_tokens.extend([f"[SPATIAL_X_{spatial_x}]", f"[SPATIAL_Y_{spatial_y}]"])

            # Add tissue type if available
            if 'tissue_type' in adata.obs:
                tissue_type = adata.obs['tissue_type'].iloc[idx].upper()
                spot_tokens.append(f"[TISSUE_{tissue_type}]")

            # Add neighbor information
            n_neighbours = self.neighbor_info[idx]['n_neighbors']
            spot_tokens.append(f"[NEIGHBOURS_{int(n_neighbours)}]")

            # Add feature levels
            for feature_name in self.feature_names:
                # Map old feature names to new ones
                if feature_name == 'n_genes':
                    obs_name = 'n_genes_by_counts'
                else:
                    obs_name = feature_name

                value = adata.obs[obs_name].iloc[idx]
                stats = self.feature_stats[feature_name]
                z_score = (value - stats['mean']) / stats['std']

                if feature_name == 'cancer_score':
                    token_prefix = 'CANCER'
                elif feature_name == 'total_counts':
                    token_prefix = 'TOTAL_COUNT'
                elif feature_name == 'n_genes':
                    token_prefix = 'GENE_COUNT'
                elif feature_name == 'pct_counts_mito':
                    token_prefix = 'MITO'
                elif feature_name == 'pct_counts_ribo':
                    token_prefix = 'RIBO'
                elif feature_name == 'total_counts_mito':
                    token_prefix = 'TOTAL_MITO'
                elif feature_name == 'total_counts_ribo':
                    token_prefix = 'TOTAL_RIBO'
                else:
                    continue

                if z_score > 0.5:
                    level = "HIGH"
                elif z_score > -0.5:
                    level = "MED"
                else:
                    level = "LOW"

                spot_tokens.append(f"[{level}_{token_prefix}]")

            # Add gene expression values
            if isinstance(adata.X, np.ndarray):
                gene_expr = adata.X[idx]
            else:
                gene_expr = adata.X[idx].toarray().flatten()

            # Get indices of expressed genes (non-zero expression)
            expressed_indices = np.where(gene_expr > 0)[0]

            # Sort by expression level
            sorted_indices = expressed_indices[
                np.argsort(-gene_expr[expressed_indices])]  # Note the minus sign for descending order

            # Debug information
            if idx == 0:  # Print for first spot only
                console.print(f"[yellow]Gene expression stats for first spot:[/]")
                console.print(f"Total genes: {len(gene_expr)}")
                console.print(f"Expressed genes: {len(expressed_indices)}")
                console.print(f"Max expression: {gene_expr.max():.2f}")
                console.print(f"Mean expression: {gene_expr[gene_expr > 0].mean():.2f}")

            # Add gene expression tokens (take top 50 most expressed genes)
            gene_tokens = []
            for gene_idx in sorted_indices[:50]:  # Limit to top 50 to avoid sequence length issues
                expr_val = gene_expr[gene_idx]
                # Convert to log scale and normalize to 0-9 range
                log_val = np.log1p(expr_val)
                max_log = np.log1p(gene_expr.max())
                normalized_val = int(np.minimum(9, (log_val / max_log) * 9))
                gene_token = f"gene{gene_idx}_{normalized_val}"
                gene_tokens.append(gene_token)

            # Debug first spot
            if idx == 0:
                console.print("[yellow]Sample gene expression tokens:[/]")
                console.print(gene_tokens[:5])

            # Add gene tokens to spot tokens
            spot_tokens.extend(gene_tokens)

            # Debug token sequence for first spot
            if idx == 0:
                console.print("[yellow]Complete token sequence for first spot:[/]")
                console.print(" ".join(spot_tokens))

            # End sequence
            spot_tokens.extend(["[END_SPOT]", "[SEP]"])

            # Join tokens and encode
            token_str = " ".join(spot_tokens)
            encoded = self.tokenizer(
                token_str,
                padding='max_length',
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )

            # Ensure token type IDs are all zeros
            token_type_ids = torch.zeros_like(encoded['input_ids'])

            self.text_tokens.append({
                'input_ids': encoded['input_ids'].squeeze(),
                'attention_mask': encoded['attention_mask'].squeeze(),
                'token_type_ids': token_type_ids.squeeze()
            })

        console.print(f"[green]✓[/green] Created dataset with {len(self.indices)} samples")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """Get a single sample from the dataset"""
        # Get actual index from indices list
        actual_idx = self.indices[idx]

        # Get text tokens
        tokens = self.text_tokens[idx]

        # Get features and neighbor info
        features = self.combined_features[actual_idx]
        neighbor_info = self.neighbor_info[actual_idx]

        # Debug neighbor info structure if it's the first item
        if idx == 0:
            console.print("[yellow]Neighbor info structure:[/]")
            console.print(neighbor_info)

        # Handle gene expression data
        original_gene_expression = None
        if isinstance(self.adata.X, torch.Tensor):
            original_gene_expression = self.adata.X
        elif hasattr(self.adata.X, 'toarray'):
            original_gene_expression = torch.tensor(self.adata.X.toarray(), dtype=torch.float32)
        elif isinstance(self.adata.X, np.ndarray):
            original_gene_expression = torch.tensor(self.adata.X, dtype=torch.float32)

        # Create sample dictionary with proper key names
        sample = {
            'input_ids': tokens['input_ids'],
            'attention_mask': tokens['attention_mask'],
            'token_type_ids': tokens['token_type_ids'],
            'features': features,
            'neighbor_indices': torch.tensor(neighbor_info['indices'], dtype=torch.int64),
            'neighbor_distances': torch.tensor(neighbor_info['distances'], dtype=torch.float32),
            'spatial_coords': torch.tensor(self.adata.obsm['spatial'][actual_idx], dtype=torch.float32),
            'neighbor_coords': torch.tensor(self.adata.obsm['spatial'][neighbor_info['indices']], dtype=torch.float32),
            'spot_id': torch.tensor(actual_idx, dtype=torch.int32),
            'original_features': self.combined_features  # Add full feature matrix
        }

        # Only add gene expression if available
        if original_gene_expression is not None:
            sample['original_gene_expression'] = original_gene_expression
            if idx == 0:  # Debug log for first item
                console.print(f"[yellow]Gene expression shape: {original_gene_expression.shape}[/]")

        return sample


class SpatialDataset(Dataset):
    """Dataset for spatial transcriptomics data with biological focus"""
    def __init__(self, adata: AnnData, processed_data: Dict, tokenizer: IncrementalTissueTokenizer, indices: np.ndarray):
        self.indices = indices
        self.adata = adata
        self.processed_data = processed_data
        self.tokenizer = tokenizer

        # Get feature information
        self.feature_info = processed_data['feature_info']
        self.distance_info = processed_data['distance_info']

        # Extract spatial coordinates
        self.spatial_coords = adata.obsm['spatial']

        # Extract gene expression
        if isinstance(adata.X, np.ndarray):
            self.gene_expression = adata.X
        else:
            self.gene_expression = adata.X.toarray()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get actual index
        actual_idx = self.indices[idx]

        # Get neighbor information
        neighbor_info = self.distance_info['neighbor_info'][actual_idx]
        neighbor_indices = neighbor_info['indices']
        neighbor_distances = neighbor_info['distances']

        # Get spatial coordinates
        spot_coords = self.spatial_coords[actual_idx]
        neighbor_coords = self.spatial_coords[neighbor_indices]

        # Get features
        features = self.processed_data['combined_features'][actual_idx]

        # Get gene expression
        gene_expr = self.gene_expression[actual_idx]

        # Get spatial context with default values if not available
        spatial_context = {}
        spatial_features = ['local_density', 'local_heterogeneity', 'border_probability']
        for feature in spatial_features:
            spatial_context[feature] = torch.tensor(
                self.adata.obs[feature].iloc[actual_idx] if feature in self.adata.obs else 0.0,
                dtype=torch.float32
            )

        # Get cell state information with default values if not available
        cell_state = {}
        state_features = ['cell_cycle_score', 's_score', 'g2m_score', 'stress_score']

        for feature in state_features:
            cell_state[feature] = torch.tensor(
                self.adata.obs[feature].iloc[actual_idx] if feature in self.adata.obs else 0.0,
                dtype=torch.float32
            )

        # Create text input for tokenizer
        text_input = f"[CLS] [TISSUE_1] "

        # Add spatial information
        text_input += f"[LOC_{int(spot_coords[0])}_{int(spot_coords[1])}] "

        # Add feature information
        for feature_name, value in zip(self.feature_info['feature_names']['features'], features):
            text_input += f"[{feature_name.upper()}_{value:.2f}] "

        # Add neighbor information
        for n_idx, n_dist in zip(neighbor_indices, neighbor_distances):
            if n_idx != -1:  # Valid neighbor
                n_coords = self.spatial_coords[n_idx]
                text_input += f"[NEIGHBOR_{int(n_coords[0])}_{int(n_coords[1])}_{n_dist:.2f}] "

        text_input += "[SEP]"

        # Tokenize text
        tokenized = self.tokenizer.tokenize(text_input)

        return {
            'input_ids': tokenized['input_ids'].squeeze(),
            'attention_mask': tokenized['attention_mask'].squeeze(),
            'spatial_coords': torch.tensor(spot_coords, dtype=torch.float32),
            'features': torch.tensor(features, dtype=torch.float32),
            'neighbor_indices': torch.tensor(neighbor_indices, dtype=torch.long),
            'neighbor_distances': torch.tensor(neighbor_distances, dtype=torch.float32),
            'neighbor_coords': torch.tensor(neighbor_coords, dtype=torch.float32),
            'spatial_context': spatial_context,
            'cell_state': cell_state,
            'gene_expression': torch.tensor(gene_expr, dtype=torch.float32)
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate batch with padding"""
    # Get maximum sequence length in batch
    max_len = max(x['input_ids'].size(0) for x in batch)

    # Initialize tensors
    batch_size = len(batch)
    input_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

    # Fill tensors
    for i, item in enumerate(batch):
        seq_len = item['input_ids'].size(0)
        input_ids[i, :seq_len] = item['input_ids']
        attention_mask[i, :seq_len] = item['attention_mask']

    # Collect other tensors
    spatial_coords = torch.stack([x['spatial_coords'] for x in batch])
    features = torch.stack([x['features'] for x in batch])
    neighbor_indices = torch.stack([x['neighbor_indices'] for x in batch])
    neighbor_distances = torch.stack([x['neighbor_distances'] for x in batch])
    neighbor_coords = torch.stack([x['neighbor_coords'] for x in batch])
    gene_expression = torch.stack([x['gene_expression'] for x in batch])

    # Collect spatial context
    spatial_context = {
        key: torch.stack([x['spatial_context'][key] for x in batch])
        for key in batch[0]['spatial_context']
    }

    # Collect cell state
    cell_state = {
        key: torch.stack([x['cell_state'][key] for x in batch])
        for key in batch[0]['cell_state']
    }

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'spatial_coords': spatial_coords,
        'features': features,
        'neighbor_indices': neighbor_indices,
        'neighbor_distances': neighbor_distances,
        'neighbor_coords': neighbor_coords,
        'spatial_context': spatial_context,
        'cell_state': cell_state,
        'gene_expression': gene_expression
    }


def create_st_dataloaders(
        adata: AnnData,
        processed_data: Dict,
        tokenizer: IncrementalTissueTokenizer,
        batch_size: int,
        val_split: float = 0.2
) -> Tuple[DataLoader, DataLoader]:
    """Create dataloaders for spatial transcriptomics data with biological focus"""
    
    # Create train/val split
    n_samples = len(adata)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    split_idx = int(n_samples * (1 - val_split))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    # Create datasets
    train_dataset = SpatialDataset(adata, processed_data, tokenizer, train_indices)
    val_dataset = SpatialDataset(adata, processed_data, tokenizer, val_indices)

    # Create dataloaders with CPU pinning
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=True if torch.cuda.is_available() else False
    )

    return train_loader, val_loader

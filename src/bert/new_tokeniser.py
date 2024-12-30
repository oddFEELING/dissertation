import os
import numpy as np
from pathlib import Path
from numba.np.arrayobj import np_column_stack
from scripts.regsetup import description
from sklearn.preprocessing import StandardScaler
from tensorboard.compat.tensorflow_stub.io.gfile import exists
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from rich.progress import track
from rich.console import Console
from anndata import AnnData
from typing import Tuple, List, Dict
import torch
import logging
from sklearn.cluster import KMeans

console = Console()
logger = logging.getLogger(__name__)


class IncrementalTissueTokenizer:
    """Tokenizer that can incrementally learn from new tissue types with biological focus"""
    
    def __init__(self, max_clusters: int = 50):
        """Initialize tokenizer with biological focus"""
        self.tissue_types = set()
        self.vocab_file = Path("tokens/vocab.txt")
        self.tokenizer_file = Path("tokens/tokenizer.json")
        self.max_clusters = max_clusters
        self.cluster_mappings = {}
        self.hvg_indices = {}  # Store highly variable gene indices per tissue
        self.hvg_names = {}    # Store gene names for HVGs per tissue
        
        # Initialize with WordPiece model
        self.base_tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
        self.base_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        
        self.tokenizer = None
        
    def add_tissue(self, tissue_type: str, adata: AnnData, processed_data: Dict):
        """Add new tissue type with biological focus"""
        self.tissue_types.add(tissue_type)
        
        # Extract highly variable genes if available
        if 'highly_variable' in adata.var:
            self.hvg_indices[tissue_type] = np.where(adata.var['highly_variable'])[0]
        else:
            # Compute highly variable genes
            import scanpy as sc
            sc.pp.highly_variable_genes(adata, n_top_genes=1000, flavor='seurat_v3')
            self.hvg_indices[tissue_type] = np.where(adata.var['highly_variable'])[0]
        
        # Store gene names for HVGs
        self.hvg_names[tissue_type] = adata.var_names[self.hvg_indices[tissue_type]].tolist()
        
        # Extract cluster information
        if 'leiden' in adata.obs:  # Prefer Leiden clusters
            cluster_key = 'leiden'
        elif 'louvain' in adata.obs:
            cluster_key = 'louvain'
        elif 'clusters' in adata.obs:
            cluster_key = 'clusters'
        
        if cluster_key:
            unique_clusters = adata.obs[cluster_key].unique()
            self.cluster_mappings[tissue_type] = {
                cluster: idx % self.max_clusters 
                for idx, cluster in enumerate(unique_clusters)
            }
        
        # Create and save tokens
        new_tokens, _, _ = self._create_tokens(tissue_type, adata, processed_data)
        self.vocab_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.vocab_file, 'w', encoding='utf-8') as f:
            for tokens in new_tokens:
                f.write(tokens + '\n')
        
        self._retrain_tokenizer()

    def _create_tokens(self, tissue_type: str, adata: AnnData, processed_data: Dict) -> Tuple[List[str], np.ndarray, None]:
        """Create biologically focused tokens"""
        spatial_coords = adata.obsm['spatial']
        
        # Compute local density using KNN
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=15).fit(spatial_coords)
        distances, _ = nbrs.kneighbors(spatial_coords)
        local_density = 1 / (distances.mean(axis=1) + 1e-6)
        density_percentiles = np.percentile(local_density, [33, 66])
        
        def encode_spot_features(spot_idx: int) -> List[str]:
            """Encode spot features with biological focus"""
            tokens = []
            
            # Basic spot information
            tokens.append(f"[TISSUE_{tissue_type}]")
            
            # Add cluster information if available
            if any(key in adata.obs for key in ['leiden', 'louvain', 'clusters']):
                cluster_key = next(key for key in ['leiden', 'louvain', 'clusters'] if key in adata.obs)
                cluster = adata.obs[cluster_key].iloc[spot_idx]
                mapped_cluster = self.cluster_mappings[tissue_type][cluster]
                tokens.append(f"[CLUSTER_{mapped_cluster}]")
            
            # Local density token
            density = local_density[spot_idx]
            if density > density_percentiles[1]:
                tokens.append("[DENSITY_HIGH]")
            elif density > density_percentiles[0]:
                tokens.append("[DENSITY_MED]")
            else:
                tokens.append("[DENSITY_LOW]")
            
            # Add cell state markers if available
            if 'cell_cycle_score' in adata.obs:
                score = adata.obs['cell_cycle_score'].iloc[spot_idx]
                tokens.append(f"[CYCLE_{'HIGH' if score > 0.5 else 'LOW'}]")
            
            if 'pct_counts_mito' in adata.obs:
                mito = adata.obs['pct_counts_mito'].iloc[spot_idx]
                if mito > 10:  # Biological threshold for high mitochondrial content
                    tokens.append("[MITO_HIGH]")
            
            # Encode highly variable genes
            hvg_expr = adata.X[spot_idx, self.hvg_indices[tissue_type]]
            if isinstance(hvg_expr, np.ndarray):
                gene_expr = hvg_expr
            else:
                gene_expr = hvg_expr.toarray().flatten()
            
            # Get top expressed HVGs
            top_hvg_indices = np.argsort(-gene_expr)[:10]  # Focus on top 10 HVGs
            for idx in top_hvg_indices:
                if gene_expr[idx] > 0:
                    gene_name = adata.var_names[self.hvg_indices[tissue_type][idx]]
                    tokens.append(f"[GENE_{gene_name}]")
            
            return tokens
        
        # Create tokens for each spot
        tokens_list = []
        for idx in track(range(len(adata)), description=f"Tokenizing {tissue_type} spots"):
            spot_tokens = ["[CLS]"]
            spot_tokens.extend(encode_spot_features(idx))
            spot_tokens.append("[SEP]")
            tokens_list.append(" ".join(spot_tokens))
        
        return tokens_list, spatial_coords, None

    def _retrain_tokenizer(self):
        """Retrain tokenizer with biologically focused tokens"""
        special_tokens = [
            # Basic tokens
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
            
            # Tissue and cluster tokens
            *[f"[TISSUE_{t}]" for t in sorted(self.tissue_types)],
            *[f"[CLUSTER_{i}]" for i in range(self.max_clusters)],
            
            # Density tokens
            "[DENSITY_HIGH]", "[DENSITY_MED]", "[DENSITY_LOW]",
            
            # Cell state tokens
            "[CYCLE_HIGH]", "[CYCLE_LOW]",
            "[MITO_HIGH]",
            
            # Gene-specific tokens (for highly variable genes)
            *[f"[GENE_{gene}]" for tissue_type in self.hvg_names 
              for gene in self.hvg_names[tissue_type][:100]]  # Top 100 HVGs per tissue
        ]
        
        logger.info(f"Number of special tokens: {len(special_tokens)}")
        logger.info(f"Number of tissue types: {len(self.tissue_types)}")
        
        trainer = trainers.WordPieceTrainer(
            vocab_size=30522,
            special_tokens=special_tokens,
            min_frequency=1
        )
        
        self.base_tokenizer.train(files=[str(self.vocab_file)], trainer=trainer)
        self.base_tokenizer.save(str(self.tokenizer_file))
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(self.tokenizer_file))
        self._set_special_tokens(self.tokenizer)

    def _set_special_tokens(self, tokenizer):
        """Set special tokens for the tokenizer"""
        tokenizer.pad_token = "[PAD]"
        tokenizer.mask_token = "[MASK]"
        tokenizer.sep_token = "[SEP]"
        tokenizer.cls_token = "[CLS]"
        tokenizer.unk_token = "[UNK]"
        
    def tokenize(self, text: str) -> Dict:
        """Tokenize text using the current tokenizer"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained on any tissue yet")
        return self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )

    def load_existing_tokenizer(self) -> PreTrainedTokenizerFast:
        """
        Load an existing tokenizer without adding new tissue
        
        :return: Loaded HuggingFace tokenizer
        """
        if not self.tokenizer_file.exists():
            raise ValueError("No existing tokenizer found. Please train the tokenizer first.")
        
        # Create HuggingFace tokenizer from existing file
        hf_tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(self.tokenizer_file))
        hf_tokenizer.pad_token = "[PAD]"
        hf_tokenizer.mask_token = "[MASK]"
        hf_tokenizer.sep_token = "[SEP]"
        hf_tokenizer.cls_token = "[CLS]"
        hf_tokenizer.unk_token = "[UNK]"
        
        self.tokenizer = hf_tokenizer
        return self.tokenizer

    def save(self):
        """Save tokenizer files"""
        # Create directory if it doesn't exist
        self.vocab_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save tokenizer configuration
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(str(self.vocab_file.parent))
            console.print(f"[green]✓[/green] Saved tokenizer to {self.vocab_file.parent}")
        else:
            console.print("[yellow]Warning: No tokenizer to save[/]")
            
    def load(self):
        """Load existing tokenizer"""
        if self.vocab_file.exists() and self.tokenizer_file.exists():
            self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(self.tokenizer_file))
            self.tokenizer.pad_token = "[PAD]"
            self.tokenizer.mask_token = "[MASK]"
            self.tokenizer.sep_token = "[SEP]"
            self.tokenizer.cls_token = "[CLS]"
            self.tokenizer.unk_token = "[UNK]"
            console.print(f"[green]✓[/green] Loaded tokenizer from {self.tokenizer_file}")
        else:
            raise FileNotFoundError(f"Tokenizer files not found in {self.vocab_file.parent}")

# Example of a complete cell/spot token sequence:
# [CLS] [SPOT] [SPATIAL_X_10] [SPATIAL_Y_15] [TISSUE_BRAIN] [NEIGHBOURS_5] [MED_CONNECTIVITY] 
# [HIGH_GENE_COUNT] [MED_TOTAL_COUNT] [LOW_MITO] [MED_RIBO] [LOW_CANCER] [END_SPOT] [SEP]
#
# This represents:
# - Start of sequence token [CLS]
# - Start of spot marker [SPOT] 
# - Spatial location at grid position (10,15)
# - Tissue type is brain
# - Has 5 neighboring cells
# - Medium connectivity with neighbors
# - High number of genes expressed
# - Medium total RNA count
# - Low mitochondrial percentage
# - Medium ribosomal percentage  
# - Low cancer score
# - End of spot marker [END_SPOT]
# - End of sequence token [SEP]

# Example usage:
# Create new tokenizer
# tokenizer = IncrementalTissueTokenizer()
# tokenizer.add_tissue('brain', brain_adata, brain_processed) # Train on first tissue
# tokenizer.add_tissue('liver', liver_adata, liver_processed) # Add another tissue
# tokenizer.save() # Save trained tokenizer

# Load existing tokenizer
# tokenizer = IncrementalTissueTokenizer()
# tokenizer.load_existing_tokenizer() # Load previously saved tokenizer

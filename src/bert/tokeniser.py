import numpy as np
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from rich.progress import track
from rich.console import Console
from sklearn.preprocessing import StandardScaler
import torch

console = Console()


def create_spatial_tokens(adata):
    """Create rich spatial and expression tokens from AnnData object"""
    console.rule("[bold blue]Creating Spatial Tokens")
    console.print(f"[yellow]Processing dataset with {adata.n_obs} cells[/yellow]")

    with console.status("[bold green]Extracting data matrices...") as status:
        # Get expression and spatial data
        X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
        spatial_coords = adata.obsm['spatial']
        clusters = adata.obs['clusters']

        # Print original spatial ranges
        x_range = (spatial_coords[:, 0].min(), spatial_coords[:, 0].max())
        y_range = (spatial_coords[:, 1].min(), spatial_coords[:, 1].max())
        console.print(f"[blue]Original spatial ranges:")
        console.print(f"  X: {x_range}")
        console.print(f"  Y: {y_range}")

        # Normalize coordinates to [0, 1] - use the same normalization as in data_prep
        x_norm = (spatial_coords[:, 0] - x_range[0]) / (x_range[1] - x_range[0])
        y_norm = (spatial_coords[:, 1] - y_range[0]) / (y_range[1] - y_range[0])
        spatial_coords = np.column_stack((x_norm, y_norm))

        # Print normalized ranges
        console.print(f"[blue]Normalized spatial ranges [0,1]:")
        console.print(f"  X: ({spatial_coords[:, 0].min():.3f}, {spatial_coords[:, 0].max():.3f})")
        console.print(f"  Y: ({spatial_coords[:, 1].min():.3f}, {spatial_coords[:, 1].max():.3f})")

        # Normalize expression data
        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X)

        # Calculate spatial statistics
        x_coords = spatial_coords[:, 0]
        y_coords = spatial_coords[:, 1]
        x_bins = np.linspace(x_coords.min(), x_coords.max(), 20)
        y_bins = np.linspace(y_coords.min(), y_coords.max(), 20)

        console.print(f"[bold cyan]Data Shape: {X.shape}")
        console.print(f"[bold cyan]Unique clusters: {len(set(clusters))}")

    def create_spatial_bucket(coords):
        """Create spatial buckets with 20 divisions"""
        x, y = coords
        x_bucket = f'x_{int(x * 20)}'  # 20 buckets for normalized coords
        y_bucket = f'y_{int(y * 20)}'
        return f'[ZONE_{x_bucket}_{y_bucket}]'

    def encode_expression(expr_val):
        """Encode expression values with granularity"""
        if expr_val <= -2:
            return "expr_very_low"
        elif expr_val <= -1:
            return "expr_low"
        elif expr_val <= 0:
            return "expr_medium_low"
        elif expr_val <= 1:
            return "expr_medium"
        elif expr_val <= 2:
            return "expr_medium_high"
        else:
            return "expr_high"

    tokens_list = []
    console.print("\n[bold cyan]Starting cell tokenization...")

    for i in track(range(len(adata)), description="[green]Tokenizing cells"):
        cell_tokens = [
            "[CELL]",
            create_spatial_bucket(spatial_coords[i]),
            f'[CLUSTER_{clusters.iloc[i]}]',
            "[GENES]"
        ]

        # Add gene expression tokens
        for j, gene_name in enumerate(adata.var_names):
            gene_name = gene_name.lower().replace('-', '_').replace('.', '_').replace(' ', '_')
            expr_level = encode_expression(X_normalized[i, j])
            cell_tokens.extend([gene_name, expr_level])

        cell_tokens.append("[END]")
        tokens_list.append(" ".join(cell_tokens))

        # Progress updates
        if (i + 1) % 1000 == 0:
            console.print(f"[dim]Processed {i + 1}/{len(adata)} cells[/dim]")

    console.print(f"\n[bold green]✓ Successfully tokenized {len(adata)} cells")
    return tokens_list, spatial_coords, X_normalized


def configure_tokenizer():
    """Create tokenizer with expanded vocab and special tokens"""
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))

    special_tokens = [
        # Basic BERT tokens
        "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
        # Structure tokens
        "[CELL]", "[END]", "[GENES]",
        # Expression levels
        "expr_very_low", "expr_low", "expr_medium_low",
        "expr_medium", "expr_medium_high", "expr_high",
        # Generated tokens
        *[f"[CLUSTER_{i}]" for i in range(20)],
        *[f"[ZONE_x_{i}]" for i in range(20)],
        *[f"[ZONE_y_{i}]" for i in range(20)]
    ]

    trainer = trainers.WordPieceTrainer(
        vocab_size=50000,
        special_tokens=special_tokens,
        min_frequency=2
    )

    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return tokenizer, trainer


def prep_st_data(adata, force_retrain=False):
    """Process spatial transcriptomics data and prepare tokenizer"""
    import os
    console.rule("[bold blue]Preparing Spatial Transcriptomics Data")

    # Load existing tokenizer if available
    if not force_retrain and os.path.exists('st_tokenizer.json'):
        console.print("[bold yellow]Loading existing tokenizer...")
        tokenizer = PreTrainedTokenizerFast(tokenizer_file="st_tokenizer.json")
        tokenizer.pad_token = "[PAD]"
        tokenizer.mask_token = "[MASK]"
        tokenizer.sep_token = "[SEP]"
        tokenizer.cls_token = "[CLS]"
        tokenizer.unk_token = "[UNK]"
        return tokenizer, None, None

    # Create tokens and process data
    console.print("[bold yellow]Creating spatial tokens...")
    tokens, spatial_coords, X_normalized = create_spatial_tokens(adata)

    # Save tokens
    with console.status("[bold green]Saving tokens..."):
        with open('gene_expression_corpus.txt', 'w') as f:
            for line in track(tokens, description="Writing tokens"):
                f.write(line + '\n')

    # Configure and train tokenizer
    console.print("[bold yellow]Configuring tokenizer...")
    tokenizer, trainer = configure_tokenizer()

    with console.status("[bold cyan]Training tokenizer..."):
        tokenizer.train(files=["gene_expression_corpus.txt"], trainer=trainer)
        tokenizer.save("st_tokenizer.json")

    # Create HuggingFace tokenizer
    console.print("[bold magenta]Converting to HuggingFace tokenizer...")
    hf_tokenizer = PreTrainedTokenizerFast(tokenizer_file="st_tokenizer.json")

    # Set special tokens
    hf_tokenizer.pad_token = '[PAD]'
    hf_tokenizer.mask_token = '[MASK]'
    hf_tokenizer.sep_token = '[SEP]'
    hf_tokenizer.cls_token = '[CLS]'
    hf_tokenizer.unk_token = '[UNK]'

    console.print("[bold green]✓ Data preparation completed!")
    return hf_tokenizer, spatial_coords, X_normalized

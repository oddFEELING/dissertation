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

console = Console()


def create_spatial_tokens(adata: AnnData, processed_data: Dict) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Create tokens for spatial transcriptomics data.

    :param adata: AnnData object containing spatial transcriptomics data
    :param processed_data: Dictionary containing processed features and distances
    :return: Tuple containing (token sequences, normalised coordinates, normalised expression matrix)
    """
    console.print('[bold]Creating spatial tokens...')

    # Extract data
    spatial_coords = adata.obsm['spatial']
    cancer_scores = adata.obs['cancer_score']
    distances = processed_data['distance_info']['distances']

    # Normalise expression matrix
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    scaler = StandardScaler()
    X_normalised = scaler.fit_transform(X)

    # Normilise coordinates to [0, 20] for discrete spatial buckets
    # Creates a 20x20 grid of spatial locations
    x_norm = np.floor(20 * (spatial_coords[:, 0] - spatial_coords[:, 0].min()) /
                      (spatial_coords[:, 0].max() - spatial_coords[:, 0].min()))
    y_norm = np.floor(20 * (spatial_coords[:, 1] - spatial_coords[:, 1].min()) /
                      (spatial_coords[:, 1].max() - spatial_coords[:, 1].min()))

    def encode_cancer_score(score: float) -> str:
        """Convert cancer score to discrete token"""
        if score <= 0.33:
            return "cancer_low"
        elif score <= 0.66:
            return "cancer_med"
        else:
            return "cancer_high"

    def encode_expression_state(spot_expr: np.ndarray) -> List[str]:
        """
        Encode overall expression state for a spot

        :param spot_expr: Expression values for a spot
        :return: List of expression state tokens
        """
        # Get high/low expression genes
        high_expr = np.where(spot_expr > 1.0)[0]
        low_expr = np.where(spot_expr < -1.0)[0]

        tokens = []
        # Add overall expression state
        tokens.append(f"[HIGH_EXPR_{len(high_expr)}]")
        tokens.append(f"[LOW_EXPR_{len(low_expr)}]")

        # Check if cancer genes info is available
        if 'cancer_genes' in processed_data['feature_info']:
            cancer_genes_idx = [i for i, gene in enumerate(adata.var_names)
                                if gene in processed_data['feature_info']['cancer_genes']]
            high_cancer = len(set(high_expr) & set(cancer_genes_idx))
            tokens.append(f"[HIGH_CANCER_{high_cancer}]")
        else:
            tokens.append("[HIGH_CANCER_0]")  # Default if no cancer genes info

        return tokens

    tokens_list = []
    for idx in track(range(len(adata)), description="Tokenising spots"):
        # Start spot sequence
        spot_tokens = ["[SPOT]"]

        # Add spatial location
        spot_tokens.append(f"[SPATIAL_X_{int(x_norm[idx])}]")
        spot_tokens.append(f"[SPATIAL_Y_{int(y_norm[idx])}]")

        # Add cancer score
        spot_tokens.append(f"[{encode_cancer_score(cancer_scores[idx])}]")

        # Add neighbour count
        n_neighbours = (distances[idx] > 0).sum()
        spot_tokens.append(f"[NEIGHBOURS_{n_neighbours}]")

        # Add expression states
        spot_tokens.extend(encode_expression_state(X_normalised[idx]))

        spot_tokens.append("[END_SPOT]")
        tokens_list.append(" ".join(spot_tokens))

    return tokens_list, np.column_stack((x_norm, y_norm)), X_normalised


def configure_tokeniser() -> Tuple[Tokenizer, trainers.WordPieceTrainer]:
    """
    Configure tokeniser with special tokens and settings.

    :return: Tuple of tokeniser and trainer objects
    """
    tokeniser = Tokenizer(models.WordPiece(unk_token="[UNK]"))

    # Define special tokens
    special_tokens = [
        # Basic tokens
        "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",

        # Structure tokens
        "[SPOT]", "[END_SPOT]",

        # Cancer levels
        "[cancer_low]", "[cancer_med]", "[cancer_high]",

        # Spatial tokens (20x20 grid)
        *[f"[SPATIAL_X_{i}]" for i in range(20)],
        *[f"[SPATIAL_Y_{i}]" for i in range(20)],

        # Neighbour counts (0-10)
        *[f"[NEIGHBOURS_{i}]" for i in range(11)],

        # Expression state tokens (limit to reasonable ranges)
        *[f"[HIGH_EXPR_{i}]" for i in range(51)],  # 0-50 highly expressed genes
        *[f"[LOW_EXPR_{i}]" for i in range(51)],  # 0-50 lowly expressed genes
        *[f"[HIGH_CANCER_{i}]" for i in range(31)]  # 0-30 highly expressed cancer genes
    ]

    trainer = trainers.WordPieceTrainer(
        vocab_size=30000,
        special_tokens=special_tokens,
        min_frequency=2
    )

    tokeniser.pre_tokenizer = pre_tokenizers.Whitespace()
    return tokeniser, trainer


def prepare_spatial_tokeniser(adata: AnnData,
                              processed_data: Dict,
                              force_retrain: bool = False) -> Tuple[PreTrainedTokenizerFast, np.ndarray, np.ndarray]:
    """
    Prepare tokeniser for spatial transcriptomics data.

    :param adata: AnnData object containing spatial transcriptomics data
    :param processed_data: Dictionary containing processed features and distances
    :param force_retrain: Whether to force retraining of tokeniser
    :return: Tuple containing (tokeniser, spatial coordinates, normalised expression matrix)
    """
    console.print('Preparing ST tokeniser')

    # Load existing tokeniser if available (No force retrain)
    if not force_retrain and os.path.exists("tokens/st_tokenizer.json"):
        console.print('[bold yellow]Loading existing tokeniser... [/]')
        tokeniser = PreTrainedTokenizerFast(tokenizer_file="tokens/st_tokenizer.json")
        tokeniser.pad_token = "[PAD]"
        tokeniser.mask_token = "[MASK]"
        tokeniser.sep_token = "[SEP]"
        tokeniser.cls_token = "[CLS]"
        tokeniser.unk_token = "[UNK]"
        return tokeniser, None, None

    # Create tokens and coordinates
    console.print('[yellow]Creating spatial tokens...[/]')
    tokens, spatial_coords, X_normalised = create_spatial_tokens(adata, processed_data)

    # Save tokens to file for training
    with console.status('[bold green]Saving tokens...[/]'):
        output_path = Path('tokens')
        output_path.mkdir(exist_ok=True)
        with open(output_path / "st_corpus.txt", 'w') as f:
            for line in tokens:
                f.write(line + '\n')

    # Configure and train tokeniser
    console.print('Configuring and training the tokeniser...')
    tokeniser, trainer = configure_tokeniser()
    tokeniser.train(files=["tokens/st_corpus.txt"], trainer=trainer)

    # Save trained tokeniser
    tokeniser.save('tokens/st_tokenizer.json')

    # Create Huggingface tokeniser
    console.print("Creating huggingface tokeniser...")
    hf_tokeniser = PreTrainedTokenizerFast(tokenizer_file="tokens/st_tokenizer.json")

    # Set special tokens
    hf_tokeniser.pad_token = '[PAD]'
    hf_tokeniser.mask_token = '[MASK]'
    hf_tokeniser.sep_token = '[SEP]'
    hf_tokeniser.cls_token = '[CLS]'
    hf_tokeniser.unk_token = '[UNK]'

    # Print token statistics
    console.print(f"[green]Vocabulary size: {len(hf_tokeniser.get_vocab())}")
    console.print(f"[green]Number of spots tokenised: {len(tokens)}")

    # Add token type information to tokeniser
    hf_tokeniser.token_categories = {
        'spatial': [f"SPATIAL_X_{i}" for i in range(20)] + [f"SPATIAL_Y_{i}" for i in range(20)],
        'cancer': ['cancer_low', 'cancer_med', 'cancer_high'],
        'neighbours': [f"NEIGHBOURS_{i}" for i in range(11)],
        'expression': [f"HIGH_EXPR_{i}" for i in range(51)] +
                      [f"LOW_EXPR_{i}" for i in range(51)] +
                      [f"HIGH_CANCER_{i}" for i in range(31)]
    }

    return hf_tokeniser, spatial_coords, X_normalised

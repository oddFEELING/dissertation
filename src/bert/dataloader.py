import torch
from torch.utils.data import Dataset, DataLoader, random_split
from rich.console import Console
import numpy as np
from rich.progress import track

console = Console()


class SpatialTranscriptomicsDataset(Dataset):
    """
    Dataset for spatial transcriptomics data with cancer scores
    """

    def __init__(self, adata, processed_data, tokenizer, indices=None):
        """
        Initialize dataset with processed spatial transcriptomics data

        :param adata: AnnData object
        :param processed_data: Dictionary containing processed features and distances
        :param tokenizer: HuggingFace tokenizer
        :param indices: Optional indices for train/val split
        """
        self.adata = adata
        self.combined_features = processed_data['combined_features']
        self.distances = processed_data['distance_info']['distances']
        self.tokenizer = tokenizer
        self.indices = indices if indices is not None else range(len(adata))

        # Create text tokens for each spot
        console.print("[yellow]Creating text tokens for dataset...[/]")
        self.text_tokens = []

        for idx in track(self.indices, description="Processing spots"):
            # Get spot coordinates
            spatial_x = int(np.floor(20 * (adata.obsm['spatial'][idx, 0] - adata.obsm['spatial'][:, 0].min()) /
                                     (adata.obsm['spatial'][:, 0].max() - adata.obsm['spatial'][:, 0].min())))
            spatial_y = int(np.floor(20 * (adata.obsm['spatial'][idx, 1] - adata.obsm['spatial'][:, 1].min()) /
                                     (adata.obsm['spatial'][:, 1].max() - adata.obsm['spatial'][:, 1].min())))

            # Create token sequence
            spot_tokens = [
                "[CLS]",
                "[SPOT]",
                f"[SPATIAL_X_{spatial_x}]",
                f"[SPATIAL_Y_{spatial_y}]"
            ]

            # Add cancer score token
            cancer_score = adata.obs['cancer_score'].iloc[idx]
            if cancer_score <= 0.33:
                spot_tokens.append("[cancer_low]")
            elif cancer_score <= 0.66:
                spot_tokens.append("[cancer_med]")
            else:
                spot_tokens.append("[cancer_high]")

            # Add number of neighbours
            n_neighbours = (self.distances[idx] > 0).sum()
            spot_tokens.append(f"[NEIGHBOURS_{min(n_neighbours, 10)}]")

            # Add end tokens
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

            self.text_tokens.append({
                'input_ids': encoded['input_ids'].squeeze(),
                'attention_mask': encoded['attention_mask'].squeeze()
            })

        console.print(f"[green]✓[/green] Created dataset with {len(self.indices)} samples")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]

        return {
            'input_ids': self.text_tokens[idx]['input_ids'],
            'attention_mask': self.text_tokens[idx]['attention_mask'],
            'features': self.combined_features[real_idx],
            'distances': self.distances[real_idx]
        }


def create_st_dataloaders(
        adata,
        processed_data,
        tokenizer,
        batch_size: int = 32,
        val_split: float = 0.2,
        **kwargs
) -> tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders for spatial transcriptomics data

    :param adata: AnnData object
    :param processed_data: Dictionary containing processed features and distances
    :param tokenizer: Custom tokenizer
    :param batch_size: Batch size for dataloaders
    :param val_split: Validation split ratio
    :return: Tuple of (train_loader, val_loader)
    """
    # Create indices for train/val split
    indices = np.random.permutation(len(adata))
    val_size = int(len(adata) * val_split)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    # Create datasets
    train_dataset = SpatialTranscriptomicsDataset(
        adata=adata,
        processed_data=processed_data,
        tokenizer=tokenizer,
        indices=train_indices,
        **kwargs
    )

    val_dataset = SpatialTranscriptomicsDataset(
        adata=adata,
        processed_data=processed_data,
        tokenizer=tokenizer,
        indices=val_indices,
        **kwargs
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    console.print(f"Created dataloaders:")
    console.print(f"  Train: {len(train_dataset)} samples")
    console.print(f"  Validation: {len(val_dataset)} samples")

    return train_loader, val_loader

from torch.utils.data import DataLoader, Dataset, random_split
from typing import List, Dict, Tuple
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import json
import random
from src.bert.tokenizer.tokeniser import TissueTokenizer


class SpatialMLMDataset(Dataset):
    def __init__(self,
                 sequences: List[str],
                 tokenizer,
                 device: str = 'cuda',
                 mask_probability: float = 0.15,
                 ):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.mask_probability = mask_probability
        self.device = device

    def __len__(self):
        return len(self.sequences)

    def mask_sequence(self, sequence: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply masking to sequence for MLM"""
        tokens = sequence.split()
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        labels = [-100] * len(input_ids)
        attention_mask = [1] * len(input_ids)

        # Find maskable positions (avoid special tokens)
        maskable_positions = []
        for i, token in enumerate(tokens):
            if not token.startswith('[') and not token.endswith('##'):
                maskable_positions.append(i)

        # Randomly mask tokens
        if maskable_positions:
            n_masks = max(1, int(len(maskable_positions) * self.mask_probability))
            mask_positions = random.sample(maskable_positions, n_masks)

            for pos in mask_positions:
                labels[pos] = input_ids[pos]
                input_ids[pos] = self.tokenizer.mask_token_id

        return (torch.tensor(input_ids, devicde=self.device),
                torch.tensor(attention_mask, device=self.device),
                (torch.tensor(labels, device=self.device)))


class SpatialDataModule:
    def __init__(self,
                 corpus_file: str,
                 tissue_tokenizer: TissueTokenizer,
                 device: str = 'cuda',
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 mask_probability: float = 0.15
                 ):
        print('\n\n------------ Starting Data Loader  --\n')
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers

        with open(corpus_file, 'r') as f:
            sequences = [line.strip() for line in f if line.strip()]

        # Split data
        train_val_size = int(len(sequences) * (train_ratio + val_ratio))
        test_size = len(sequences) - train_val_size

        train_val_seqs, self.test_seqs = train_test_split(
            sequences,
            test_size=test_size,
            random_state=42
        )

        val_size = int(len(sequences) * val_ratio)
        self.train_seqs, self.val_seqs = train_test_split(
            train_val_seqs,
            test_size=val_size,
            random_state=42
        )

        # Generate Datasets
        self.train_dataset = SpatialMLMDataset(
            self.train_seqs, tissue_tokenizer, self.device, mask_probability
        )
        self.val_dataset = SpatialMLMDataset(
            self.val_seqs, tissue_tokenizer, self.device, mask_probability
        )
        self.test_dataset = SpatialMLMDataset(
            self.test_seqs, tissue_tokenizer, self.device, mask_probability
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )

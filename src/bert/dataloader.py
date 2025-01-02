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

        # Debugging logs
        # tokens = self.sequences[0].split()
        # input_ids = self.tokenizer.tokenizer.convert_tokens_to_ids(tokens)
        # print("\nFirst few token IDs:", input_ids[:10])
        # # Print first sequence for debugging
        # print("\nExample sequence:")
        # print(self.sequences[0])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sequence = self.sequences[idx]
        # input_ids, attention_mask, labels = self.mask_sequence(sequence)
        return self.mask_sequence(sequence)

    def mask_sequence(self, sequence: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply masking to sequence for MLM"""
        # Use the tokenizer for encoding
        encoding = self.tokenizer.tokenizer(
            sequence,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        # Add token_type_ids (all zeros for single sequence)
        token_type_ids = torch.zeros_like(input_ids)

        # Create labels (copy of input_ids)
        labels = input_ids.clone()

        # Find maskable positions (avoid special tokens)
        special_tokens_mask = torch.tensor(
            self.tokenizer.tokenizer.get_special_tokens_mask(input_ids, already_has_special_tokens=True)
        ).bool()

        # Create probability matrix for masking
        probability_matrix = torch.full(labels.shape, self.mask_probability)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        # Mask tokens
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # Replace masked input tokens with [MASK] token
        input_ids[masked_indices] = self.tokenizer.tokenizer.mask_token_id

        return {
            'input_ids': input_ids.to(self.device),
            'attention_mask': attention_mask.to(self.device),
            'token_type_ids': token_type_ids.to(self.device),
            'labels': labels.to(self.device)
        }


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

        print(f'--> Loading data with {self.num_workers} workers')

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
            num_workers=self.num_workers,
            collate_fn=collate_batch,
            multiprocessing_context='spawn',
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_batch,
            multiprocessing_context='spawn',
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_batch,
            multiprocessing_context='spawn',
            persistent_workers=True,
        )


def collate_batch(batch):
    """Custom collate function to handle padding"""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'token_type_ids': torch.stack([item['token_type_ids'] for item in batch]),
    }

from tokenize import tokenize

import torch
from pycparser.ply.yacc import token
from torch.utils.data import DataLoader
import logging
import wandb
from pathlib import Path
import numpy as np
from rich.pretty import pprint
from src.bert.dataloader import SpatialDataModule
from src.bert.tokenizer.corpus_gen import CorpusGenerator
from src.bert.tokenizer.tokeniser import TissueTokenizer

from src.bert.data_prep import DataPrep

dataprep = DataPrep('../ingest/lung_cancer_results.h5ad')
features = dataprep.prepare_data()

generator = CorpusGenerator()

tokenizer = TissueTokenizer()

tokenizer.add_gene_tokens(dataprep.adata.var_names.unique().tolist())
tokenizer.load_tokenizer('tokenizer/_internal')

data_module = SpatialDataModule(
    corpus_file='tokenizer/corpus.txt',
    tissue_tokenizer=tokenizer,
    batch_size=32,
    num_workers=4,
    device='cuda'
)

# Get the dataloaders
train_loader = data_module.train_dataloader()
val_dataloader = data_module.val_dataloader()
test_dataloader = data_module.test_dataloader()

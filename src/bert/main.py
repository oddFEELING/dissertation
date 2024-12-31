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
from src.bert.experiments.MLM import MLMTrainer
import torch.multiprocessing as mp

from src.bert.data_prep import DataPrep


def main():
    dataprep = DataPrep('../ingest/lung_cancer_results.h5ad')
    # features = dataprep.prepare_data()
    #
    # generator = CorpusGenerator()
    # sequences = generator.generate_corpus(features, 'tokenizer/corpus.txt')

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

    trainer = MLMTrainer(
        model_name='bert-base-uncased',
        data_module=data_module,
        device='cuda'
    )

    trainer.train(
        num_epochs=3,
        output_dir='outputs',
        save_steps=1000
    )


if __name__ == '__main__':
    mp.freeze_support()
    main()

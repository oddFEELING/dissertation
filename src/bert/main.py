from tokenize import tokenize
import torch
from pycparser.ply.yacc import token
from torch.utils.data import DataLoader
import logging
import wandb
from pathlib import Path
import numpy as np
from rich.pretty import pprint
from rich.progress import track
from src.bert.dataloader import SpatialDataModule
from src.bert.tokenizer.corpus_gen import CorpusGenerator
from src.bert.tokenizer.tokeniser import TissueTokenizer
from src.bert.experiments.MLM import MLMTrainer
from src.bert.data_prep import DataPrep, EfficientDataLoader


def main():
    # Feature extraction
    dataprep = DataPrep('../figures/brain_cancer_figs/brain_cancer_results.h5ad')
    # dataprep.prepare_data()

    # Load features efficiently
    # loader = EfficientDataLoader(batch_size=1000)
    # features = loader.load_all_features('tokenizer/data_prep.json')

    # generator = CorpusGenerator()
    # generator.generate_corpus(features, 'tokenizer/corpus.txt')

    tokenizer = TissueTokenizer()

    tokenizer.add_gene_tokens(dataprep.adata.var_names.unique().tolist())
    tokenizer.load_tokenizer('./tokenizer/_internal')
    test = tokenizer.validate_token_sequence(
        '[CLS] [TISSUE] brain_cancer [SPATIAL] -16.49 82.62 [CANCER] 0.68 [REACT] 40.54 [GENE] gene_IGHG3 0.20 15.35 [GENE] gene_TAS2R1 0.28 11.40 [GENE] gene_MYO1A 0.27 11.27 [GENE] gene_OFD1 0.37 7.62 [GENE] gene_CXCL8 0.60 4.17 [NOT_BORDER] [NEIGHBOR] -17.39 84.18 1.8 2.09 99.6 [NEIGHBOR] -15.60 84.18 1.8 1.05 79.48 [NEIGHBOR] -14.70 82.62 1.8 0.0 72.69 [NEIGHBOR] -17.39 81.04 1.82 -2.09 71.35 [NEIGHBOR] -18.29 82.62 1.79 3.14 70.0 [MITO_HIGH] [SEP]')

    if test:
        print('tokenizer tests passed')

        # instantiate data loader module
        data_module = SpatialDataModule(
            corpus_file='tokenizer/corpus.txt',
            tissue_tokenizer=tokenizer,
            batch_size=32,
            num_workers=3,
            device='cuda'
        )

        # Create model trainer instance
        trainer = MLMTrainer(
            model_name='bert-base-uncased',
            data_module=data_module,
            device='cuda'
        )

        pprint(f'Model trainer instance created: {trainer}')

        # Train the model
        trainer.train(
            start_epoch=4,
            num_epochs=5,
            output_dir='outputs',
            save_steps=1000,
            resume_from_checkpoint='./outputs/checkpoint-1000'
        )
    else:
        print('Tokenizer tests failed')


if __name__ == '__main__':
    main()

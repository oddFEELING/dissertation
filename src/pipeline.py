import numpy as np
import scanpy as sc
import torch
from sklearn.metrics.pairwise import euclidean_distances
from anndata import AnnData
from rich.pretty import pprint
from sklearn.preprocessing import MinMaxScaler
from rich.progress import track
from src.bert.main import main
import torch.multiprocessing as mp
from src.ingest import Ingest

file = sc.read_h5ad('./ingest/lung_cancer_results.h5ad')

pprint(len(set(file.var_names.unique())))


if __name__ == '__main__':
    mp.freeze_support()
    main()

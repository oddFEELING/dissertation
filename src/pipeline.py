import numpy as np
import scanpy as sc
import torch
from sklearn.metrics.pairwise import euclidean_distances
from anndata import AnnData
from rich.pretty import pprint
from sklearn.preprocessing import MinMaxScaler
from rich.progress import track

file = sc.read_h5ad('./ingest/lung_cancer_results.h5ad')

pprint(len(set(file.var_names.unique())))

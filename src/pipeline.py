import scanpy as sc
from rich.pretty import pprint

file = sc.read_h5ad('./ingest/brain_cancer_results.h5ad')

pprint(file)

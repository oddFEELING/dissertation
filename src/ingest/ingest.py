from statsmodels.graphics.tukeyplot import results

from src.ingest import load_data, ingest_config, perform_qc, \
    cluster_data, PyPlotConfig, calc_cancer_score, get_high_cancer_regions, \
    get_top_cancer_genes
from rich.pretty import pprint
import scanpy as sc

from src.pipeline import Pipeline

config = PyPlotConfig()
# adata = sc.read_h5ad('./results.h5ad')
state = ingest_config(sc_verbosity=3, pyplot_config=config, figures_dir='figures')

# pipeline = (Pipeline(state, None)
#             | load_data(path='../data/human_lung_cancer')
#             | perform_qc(mito_threshold=20)
#             | cluster_data()
#             | calc_cancer_score(file_name='results.h5ad')
#             )

# adata, state = pipeline.run()

adata, state = load_data(state, '../data/human_lung_cancer')
adata, state = perform_qc(state, adata, mito_threshold=20)
adata, state = cluster_data(state, adata)
adata, state = calc_cancer_score(state, adata, file_name='results.h5ad')
adata, scores, _ = get_high_cancer_regions(state, adata)

adata, df, _ = get_top_cancer_genes(state, adata, high_scoring_mask=scores)

pprint(adata)

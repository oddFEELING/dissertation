"""
This file is just the analysis pipeline from script.py in the root folder
for easier pipeline building.
"""

import os
from pathlib import Path
from typing import List, Optional
import numpy as np
import scanpy as sc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import rich.pretty as pprint
from anndata import AnnData
from matplotlib.pyplot import title, savefig
from networkx.algorithms.bipartite.basic import color
from pywin.framework.interact import valueFormatOutputError
from rich import print as rprint
from rich.table import Table
from rich.console import Console
from pydantic import BaseModel
from enum import Enum, auto

console = Console()


class TissueType(str, Enum):
    """
    Keep track of tissue types added to the tokenizer.
    This is just an example list to show how to use enums.
    """
    UNKNOWN = "unknown"
    LUNG_CANCER = "lung_cancer"
    BREAST_CANCER = "breast_cancer"
    BRAIN_CANCER = "brain_cancer"
    COLON_CANCER = "colon_cancer"
    HEALTHY_LUNG = "healthy_lung"
    HEALTHY_BREAST = "healthy_breast"
    HEALTHY_BRAIN = "healthy_brain"
    HEALTHY_COLON = "healthy_colon"


class IngestState(BaseModel):
    """
    State object for the ingestion pipeline.

    :param actions_taken: List of actions performed during ingestion
    :param figures_dir: Directory to save figures
    :param cluster_key: Key for clustering results
    :param tissue_type: Type of tissue being analyzed
    :param adata: AnnData object containing the data
    """
    actions_taken: List[str] = []
    figures_dir: Path
    cluster_key: str = 'clusters'
    tissue_type: TissueType = TissueType.UNKNOWN
    adata: Optional[AnnData] = None

    model_config = {
        "arbitrary_types_allowed": True,  # Allow AnnData and other complex types
        "validate_assignment": True  # Validate data on assignment
    }


class PyPlotConfig(BaseModel):
    """
    Configuration for matplotlib pyplot settings.

    :param font_size: Default font size for all text
    :param title_size: Title font size
    :param label_size: Axis label font size
    :param xtick_size: X-axis tick label font size
    :param ytick_size: Y-axis tick label font size
    :param legend_size: Legend font size
    """
    font_size: int = 12
    title_size: int = 14
    label_size: int = 8
    xtick_size: int = 8
    ytick_size: int = 8
    legend_size: int = 8


def ingest_config(
        sc_verbosity: int,
        pyplot_config: PyPlotConfig,
        figures_dir: str,
        tissue_type: TissueType = TissueType.UNKNOWN
) -> IngestState:
    """
    Applies default parameters to the ingestion pipeline.

    :param sc_verbosity: Scanpy verbosity level
    :param pyplot_config: Initial configuration for pyplot
    :param figures_dir: Directory to store figures
    :param tissue_type: Type of tissue being analyzed
    :return: Initialized state for ingestion pipeline
    """
    console.print("----- Initialising ingestion pipeline -----\n\n")
    sc.settings.verbosity = sc_verbosity
    plot_conf = pyplot_config.dict()
    plt.rcParams.update({
        'font.size': plot_conf['font_size'],
        'axes.titlesize': plot_conf['title_size'],
        'axes.labelsize': plot_conf['label_size'],
        'xtick.labelsize': plot_conf['xtick_size'],
        'ytick.labelsize': plot_conf['ytick_size'],
        'legend.fontsize': plot_conf['legend_size']
    })
    ingest_state = IngestState(
        actions_taken=["Initialized ingestion pipeline"],
        figures_dir=Path(figures_dir),
        tissue_type=tissue_type
    )

    return ingest_state


def load_data(state: IngestState, path: str) -> IngestState:
    """
    Loads dataset from given path
    :param state: Pipeline state object
    :param path: Path to ST Dataset
    :return: State object and loaded anndata object
    """

    console.print("\n\n----- Loading data -----\n\n")
    os.makedirs(state.figures_dir, exist_ok=True)

    # Set scanpy figure path to match our figures directory
    sc.settings.figdir = state.figures_dir

    table = Table(title='Data loader')
    table.add_column('Metric')
    table.add_column('Value', justify='center')
    console.print(f'Loading dataset from {path}')
    # Load data
    adata = sc.read_visium(path)
    adata.var_names_make_unique()

    # Add tissue type to adata
    adata.obs['tissue_type'] = state.tissue_type

    # Get Mito and Ribo genes
    adata.var['mito'] = adata.var.index.str.startswith('MT')  # MT for humans
    ribo_url = "http://software.broadinstitute.org/gsea/msigdb/download_geneset.jsp?geneSetName=KEGG_RIBOSOME&fileType=txt"
    ribo_genes = pd.read_table(ribo_url, skiprows=2, header=None)
    adata.var['ribo'] = adata.var_names.isin(ribo_genes[0].values)

    # Calculate qc metrics
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mito', 'ribo'], inplace=True)

    # Plot figures
    # Cell counts per spot
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(adata.obs['total_counts'], kde=False, ax=ax)
    ax.set_title('Total Cell Counts Per Spot')
    ax.set_xlabel('Cell counts')
    ax.set_ylabel('Number of spots')
    fig.savefig(state.figures_dir / 'cell_counts_per_spot.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Genes per spot
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(adata.obs['n_genes_by_counts'], kde=False, ax=ax)
    ax.set_title('Genes Per Spot')
    ax.set_xlabel('Genes')
    fig.savefig(state.figures_dir / 'genes_per_spot.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Outliers with violin
    sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mito', 'pct_counts_ribo'],
                 jitter=0.4,
                 multi_panel=True,
                 save='_qc_violin_plots.png'
                 )
    plt.close()

    table.add_row('Cell count', str(len(adata.obs_names)))
    table.add_row('Gene count', str(len(adata.var_names)))
    console.print(table)
    state.actions_taken.append('Loaded data')
    state.adata = adata

    return state


def perform_qc(state: IngestState, mito_threshold=20, ribo_threshold=20, cell_threshold=30,
               gene_threshold=5, n_top_genes: int = 5000, norm_value=1e4, log_transform=True) -> IngestState:
    """"""
    console.print("\n\n----- Performing QC -----\n\n")
    table = Table(title='Data QC')
    table.add_column('Metric')
    table.add_column('Value', justify='center')
    # Perform quality control
    adata = state.adata
    sc.pp.filter_cells(adata, min_counts=cell_threshold)
    sc.pp.filter_genes(adata, min_cells=gene_threshold)
    adata = adata[adata.obs['pct_counts_mito'] < mito_threshold].copy()
    adata = adata[adata.obs['pct_counts_ribo'] < ribo_threshold].copy()
    sc.pp.normalize_total(adata, target_sum=norm_value, inplace=True)
    if log_transform:
        sc.pp.log1p(adata)
    # freeze current state of adata into new field of anndata
    adata.raw = adata
    sc.pp.highly_variable_genes(adata, flavor='seurat', n_top_genes=n_top_genes, inplace=True, subset=True)
    adata = adata[:, adata.var.highly_variable]
    table.add_row('Remaining Cell count', str(len(adata.obs_names)))
    table.add_row('Remaining Gene count', str(len(adata.var_names)))
    console.print(table)
    sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mito', 'pct_counts_ribo'])
    sc.pp.scale(adata, max_value=10)
    state.actions_taken.append('Performed QC')
    state.adata = adata
    # Run pca to reduce dimensionality
    return state


def cluster_data(state: IngestState, n_pcs=15, n_neighbors=30,
                 cluster_res=0.5) -> IngestState:
    """
    Perform clustering analysis on the data
    
    :param state: Pipeline state object
    :param n_pcs: Number of principal components to use
    :param n_neighbors: Fixed number of neighbors to use (default: 30)
    :param cluster_res: Resolution for clustering
    :return: Updated state object
    """
    console.print("\n\n----- Performing Clustering -----\n\n")
    adata = state.adata

    # Ensure scanpy uses the correct figure directory
    sc.settings.figdir = state.figures_dir

    # PCA
    sc.pp.pca(adata, random_state=42)
    sc.pl.pca_variance_ratio(
        adata,
        log=True,
        n_pcs=n_pcs,
        show=False,
        save='_pca_variance.png'
    )
    plt.close()

    # Store the fixed neighbor count in adata for reference
    adata.uns['spatial_neighbors'] = {
        'n_neighbors': n_neighbors
    }

    # Neighbors and UMAP with fixed n_neighbors
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    sc.tl.umap(adata)

    # Clustering
    sc.tl.louvain(
        adata,
        key_added=state.cluster_key,
        flavor='igraph',
        directed=False,
        resolution=cluster_res,
    )

    # Convert clusters column to category
    if adata.obs[state.cluster_key].dtype != 'category':
        adata.obs[state.cluster_key] = adata.obs[state.cluster_key].astype('category')

    # Plot UMAP
    sc.pl.umap(
        adata,
        color=state.cluster_key,
        wspace=0.5,
        show=False,
        save='_umap.png'
    )
    plt.close()

    # Plot spatial
    sc.pl.spatial(
        adata,
        img_key='hires',
        color=state.cluster_key,
        show=False,
        save='_spatial_clusters.png'
    )
    plt.close()

    state.actions_taken.append('Clustered data')
    state.adata = adata
    return state


def calc_cancer_score(state: IngestState, file_name: str, ):
    """
    Calculate cancer scores

    :param file_name: name toi save result as
    :param state: Ingest state Object
    :return:
    """
    console.print("\n\n----- Labelling Cancerous cells -----\n\n")
    table = Table(title='Cancer Scoring Results')
    table.add_column('Metric')
    table.add_column('Value', justify='center')
    adata = state.adata

    cancer_genes = pd.read_csv('../cancer/cancer_gene_results.csv')
    # Get overlapping genes
    overlapping_genes = [gene for gene in cancer_genes['gene_name'].unique()
                         if gene in adata.var_names.values.tolist()]
    if not overlapping_genes:
        raise ValueError('No overlapping genes found between the two datasets')

    console.print(f'Found {len(overlapping_genes)} overlapping genes in dataset')
    table.add_row('Overlapping genes', str(len(overlapping_genes)))

    # Create weights dictionary from confidence scores
    gene_weights = {}
    for gene in overlapping_genes:
        # Average confidence score if gene appears multiple times
        confidence = cancer_genes[cancer_genes['gene_name'] == gene]['confidence_score'].mean()
        gene_weights[gene] = confidence

    # Calculate weighted expression for each spot
    cancer_scores = np.zeros(adata.n_obs)
    for gene in overlapping_genes:
        # Get gene expression (ensuring dense format)
        if isinstance(adata[:, gene].X, np.ndarray):
            expr = adata[:, gene].X.flatten()
        else:
            expr = adata[:, gene].X.toarray().flatten()

        # Add weighted expression to scores
        cancer_scores += expr * gene_weights[gene]

    # Normalise scores expression to [0, 1] range (min-max)
    score_min = cancer_scores.min()
    score_max = cancer_scores.max()
    cancer_scores_norm = (cancer_scores - score_min) / (score_max - score_min)

    # Add scores to adata object
    adata.obs['cancer_score'] = cancer_scores_norm

    stats = {
        "n_genes_used": len(overlapping_genes),
        "mean_score": float(np.mean(cancer_scores_norm)),
        'median_score': float(np.median(cancer_scores_norm)),
        "std_score": float(np.std(cancer_scores_norm)),
        "score_range": (float(cancer_scores_norm.min()), float(cancer_scores_norm.max()))
    }

    # Add stats to table
    table.add_row("Mean Score", f"{stats['mean_score']:.3f}")
    table.add_row("Median Score", f"{stats['median_score']:.3f}")
    table.add_row("Score Std Dev", f"{stats['std_score']:.3f}")
    table.add_row('Score range', f"[{stats['score_range'][0]:.3f}, {stats['score_range'][1]:.3f}]")
    console.print(table)

    # Add gene weights to adata for referencing
    adata.uns['cancer_score_genes'] = [[gene, gene_weights[gene]] for gene in overlapping_genes]

    #     {
    #     "genes": overlapping_genes,
    #     "weights": [gene_weights[g] for g in overlapping_genes]
    # }
    adata.uns['present_cancer_genes'] = overlapping_genes  # Cancer genes present in dataset

    matched_genes = cancer_genes[cancer_genes['gene_name'].isin(adata.var.index)]
    print(len(matched_genes))
    state.actions_taken.append('Calculated Cancer scores for cells')
    state.adata = adata

    adata.write_h5ad(Path(file_name))
    print(f'Saved file to {file_name}')

    return state


def get_high_cancer_regions(state: IngestState, percentile_threshold=75) -> tuple[
    pd.Series, IngestState]:
    """
    Identify regions with high cancer scores
    :param state: Ingest State object
    :param percentile_threshold: percentile threshold for high scores
    :return:
    """
    console.print("\n\n----- Determining highly cancerous regions -----\n\n")
    adata = state.adata
    if 'cancer_score' not in adata.obs:
        raise ValueError('Cancer scores not found. run the calc_cancer_score function first.')

    threshold = np.percentile(adata.obs['cancer_score'], percentile_threshold)
    high_scoring = adata.obs['cancer_score'] > threshold

    console.print(f'Identified {high_scoring.sum()} spots above {percentile_threshold}% threshold',
                  f'(Score > {threshold:.3f})')

    return high_scoring, state


def get_top_cancer_genes(state: IngestState, high_scoring_mask: pd.Series,
                         n_genes: int = 20) -> tuple[pd.DataFrame, IngestState]:
    """
    Identify top differentially expressed genes in high-scoring regions

    :param state: Ingest State object
    :param high_scoring_mask: Boolean mask of high-scoring regions
    :param n_genes: Number of top genes to return
    :return: Dataframe with gene statistics
    """
    adata = state.adata
    if "cancer_score_genes" not in adata.uns:
        raise ValueError('Cancer gene information not found. Run calc_cancer_score first.')

    # Ensure scanpy uses the correct figure directory
    sc.settings.figdir = state.figures_dir

    # Create grouping for differential expression
    adata.obs['scoring_group'] = 'low'
    adata.obs.loc[high_scoring_mask, 'scoring_group'] = 'high'

    # Perform differential expression analysis
    sc.tl.rank_genes_groups(adata, 'scoring_group', method='wilcoxon')

    # Get results
    result = sc.get.rank_genes_groups_df(adata, group='high')

    # Plot ranked genes
    sc.pl.rank_genes_groups(
        adata,
        n_genes=10,
        save='_ranked_genes.png'
    )
    plt.close()

    # Check if gene was in cancer_gene set
    cancer_genes = set(adata.uns['cancer_score_genes'])
    result['is_cancer_gene'] = result[0].isin(cancer_genes)
    state.adata = adata

    return result.head(n_genes), state

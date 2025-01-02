from statsmodels.graphics.tukeyplot import results
from src.ingest._methods import load_data, ingest_config, perform_qc, \
    cluster_data, PyPlotConfig, calc_cancer_score, get_high_cancer_regions, \
    get_top_cancer_genes
from src.ingest._methods import TissueType
from rich.pretty import pprint
from rich.prompt import Prompt, IntPrompt, Confirm
from rich.console import Console
import scanpy as sc
from pathlib import Path
import os
from datetime import datetime

console = Console()

# Define base data directory relative to project root
DATA_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))) / 'src/data'


def get_available_datasets():
    """Get list of available datasets in the data directory"""
    if not DATA_DIR.exists():
        console.print("[yellow]Warning: Data directory not found. Creating it...[/]")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        return []

    # Get all subdirectories in the data directory
    datasets = [d.name for d in DATA_DIR.iterdir() if d.is_dir()]
    return datasets


def get_initial_config():
    """Get initial configuration from user"""

    console.print("\n[bold blue]Initial Configuration[/bold blue]")

    # Get verbosity level
    verbosity = IntPrompt.ask(
        "Enter Scanpy verbosity level (0-3)",
        default=3,
        choices=["0", "1", "2", "3"]
    )

    # Get figures directory
    figures_dir = Prompt.ask(
        "Enter directory for saving figures",
        default=f"ingest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    # Get tissue type
    tissue_options = {str(i): t.value for i, t in enumerate(TissueType)}
    console.print("\nAvailable tissue types:")
    for idx, tissue in tissue_options.items():
        console.print(f"{idx}: {tissue}")

    tissue_choice = Prompt.ask(
        "Select tissue type number",
        choices=list(tissue_options.keys()),
        default="0"
    )
    tissue_type = TissueType(tissue_options[tissue_choice])

    return verbosity, f'figures/{figures_dir}', tissue_type


def get_qc_params():
    """Get QC parameters from user"""
    console.print("\n[bold blue]Quality Control Parameters[/bold blue]")

    mito_threshold = IntPrompt.ask(
        "Enter mitochondrial genes threshold (%)",
        default=20
    )

    ribo_threshold = IntPrompt.ask(
        "Enter ribosomal genes threshold (%)",
        default=20
    )

    cell_threshold = IntPrompt.ask(
        "Enter minimum cell count threshold",
        default=30
    )

    gene_threshold = IntPrompt.ask(
        "Enter minimum gene count threshold",
        default=5
    )

    n_top_genes = IntPrompt.ask(
        "Enter number of top variable genes",
        default=5000
    )

    return mito_threshold, ribo_threshold, cell_threshold, gene_threshold, n_top_genes


def get_clustering_params():
    """Get clustering parameters from user"""
    console.print("\n[bold blue]Clustering Parameters[/bold blue]")

    n_pcs = IntPrompt.ask(
        "Enter number of principal components",
        default=15
    )

    n_neighbors = IntPrompt.ask(
        "Enter number of neighbors",
        default=30
    )

    cluster_res = float(Prompt.ask(
        "Enter clustering resolution",
        default="0.5"
    ))

    return n_pcs, n_neighbors, cluster_res


def Ingest():
    # Get initial configuration
    verbosity, figures_dir, tissue_type = get_initial_config()

    # Initialize configuration
    config = PyPlotConfig()
    state = ingest_config(
        sc_verbosity=verbosity,
        pyplot_config=config,
        figures_dir=figures_dir,
        tissue_type=tissue_type
    )

    # Get available datasets
    datasets = get_available_datasets()
    if not datasets:
        console.print("[red]No datasets found in data directory. Please add datasets to:", DATA_DIR)
        return

    # Show available datasets
    console.print("\n[bold blue]Available Datasets:[/]")
    for idx, dataset in enumerate(datasets):
        console.print(f"{idx}: {dataset}")

    # Get dataset choice
    dataset_idx = IntPrompt.ask(
        "\nSelect dataset number",
        choices=[str(i) for i in range(len(datasets))],
        default="0"
    )
    dataset_name = datasets[dataset_idx]

    # Construct full path
    data_path = DATA_DIR / dataset_name
    console.print(f"\nUsing dataset at: {data_path}")

    # Load data
    if Confirm.ask("\nProceed with loading data?"):
        state = load_data(state, str(data_path))
        console.print("[green]✓ Data loaded successfully[/green]")

    # Quality control
    if Confirm.ask("\nProceed with quality control?"):
        mito_threshold, ribo_threshold, cell_threshold, gene_threshold, n_top_genes = get_qc_params()
        state = perform_qc(
            state,
            mito_threshold=mito_threshold,
            ribo_threshold=ribo_threshold,
            cell_threshold=cell_threshold,
            gene_threshold=gene_threshold,
            n_top_genes=n_top_genes
        )
        console.print("[green]✓ Quality control completed[/green]")

    # Clustering
    if Confirm.ask("\nProceed with clustering?"):
        n_pcs, n_neighbors, cluster_res = get_clustering_params()
        state = cluster_data(
            state,
            n_pcs=n_pcs,
            n_neighbors=n_neighbors,
            cluster_res=cluster_res
        )
        console.print("[green]✓ Clustering completed[/green]")

    # Cancer scoring
    if Confirm.ask("\nProceed with cancer scoring?"):
        output_file = Prompt.ask(
            "Enter output file name",
            default="results.h5ad"
        )
        state = calc_cancer_score(state, file_name=output_file)
        console.print("[green]✓ Cancer scoring completed[/green]")

    # High cancer regions
    if Confirm.ask("\nIdentify high cancer regions?"):
        percentile = IntPrompt.ask(
            "Enter percentile threshold for high cancer scores",
            default=75
        )
        high_scoring_mask, state = get_high_cancer_regions(state, percentile_threshold=percentile)
        console.print("[green]✓ High cancer regions identified[/green]")

        # Top cancer genes
        if Confirm.ask("\nAnalyze top cancer genes?"):
            n_genes = IntPrompt.ask(
                "Enter number of top genes to analyze",
                default=20
            )
            result, state = get_top_cancer_genes(state, high_scoring_mask, n_genes=n_genes)
            console.print("\n[bold]Top differentially expressed genes:[/bold]")
            console.print(result)

    console.print("\n[bold green]Pipeline completed successfully![/bold green]")



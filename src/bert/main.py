from sympy import pprint

from src.bert.data_prep import TransformerDataPrep
from src.bert.new_tokeniser import prepare_spatial_tokeniser
from src.bert.dataloader import create_st_dataloaders
from rich.console import Console
import scanpy as sc

console = Console()


def main():
    # Load adata object
    adata = sc.read_h5ad('../ingest/results.h5ad')

    # prepare data
    data_prep = TransformerDataPrep(adata_path='../ingest/results.h5ad')
    combined_tensor, distances, feature_info = data_prep.prepare_data()

    processed_data = {
        "combined_features": combined_tensor,
        "distance_info": distances,
        "feature_info": feature_info
    }

    # Initialise tokeniser
    tokeniser, spatial_coords, X_normalised = prepare_spatial_tokeniser(
        adata=adata,
        processed_data=processed_data,
        force_retrain=False
    )

    train_datasets, val_datasets = create_st_dataloaders(
        adata=adata,
        processed_data=processed_data,
        tokenizer=tokeniser,
        batch_size=32
    )

    pprint(train_datasets)


if __name__ == "__main__":
    main()

import torch
from rich.console import Console
from src.bert.base import CellMetaBERT
from src.bert.data_prep import TransformerDataPrep
from src.bert.evaluate.metabolic_metrics import MetabolicEvaluator
from src.bert.dataloader import create_train_val_datasets
from src.bert.tokeniser import prep_st_data
import scanpy as sc

console = Console()


def load_and_evaluate_model(checkpoint_path, data_loader, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Load and evaluate a saved model checkpoint"""

    console.print(f"[bold blue]Loading model from {checkpoint_path}...")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Initialize model
    model = CellMetaBERT(
        bert_model_name='bert-base-uncased',
        n_metabolic_features=4,
        dropout=0.1
    )

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # Print checkpoint info
    console.print(f"[green]Checkpoint loaded from epoch: {checkpoint['epoch']}")
    console.print("[cyan]Previous metrics:")
    for metric_name, value in checkpoint['metrics'].items():
        console.print(f"  {metric_name}: {value}")

    # Evaluate
    console.print("\n[bold blue]Evaluating model...")
    evaluator = MetabolicEvaluator(model, data_loader, device)
    metrics, predictions, true_values = evaluator.compute_metrics()

    # Print current metrics
    console.print("\n[bold green]Current Evaluation Metrics:")
    for feature, values in metrics.items():
        for metric_name, value in values.items():
            console.print(f"{feature} - {metric_name}: {value:.4f}")

    return model, metrics, predictions, true_values


# Usage example:
if __name__ == "__main__":
    checkpoint_path = "../checkpoints/metabolic/best_model.pt"

    console.print("\n[bold blue]Preparing data...")
    data_prep = TransformerDataPrep(
        adata_path='../../analysed_doc.h5ad',
        use_pca=True,
        use_spatial=True
    )
    adata = sc.read_h5ad('../../analysed_doc.h5ad')
    with console.status("[bold green]Processing data..."):
        combined_features, feature_info = data_prep.prepare_data()
        console.print("✓ Data processing complete")

        tokeniser, spatial_coords, _ = prep_st_data(
            adata=data_prep.adata,
            force_retrain=False
        )

    # Recreate your validation dataset
    # (You'll need the same data and tokenizer used during training)
    train_loader, val_loader = create_train_val_datasets(
        adata=adata,
        combined_features=combined_features,
        metabolic_features=feature_info['metabolic_features'],
        spatial_coords=spatial_coords,
        tokenizer=tokeniser,
        batch_size=32
    )

    # Load and evaluate
    model, metrics, predictions, true_values = load_and_evaluate_model(
        checkpoint_path=checkpoint_path,
        data_loader=val_loader
    )

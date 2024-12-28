from pathlib import Path
import scanpy as sc
from src.bert.usage import model, gat_model, val_dataset, adata

from src.bert.evaluate import ModelAnalyser


def main():
    analyser = ModelAnalyser(
        model=model,
        gat_model=gat_model,
        dataset=val_dataset,
        device='cuda',
        checkpoint_path="../training/checkpoints/brain_cancer_model/best_model.pt"
    )

    # Create evaluation directory
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)

    # Plot training history
    analyser.plot_training_history(save_path=output_dir / "training_history.png")

    # Evaluate model
    predictions, true_values, attn_weights, metrics = analyser.evaluate_model(num_samples=10)

    # Print metrics
    analyser.print_eval_metrics(metrics)

    # Plot predictions
    analyser.plot_prediction_scatter(
        true_values=true_values,
        predictions=predictions,
        save_path=output_dir / 'prediction_scatter.png'
    )

    # Analyse attention weights
    gene_names = list(adata.var_names)
    analyser.analyse_attn_weights(attn_weights, gene_names)

    # Save predictions
    analyser.save_predictions(
        predictions,
        true_values,
        gene_names,
        save_path=output_dir / 'predictions.csv'
    )


if __name__ == "__main__":
    main()

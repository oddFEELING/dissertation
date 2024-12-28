import torch
from pathlib import Path
from rich.console import Console
from rich.progress import track
from rich.table import Table
from src.bert.trainer_new import MetabolicTrainer
from src.bert.base import CellMetaBERT
from src.bert.dataloader import create_train_val_datasets
from src.bert.data_prep import TransformerDataPrep
from src.bert.evaluate.metabolic_metrics import MetabolicEvaluator
import matplotlib.pyplot as plt
from src.bert.tokeniser import prep_st_data  # Your custom tokenizer

console = Console()


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[bold blue]Using device: {device}")

    try:
        config = {
            'batch_size': 32,
            'learning_rate': 2e-5,
            'num_epochs': 10,
            'warmup_epochs': 2
        }

        # Print configuration
        console.print("\n[bold cyan]Configuration:")
        for key, value in config.items():
            console.print(f"  {key}: {value}")

        # Prepare data
        console.print("\n[bold blue]Preparing data...")
        data_prep = TransformerDataPrep(
            adata_path='path/to/your/data.h5ad',
            use_pca=True,
            use_spatial=True
        )

        # Get processed data
        with console.status("[bold green]Processing data..."):
            combined_features, feature_info = data_prep.prepare_data()
            console.print("✓ Data processing complete")

        # Initialize tokenizer and get spatial coordinates
        console.print("\n[bold blue]Initializing tokenizer...")
        tokenizer, spatial_coords, _ = prep_st_data(
            adata=data_prep.adata,
            force_retrain=False
        )
        console.print("✓ Tokenizer initialized")

        # Create dataloaders
        console.print("\n[bold blue]Creating data loaders...")
        train_loader, val_loader = create_train_val_datasets(
            adata=data_prep.adata,
            combined_features=combined_features,
            metabolic_features=feature_info['metabolic_features'],
            spatial_coords=spatial_coords,  # Use spatial_coords from prep_st_data
            tokenizer=tokenizer,
            batch_size=config['batch_size']
        )
        console.print(f"✓ Train samples: {len(train_loader.dataset)}")
        console.print(f"✓ Validation samples: {len(val_loader.dataset)}")

        # Initialize model
        console.print("\n[bold blue]Initializing model...")
        model = CellMetaBERT(
            bert_model_name='bert-base-uncased',
            n_metabolic_features=4,
            dropout=0.1
        ).to(device)
        console.print("✓ Model initialized")

        # Set up checkpoint directory
        checkpoint_dir = Path("checkpoints/metabolic")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"✓ Checkpoint directory: {checkpoint_dir}")

        # Initialize trainer
        trainer = MetabolicTrainer(
            model=model,
            train_dataset=train_loader,
            val_dataset=val_loader,
            device=device,
            learning_rate=config['learning_rate'],
            num_epochs=config['num_epochs'],
            warmup_epochs=config['warmup_epochs'],
            checkpoint_dir=str(checkpoint_dir)
        )

        # Training loop
        console.print("\n[bold blue]Starting training...")
        best_loss = float('inf')

        for epoch in range(config['num_epochs']):
            # Create epoch header
            console.rule(f"[bold cyan]Epoch {epoch + 1}/{config['num_epochs']}")

            # Train
            train_loss = trainer.train_epoch(epoch)
            console.print(f"📈 Training Loss: {train_loss:.4f}")

            # Evaluate
            console.print("\n🔍 [bold blue]Evaluating model...")
            evaluator = MetabolicEvaluator(model, val_loader, device)
            metrics, predictions, true_values = evaluator.compute_metrics()

            # Create metrics table
            table = Table(title=f"Epoch {epoch + 1} Metrics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            for feature, values in metrics.items():
                if feature != 'spatial':
                    for metric_name, value in values.items():
                        table.add_row(f"{feature} - {metric_name}", f"{value:.4f}")

            console.print(table)

            # Generate and save visualization
            console.print("\n📊 [bold blue]Generating visualizations...")
            evaluator.visualize_predictions(
                predictions=predictions,
                true_values=true_values,
                spatial_coords=spatial_coords[val_loader.dataset.indices]
            )
            plt.savefig(f"{checkpoint_dir}/eval_epoch_{epoch + 1}.png")
            plt.close()
            console.print(f"✓ Saved visualization to: eval_epoch_{epoch + 1}.png")

            # Save checkpoint
            val_loss = metrics['mt_activity']['mse']  # Using mt_activity MSE as validation loss
            is_best = val_loss < best_loss
            if is_best:
                best_loss = val_loss
                console.print("\n🏆 [bold green]New best model!")

            trainer.save_checkpoint(epoch, metrics, is_best)
            console.print(f"✓ Saved checkpoint for epoch {epoch + 1}")

        console.print("\n[bold green]Training complete! 🎉")

    except Exception as e:
        console.print(f"\n[bold red]Error: {str(e)}")
        console.print_exception()
        raise


if __name__ == "__main__":
    main()

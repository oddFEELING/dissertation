import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from pathlib import Path
import logging
import json
from datetime import datetime
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.console import Console
from rich.table import Table
import wandb
from typing import Dict, Optional
import numpy as np
from tqdm import tqdm

from src.bert.compute_loss import SpatialBertLoss

console = Console()
logger = logging.getLogger(__name__)

class SpatialTrainer:
    """Trainer for SpatialBERT model with checkpointing and detailed metrics"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = 1e-4,
        num_epochs: int = 10,
        save_dir: str = "checkpoints",
        feature_weights: Optional[Dict[str, float]] = None,
        loss_weights: Optional[Dict[str, float]] = None,
        use_wandb: bool = True,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize trainer
        
        Args:
            model: The CellMetaBERT model
            train_loader: Training data loader
            val_loader: Validation data loader
            learning_rate: Learning rate for optimizer
            num_epochs: Number of training epochs
            save_dir: Directory to save checkpoints
            feature_weights: Weights for different features in loss computation
            loss_weights: Weights for different loss components
            use_wandb: Whether to use Weights & Biases for logging
            device: Device to train on
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.save_dir = Path(save_dir)
        self.use_wandb = use_wandb
        
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize optimizer and scheduler
        self.optimizer = AdamW(model.parameters(), lr=learning_rate)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs)
        
        # Initialize loss function with both feature and loss weights
        self.criterion = SpatialBertLoss(
            feature_weights=feature_weights,
            loss_weights=loss_weights,
            use_spatial_context=getattr(model.config, 'use_spatial_context', True),
            use_gene_context=getattr(model.config, 'use_gene_context', True),
            use_cell_state=getattr(model.config, 'use_cell_state', True)
        )
        
        # Initialize best metrics for checkpointing
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
        # Setup logging
        if use_wandb:
            wandb.init(project="spatial-bert", config={
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
                "feature_weights": feature_weights,
                "loss_weights": loss_weights,
                "model_config": model.config.to_dict()
            })
            
    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics
        }
        
        # Save latest checkpoint
        checkpoint_path = self.save_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint if this is the best model
        if is_best:
            best_path = self.save_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            
        # Save metrics
        metrics_path = self.save_dir / f"metrics_epoch_{epoch}.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
            
        logger.info(f"Saved checkpoint for epoch {epoch}")
        
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint['epoch'], checkpoint['metrics']
        
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        metrics = {}

        # Create progress bar
        pbar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f"Training Epoch {epoch}"
        )

        for batch_idx, batch in pbar:
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            predictions = self.model(**batch)

            # Compute loss
            loss, batch_metrics = self.criterion.compute_loss(predictions, batch)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            for k, v in batch_metrics.items():
                if k not in metrics:
                    metrics[k] = 0
                metrics[k] += v.item() if isinstance(v, torch.Tensor) else v

            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})

        # Average metrics
        metrics = {k: v / len(self.train_loader) for k, v in metrics.items()}
        metrics['epoch'] = epoch

        return metrics
        
    def validate(self) -> Dict:
        """Run validation"""
        self.model.eval()
        total_loss = 0
        all_metrics = []
        
        with torch.no_grad():
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn()
            ) as progress:
                task = progress.add_task("[cyan]Validating", total=len(self.val_loader))
                
                for batch in self.val_loader:
                    # Move batch to device
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                    
                    # Forward pass
                    predictions = self.model(**batch)
                    
                    # Compute loss and metrics
                    loss, metrics = self.criterion.compute_loss(predictions, batch)
                    
                    total_loss += loss.item()
                    all_metrics.append(metrics)
                    progress.update(task, advance=1)
                    
        # Compute validation metrics
        val_metrics = self.aggregate_metrics(all_metrics)
        val_metrics['loss'] = total_loss / len(self.val_loader)
        
        return val_metrics
        
    def aggregate_metrics(self, metrics_list: list) -> Dict:
        """Aggregate metrics from multiple batches"""
        aggregated = {}
        
        # Get all metric keys
        all_keys = set()
        for metrics in metrics_list:
            all_keys.update(metrics.keys())
            
        # Aggregate each metric
        for key in all_keys:
            values = [m[key] for m in metrics_list if key in m]
            if isinstance(values[0], dict):
                # Handle nested metrics (e.g., accuracies at different thresholds)
                aggregated[key] = {
                    k: np.mean([v[k].item() if isinstance(v[k], torch.Tensor) else v[k] 
                              for v in values if k in v])
                    for k in values[0].keys()
                }
            else:
                # Handle simple metrics
                aggregated[key] = np.mean([v.item() if isinstance(v, torch.Tensor) else v 
                                         for v in values])
                
        return aggregated
        
    def display_metrics(self, metrics: Dict, phase: str):
        """Display metrics in a formatted table"""
        table = Table(title=f"{phase} Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        def add_metric_to_table(name: str, value: float):
            if isinstance(value, float):
                table.add_row(name, f"{value:.4f}")
            else:
                table.add_row(name, str(value))
                
        # Add main metrics
        add_metric_to_table("Loss", metrics['loss'])
        add_metric_to_table("Location Accuracy", metrics['location_accuracy'])
        add_metric_to_table("Distance Accuracy", metrics['distance_accuracy'])
        add_metric_to_table("Border Accuracy", metrics['border_accuracy'])
        
        # Add detailed metrics
        for metric_name, metric_dict in metrics.items():
            if isinstance(metric_dict, dict):
                for threshold, value in metric_dict.items():
                    add_metric_to_table(f"{metric_name}_{threshold}", value)
                    
        console.print(table)
        
    def train(self):
        """Train the model"""
        console.print("[bold blue]Starting training...[/]")
        
        for epoch in range(self.num_epochs):
            # Train epoch
            console.print(f"\n[bold cyan]Epoch {epoch + 1}/{self.num_epochs}[/]")
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate()
            
            # Update learning rate
            self.scheduler.step()
            
            # Display metrics
            console.print("\n[bold green]Training Metrics:[/]")
            self.display_metrics(train_metrics, "Training")
            console.print("\n[bold green]Validation Metrics:[/]")
            self.display_metrics(val_metrics, "Validation")
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train': train_metrics,
                    'val': val_metrics
                })
                
            # Save checkpoint
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
                self.best_epoch = epoch
                
            self.save_checkpoint(
                epoch=epoch,
                metrics={'train': train_metrics, 'val': val_metrics},
                is_best=is_best
            )
            
        console.print(f"\n[bold green]Training completed! Best validation loss: {self.best_val_loss:.4f} at epoch {self.best_epoch}[/]")
        
    def predict(self, batch: Dict) -> Dict:
        """Make predictions for a batch of data"""
        self.model.eval()
        
        with torch.no_grad():
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Get predictions
            predictions = self.model(**batch)
            
            # Format predictions
            formatted_predictions = {}
            for spot_idx in range(len(batch['input_ids'])):
                spot_id = batch['spot_id'][spot_idx].item()
                spot_location = batch['spatial_coords'][spot_idx].cpu().numpy()
                
                # Get neighbor predictions
                neighbors = []
                for n_idx in range(len(batch['neighbor_indices'][spot_idx])):
                    neighbor = {
                        'location': {
                            'x': predictions['neighbor_location'][spot_idx, 0].item(),
                            'y': predictions['neighbor_location'][spot_idx, 1].item()
                        },
                        'distance': predictions['neighbor_distance'][spot_idx].item(),
                        'interaction_strength': predictions['interaction_strength'][spot_idx].item(),
                        'features': {
                            name: predictions['features'][spot_idx, i].item()
                            for i, name in enumerate(self.model.feature_names)
                        },
                        'gene_expression': {
                            'expression_probabilities': {
                                f"gene_{i}": prob.item()
                                for i, prob in enumerate(predictions['gene_expression_probs'][spot_idx])
                            }
                        }
                    }
                    neighbors.append(neighbor)
                    
                formatted_predictions[spot_id] = {
                    'spot_id': spot_id,
                    'spot_location': {'x': spot_location[0], 'y': spot_location[1]},
                    'is_border_cell': predictions['is_border_cell'][spot_idx].item() > 0.5,
                    'neighbors': neighbors
                }
                
        return formatted_predictions

import torch
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
import matplotlib.pyplot as plt
from rich.console import Console
from typing import Dict, Tuple, Any
import warnings

console = Console()

class MetabolicEvaluator:
    def __init__(self, model, val_loader, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the evaluator
        
        Args:
            model: The CellMetaBERT model
            val_loader: DataLoader for validation data
            device: Computing device (cuda/cpu)
        """
        self.model = model
        self.val_loader = val_loader
        self.device = device
        self.model.eval()
        self.feature_names = ['mt_activity', 'mt_percentage', 
                            'ribo_activity', 'ribo_percentage']

    def _safe_compute_correlation(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Safely compute correlations handling edge cases"""
        try:
            if len(x) < 2 or len(y) < 2:
                return 0.0, 1.0
            if np.all(x == x[0]) or np.all(y == y[0]):
                return 0.0, 1.0
            return pearsonr(x, y)
        except Exception as e:
            warnings.warn(f"Correlation computation failed: {str(e)}")
            return 0.0, 1.0

    @torch.no_grad()
    def compute_metrics(self) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
        """Compute comprehensive evaluation metrics"""
        all_predictions = []
        all_true_values = []
        all_spatial_coords = []

        try:
            for batch in self.val_loader:
                # Ensure all required batch elements are present
                required_keys = ['input_ids', 'attention_mask', 'spatial_coords', 'metabolic_features']
                if not all(key in batch for key in required_keys):
                    raise KeyError(f"Batch missing required keys. Expected {required_keys}")

                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                spatial_coords = batch['spatial_coords'].to(self.device)
                true_values = batch['metabolic_features'].to(self.device)

                # Get predictions
                predictions = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    spatial_coords=spatial_coords
                )

                # Store values
                all_predictions.append(predictions.cpu().numpy())
                all_true_values.append(true_values.cpu().numpy())
                all_spatial_coords.append(spatial_coords.cpu().numpy())

            # Concatenate all data
            predictions = np.vstack(all_predictions)
            true_values = np.vstack(all_true_values)
            spatial_coords = np.vstack(all_spatial_coords)

            # Verify shapes match
            if not (predictions.shape == true_values.shape):
                raise ValueError(f"Shape mismatch: predictions {predictions.shape} != true_values {true_values.shape}")

            # Compute comprehensive metrics
            metrics = {}
            for i, feature in enumerate(self.feature_names):
                pred_feature = predictions[:, i]
                true_feature = true_values[:, i]
                
                # Handle potential NaN/Inf values
                valid_mask = np.isfinite(pred_feature) & np.isfinite(true_feature)
                pred_feature = pred_feature[valid_mask]
                true_feature = true_feature[valid_mask]
                
                if len(pred_feature) == 0:
                    console.print(f"[yellow]Warning: No valid predictions for {feature}[/]")
                    continue

                pearson_r, p_value = self._safe_compute_correlation(true_feature, pred_feature)
                
                metrics[feature] = {
                    'mse': mean_squared_error(true_feature, pred_feature),
                    'rmse': np.sqrt(mean_squared_error(true_feature, pred_feature)),
                    'mae': mean_absolute_error(true_feature, pred_feature),
                    'r2': max(0.0, r2_score(true_feature, pred_feature)),  # Prevent negative R2
                    'pearson_r': pearson_r,
                    'pearson_p': p_value
                }

            # Add spatial correlation analysis if spatial coords are valid
            if np.isfinite(spatial_coords).all():
                spatial_metrics = self._compute_spatial_metrics(predictions, true_values, spatial_coords)
                metrics['spatial'] = spatial_metrics

            return metrics, predictions, true_values

        except Exception as e:
            console.print(f"[bold red]Error during metric computation: {str(e)}[/]")
            raise

    def _compute_spatial_metrics(self, predictions: np.ndarray, true_values: np.ndarray, 
                               spatial_coords: np.ndarray) -> Dict[str, float]:
        """Compute metrics related to spatial patterns"""
        try:
            from scipy.spatial.distance import pdist, squareform
            
            spatial_metrics = {}
            
            # Verify input shapes
            if not all(arr.shape[0] == predictions.shape[0] for arr in [true_values, spatial_coords]):
                raise ValueError("Shape mismatch in spatial metric computation")
            
            # Calculate pairwise distances
            spatial_distances = squareform(pdist(spatial_coords))
            
            for i, feature in enumerate(self.feature_names):
                # Calculate spatial autocorrelation
                moran_i = self._compute_morans_i(predictions[:, i], spatial_distances)
                spatial_metrics[f'{feature}_morans_i'] = moran_i
                
                # Calculate prediction error vs distance
                error = np.abs(predictions[:, i] - true_values[:, i])
                error_dist_corr = spearmanr(error, spatial_distances.mean(axis=1))[0]
                if np.isfinite(error_dist_corr):
                    spatial_metrics[f'{feature}_error_dist_corr'] = error_dist_corr
                else:
                    spatial_metrics[f'{feature}_error_dist_corr'] = 0.0
                    
            return spatial_metrics

        except Exception as e:
            console.print(f"[yellow]Warning: Spatial metrics computation failed: {str(e)}[/]")
            return {}

    def visualize_predictions(self, predictions: np.ndarray, true_values: np.ndarray, 
                            spatial_coords: np.ndarray) -> plt.Figure:
        """Create comprehensive visualization of predictions"""
        try:
            plt.figure(figsize=(20, 15))
            
            # 1. Scatter plots of predicted vs true values
            for i, feature in enumerate(self.feature_names):
                plt.subplot(3, 2, i+1)
                valid_mask = np.isfinite(predictions[:, i]) & np.isfinite(true_values[:, i])
                
                if np.sum(valid_mask) > 0:
                    sns.scatterplot(x=true_values[valid_mask, i], 
                                  y=predictions[valid_mask, i], 
                                  alpha=0.5)
                    
                    value_range = [
                        min(true_values[valid_mask, i].min(), predictions[valid_mask, i].min()),
                        max(true_values[valid_mask, i].max(), predictions[valid_mask, i].max())
                    ]
                    plt.plot(value_range, value_range, 'r--', alpha=0.5)
                    
                    r2 = r2_score(true_values[valid_mask, i], predictions[valid_mask, i])
                    plt.title(f'{feature}\nR² = {r2:.3f}')
                    plt.xlabel('True Values')
                    plt.ylabel('Predicted Values')

            # 2. Spatial distribution of errors
            if np.isfinite(spatial_coords).all():
                errors = np.nanmean(np.abs(predictions - true_values), axis=1)
                plt.subplot(3, 2, 5)
                plt.scatter(spatial_coords[:, 0], spatial_coords[:, 1], 
                           c=errors, cmap='viridis', alpha=0.6)
                plt.colorbar(label='Mean Absolute Error')
                plt.title('Spatial Distribution of Prediction Errors')
                plt.xlabel('Spatial X')
                plt.ylabel('Spatial Y')

            # 3. Error distribution
            plt.subplot(3, 2, 6)
            error_data = [np.abs(predictions[:, i] - true_values[:, i]) 
                         for i in range(len(self.feature_names))]
            sns.boxplot(data=error_data, labels=self.feature_names)
            plt.xticks(rotation=45)
            plt.title('Error Distribution by Feature')
            plt.ylabel('Absolute Error')

            plt.tight_layout()
            return plt.gcf()

        except Exception as e:
            console.print(f"[bold red]Error during visualization: {str(e)}[/]")
            plt.close()
            raise 

    def _compute_morans_i(self, values, distances, threshold=None):
        """
        Compute Moran's I statistic for spatial autocorrelation
        
        Args:
            values: Array of values to compute autocorrelation for
            distances: Distance matrix between points
            threshold: Distance threshold for considering neighbors (default: median distance)
        
        Returns:
            float: Moran's I statistic
        """
        try:
            if threshold is None:
                threshold = np.median(distances)
            
            # Create weight matrix
            W = (distances <= threshold).astype(float)
            np.fill_diagonal(W, 0)
            
            # Standardize values
            z = (values - np.mean(values)) / np.std(values)
            
            # Calculate Moran's I
            N = len(values)
            W_sum = W.sum()
            
            if W_sum == 0:  # No neighbors within threshold
                return 0.0
            
            I = (N / W_sum) * (z @ W @ z) / (z @ z)
            return float(I)
            
        except Exception as e:
            console.print(f"[yellow]Warning: Moran's I computation failed: {str(e)}[/]")
            return 0.0 
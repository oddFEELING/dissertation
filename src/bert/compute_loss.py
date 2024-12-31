import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class SpatialBertLoss:
    """Loss function for CellMetaBERT model"""
    
    def __init__(
        self,
        feature_weights: Dict[str, float] = None,
        loss_weights: Dict[str, float] = None
    ):
        self.feature_weights = feature_weights or {
            'cancer_score': 2.0,
            'n_genes': 1.0,
            'pct_counts_mito': 1.5,
            'pct_counts_ribo': 1.5,
            'total_counts': 1.0,
            'total_counts_mito': 1.0,
            'total_counts_ribo': 1.0
        }
        
        self.loss_weights = loss_weights or {
            'neighbor': 1.0,
            'feature': 1.0,
            'spatial_context': 1.5
        }
        
        # MSE loss for continuous values
        self.mse_loss = nn.MSELoss(reduction='none')
        
    def _validate_shapes(self, predictions: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]):
        """Validate input shapes and log debug information"""
        # Log shapes for debugging
        logger.debug("Prediction shapes:")
        for k, v in predictions.items():
            logger.debug(f"  {k}: {v.shape}")
        
        logger.debug("Batch shapes:")
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                logger.debug(f"  {k}: {v.shape}")
        
        # Basic shape validation
        assert 'neighbor_preds' in predictions, "Missing neighbor predictions"
        assert 'feature_preds' in predictions, "Missing feature predictions"
        assert 'confidence' in predictions, "Missing confidence scores"
        
        assert 'neighbor_features' in batch, "Missing neighbor features in batch"
        assert 'features' in batch, "Missing features in batch"
        assert 'neighbor_mask' in batch, "Missing neighbor mask in batch"
        
        # Shape validation
        batch_size = predictions['neighbor_preds'].size(0)
        max_neighbors = predictions['neighbor_preds'].size(1)
        feature_dim = predictions['neighbor_preds'].size(2)
        
        assert batch['neighbor_features'].size(0) == batch_size, \
            f"Batch size mismatch: {batch['neighbor_features'].size(0)} vs {batch_size}"
        assert batch['neighbor_features'].size(2) == feature_dim, \
            f"Feature dimension mismatch: {batch['neighbor_features'].size(2)} vs {feature_dim}"
            
        return batch_size, max_neighbors, feature_dim
    
    def _compute_neighbor_loss(
        self,
        pred_neighbors: torch.Tensor,  # [B, max_n, F]
        true_neighbors: torch.Tensor,  # [B, n, F]
        neighbor_mask: torch.Tensor    # [B, n]
    ) -> torch.Tensor:
        """Compute neighbor prediction loss with proper masking"""
        batch_size = pred_neighbors.size(0)
        max_neighbors = pred_neighbors.size(1)
        num_neighbors = true_neighbors.size(1)
        
        # Pad true neighbors to match prediction size
        if num_neighbors < max_neighbors:
            padding = torch.zeros(
                batch_size,
                max_neighbors - num_neighbors,
                true_neighbors.size(2),
                device=true_neighbors.device
            )
            true_neighbors = torch.cat([true_neighbors, padding], dim=1)
            # Extend mask
            mask_padding = torch.zeros(
                batch_size,
                max_neighbors - num_neighbors,
                device=neighbor_mask.device
            )
            neighbor_mask = torch.cat([neighbor_mask, mask_padding], dim=1)
        else:
            # Truncate if necessary
            true_neighbors = true_neighbors[:, :max_neighbors, :]
            neighbor_mask = neighbor_mask[:, :max_neighbors]
        
        # Compute MSE loss
        neighbor_loss = self.mse_loss(pred_neighbors, true_neighbors)  # [B, max_n, F]
        
        # Apply feature weights if available
        if hasattr(self, 'feature_weights'):
            feature_weights = torch.tensor(
                [self.feature_weights.get(f, 1.0) for f in range(neighbor_loss.size(-1))],
                device=neighbor_loss.device
            )
            neighbor_loss = neighbor_loss * feature_weights
        
        # Average over features
        neighbor_loss = neighbor_loss.mean(dim=-1)  # [B, max_n]
        
        # Apply neighbor mask
        neighbor_mask = neighbor_mask.float()
        masked_loss = neighbor_loss * neighbor_mask
        
        # Average over non-masked neighbors
        final_loss = masked_loss.sum() / (neighbor_mask.sum() + 1e-8)
        
        return final_loss
    
    def _compute_feature_loss(
        self,
        pred_features: torch.Tensor,  # [B, F]
        true_features: torch.Tensor,  # [B, F]
        confidence: torch.Tensor      # [B, 1]
    ) -> torch.Tensor:
        """Compute feature prediction loss with confidence weighting"""
        # Compute MSE loss
        feature_loss = self.mse_loss(pred_features, true_features)  # [B, F]
        
        # Apply feature weights if available
        if hasattr(self, 'feature_weights'):
            feature_weights = torch.tensor(
                [self.feature_weights.get(f, 1.0) for f in range(feature_loss.size(-1))],
                device=feature_loss.device
            )
            feature_loss = feature_loss * feature_weights
        
        # Average over features
        feature_loss = feature_loss.mean(dim=-1)  # [B]
        
        # Apply confidence weighting
        confidence = confidence.squeeze(-1)  # [B]
        weighted_loss = feature_loss * confidence
        
        # Average over batch
        final_loss = weighted_loss.mean()
        
        return final_loss
    
    def _compute_metrics(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        neighbor_loss: torch.Tensor,
        feature_loss: torch.Tensor
    ) -> Dict[str, float]:
        """Compute additional metrics for monitoring"""
        with torch.no_grad():
            metrics = {
                'neighbor_loss': neighbor_loss.item(),
                'feature_loss': feature_loss.item(),
                'total_loss': (
                    self.loss_weights['neighbor'] * neighbor_loss +
                    self.loss_weights['feature'] * feature_loss
                ).item(),
                'mean_confidence': predictions['confidence'].mean().item()
            }
            
            # Compute accuracy metrics
            neighbor_accuracy = self._compute_neighbor_accuracy(
                predictions['neighbor_preds'],
                batch['neighbor_features'],
                batch['neighbor_mask']
            )
            metrics.update(neighbor_accuracy)
            
            feature_accuracy = self._compute_feature_accuracy(
                predictions['feature_preds'],
                batch['features']
            )
            metrics.update(feature_accuracy)
            
        return metrics
    
    def _compute_neighbor_accuracy(
        self,
        pred_neighbors: torch.Tensor,
        true_neighbors: torch.Tensor,
        neighbor_mask: torch.Tensor,
        thresholds: Dict[str, float] = {'low': 0.1, 'medium': 0.2, 'high': 0.3}
    ) -> Dict[str, float]:
        """Compute neighbor prediction accuracy at different thresholds"""
        with torch.no_grad():
            # Normalize predictions and targets
            pred_norm = F.normalize(pred_neighbors, dim=-1)
            true_norm = F.normalize(true_neighbors, dim=-1)
            
            # Compute cosine similarity
            similarity = torch.sum(pred_norm * true_norm, dim=-1)  # [B, N]
            
            # Apply mask
            masked_similarity = similarity * neighbor_mask.float()
            
            # Compute accuracy at different thresholds
            metrics = {}
            for name, threshold in thresholds.items():
                correct = (masked_similarity > threshold).float() * neighbor_mask.float()
                accuracy = correct.sum() / (neighbor_mask.sum() + 1e-8)
                metrics[f'neighbor_accuracy_{name}'] = accuracy.item()
                
        return metrics
    
    def _compute_feature_accuracy(
        self,
        pred_features: torch.Tensor,
        true_features: torch.Tensor,
        threshold: float = 0.2
    ) -> Dict[str, float]:
        """Compute feature prediction accuracy"""
        with torch.no_grad():
            # Normalize predictions and targets
            pred_norm = F.normalize(pred_features, dim=-1)
            true_norm = F.normalize(true_features, dim=-1)
            
            # Compute cosine similarity
            similarity = torch.sum(pred_norm * true_norm, dim=-1)  # [B]
            
            # Compute accuracy
            accuracy = (similarity > threshold).float().mean()
            
            return {'feature_accuracy': accuracy.item()}
    
    def __call__(self, predictions: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> tuple:
        """Compute total loss and metrics"""
        try:
            # Validate shapes and get dimensions
            batch_size, max_neighbors, feature_dim = self._validate_shapes(predictions, batch)
            
            # Compute individual losses
            neighbor_loss = self._compute_neighbor_loss(
                predictions['neighbor_preds'],
                batch['neighbor_features'],
                batch['neighbor_mask']
            )
            
            feature_loss = self._compute_feature_loss(
                predictions['feature_preds'],
                batch['features'],
                predictions['confidence']
            )
            
            # Compute total loss
            total_loss = (
                self.loss_weights['neighbor'] * neighbor_loss +
                self.loss_weights['feature'] * feature_loss
            )
            
            # Compute metrics
            metrics = self._compute_metrics(predictions, batch, neighbor_loss, feature_loss)
            
            return total_loss, metrics
            
        except Exception as e:
            logger.error(f"Error computing total loss: {str(e)}")
            raise

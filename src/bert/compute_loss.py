import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class SpatialBertLoss:
    """Loss computation for CellMetaBERT with biological focus"""
    
    def __init__(self, feature_weights=None, loss_weights=None, use_spatial_context=True, 
                 use_gene_context=True, use_cell_state=True):
        self.use_spatial_context = use_spatial_context
        self.use_gene_context = use_gene_context
        self.use_cell_state = use_cell_state
        self.feature_weights = feature_weights or {}
        self.loss_weights = loss_weights or {
            'neighbor': 1.0,
            'feature': 1.0,
            'gene_expression': 1.0,
            'spatial_context': 1.0,
            'cell_state': 1.0,
            'confidence': 1.0
        }
        
        # MSE loss for continuous values
        self.mse = nn.MSELoss()
        # BCE loss for binary values
        self.bce = nn.BCELoss()
        # Cross entropy for multi-class
        self.ce = nn.CrossEntropyLoss()
        
    def compute_neighbor_loss(self, predictions, targets):
        """Compute loss for neighbor predictions"""
        losses = {}
        
        if 'neighbor_location_target' in targets:
            losses['location'] = self.mse(
                predictions['neighbor_location'],
                targets['neighbor_location_target']
            )
        
        if 'neighbor_distance_target' in targets:
            losses['distance'] = self.mse(
                predictions['neighbor_distance'],
                targets['neighbor_distance_target']
            )
        
        if 'interaction_strength_target' in targets:
            losses['interaction'] = self.bce(
                predictions['interaction_strength'],
                targets['interaction_strength_target']
            )
            
        return sum(losses.values()) / len(losses) if losses else 0.0
    
    def compute_spatial_context_loss(self, predictions, targets):
        """Compute loss for spatial context predictions"""
        losses = {}
        
        if 'local_density_target' in targets:
            losses['density'] = self.mse(
                predictions['local_density'],
                targets['local_density_target']
            )
        
        if 'local_heterogeneity_target' in targets:
            losses['heterogeneity'] = self.mse(
                predictions['local_heterogeneity'],
                targets['local_heterogeneity_target']
            )
        
        if 'border_probability_target' in targets:
            losses['border'] = self.bce(
                predictions['border_probability'],
                targets['border_probability_target']
            )
            
        return sum(losses.values()) / len(losses) if losses else 0.0
    
    def compute_feature_loss(self, predictions, targets):
        """Compute weighted loss for feature predictions"""
        if 'features_target' not in targets:
            return 0.0
            
        feature_loss = 0.0
        n_features = 0
        
        pred_features = predictions['features']
        target_features = targets['features_target']
        
        for i, feature_name in enumerate(self.feature_weights.keys()):
            weight = self.feature_weights.get(feature_name, 1.0)
            feature_loss += weight * self.mse(
                pred_features[:, i],
                target_features[:, i]
            )
            n_features += 1
            
        return feature_loss / n_features if n_features > 0 else 0.0
    
    def compute_cell_state_loss(self, predictions, targets):
        """Compute loss for cell state predictions"""
        losses = {}
        
        state_components = [
            'cell_cycle_score', 's_score', 'g2m_score', 'stress_score'
        ]
        
        for component in state_components:
            target_key = f'{component}_target'
            if target_key in targets:
                losses[component] = self.bce(
                    predictions[component],
                    targets[target_key]
                )
                
        return sum(losses.values()) / len(losses) if losses else 0.0
    
    def compute_gene_expression_loss(self, predictions, targets):
        """Compute loss for gene expression predictions"""
        if 'gene_expression_target' not in targets:
            return 0.0
            
        return self.bce(
            predictions['gene_expression_probs'],
            targets['gene_expression_target']
        )
    
    def compute_confidence_loss(self, predictions, targets):
        """Compute loss for confidence scores"""
        if 'confidence_target' not in targets:
            return 0.0
            
        return self.bce(
            predictions['confidence_scores'],
            targets['confidence_target']
        )
    
    def compute_loss(self, predictions, targets):
        """Compute total loss with all components"""
        losses = {}
        
        # Basic losses
        losses['neighbor'] = self.compute_neighbor_loss(predictions, targets) * self.loss_weights.get('neighbor', 1.0)
        losses['feature'] = self.compute_feature_loss(predictions, targets) * self.loss_weights.get('feature', 1.0)
        losses['gene_expression'] = self.compute_gene_expression_loss(predictions, targets) * self.loss_weights.get('gene_expression', 1.0)
        losses['confidence'] = self.compute_confidence_loss(predictions, targets) * self.loss_weights.get('confidence', 1.0)
        
        # Additional biological context losses
        if self.use_spatial_context:
            losses['spatial_context'] = self.compute_spatial_context_loss(predictions, targets) * self.loss_weights.get('spatial_context', 1.0)
            
        if self.use_cell_state:
            losses['cell_state'] = self.compute_cell_state_loss(predictions, targets) * self.loss_weights.get('cell_state', 1.0)
        
        # Convert any float losses to tensors
        losses = {k: torch.tensor(v, device=next(iter(predictions.values())).device, requires_grad=True) 
                 if isinstance(v, float) else v for k, v in losses.items()}
        
        # Compute total loss
        total_loss = sum(losses.values())
        
        # Store individual losses for logging
        losses['total'] = total_loss
        
        return total_loss, losses

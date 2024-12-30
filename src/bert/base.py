import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel
from typing import Dict, Optional, Tuple
import logging
import os

logger = logging.getLogger(__name__)


class CellMetaBERT(BertPreTrainedModel):
    """BERT-based model for spatial transcriptomics analysis with biological focus"""

    def __init__(self, config, feature_info: Dict, debug: bool = False):
        super().__init__(config)
        self.debug = debug

        # Load pre-trained BERT
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Feature dimensions
        self.feature_names = feature_info['feature_names']['features']
        self.n_features = len(self.feature_names)
        self.hvg_names = feature_info['feature_names'].get('hvg_names', [])
        
        # Store feature statistics for normalization
        self.register_buffer('feature_means',
                             torch.tensor([feature_info['feature_stats'][name]['mean'] for name in self.feature_names]))
        self.register_buffer('feature_stds',
                             torch.tensor([feature_info['feature_stats'][name]['std'] for name in self.feature_names]))
        
        # Store spatial context statistics
        if 'spatial_context' in feature_info:
            self.register_buffer('density_mean',
                               torch.tensor(feature_info['spatial_context']['local_density_stats']['mean']))
            self.register_buffer('density_std',
                               torch.tensor(feature_info['spatial_context']['local_density_stats']['std']))
            self.register_buffer('heterogeneity_mean',
                               torch.tensor(feature_info['spatial_context']['heterogeneity_stats']['mean']))
            self.register_buffer('heterogeneity_std',
                               torch.tensor(feature_info['spatial_context']['heterogeneity_stats']['std']))
        
        # Prediction heads
        hidden_size = config.hidden_size
        
        # Enhanced spatial projector with attention to local structure
        self.spatial_projector = nn.Sequential(
            nn.Linear(hidden_size + 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob)
        )
        
        # Spatial context predictor
        self.spatial_context_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(hidden_size // 2, 3)  # [local_density, heterogeneity, border_prob]
        )
        
        # Enhanced neighbor predictor with interaction modeling
        self.neighbor_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(hidden_size // 2, 4)  # [x, y, distance, interaction_strength]
        )
        
        # Feature predictor with biological focus
        self.feature_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(hidden_size // 2, self.n_features)
        )
        
        # Cell state predictor
        self.cell_state_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(hidden_size // 2, 4)  # [cell_cycle, s_score, g2m_score, stress]
        )
        
        # Get gene expression dimensions from config
        self.n_genes = config.n_genes if hasattr(config, 'n_genes') else 5000
        self.n_cells = config.n_cells if hasattr(config, 'n_cells') else 3857
        
        # Memory-efficient gene expression predictor
        self.gene_expression_predictor = nn.Sequential(
            nn.Linear(hidden_size, 2048),
            nn.LayerNorm(2048),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(1024, self.n_genes),
            nn.Sigmoid()
        )
        
        # Enhanced confidence predictor
        self.confidence_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

    def normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """Normalize features using stored statistics"""
        return (features - self.feature_means) / (self.feature_stds + 1e-6)

    def denormalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """Denormalize features back to original scale"""
        return features * self.feature_stds + self.feature_means

    def normalize_spatial_context(self, density: torch.Tensor, heterogeneity: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normalize spatial context features"""
        norm_density = (density - self.density_mean) / (self.density_std + 1e-6)
        norm_heterogeneity = (heterogeneity - self.heterogeneity_mean) / (self.heterogeneity_std + 1e-6)
        return norm_density, norm_heterogeneity

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            spatial_coords: Optional[torch.Tensor] = None,
            features: Optional[torch.Tensor] = None,
            neighbor_indices: Optional[torch.Tensor] = None,
            neighbor_distances: Optional[torch.Tensor] = None,
            neighbor_coords: Optional[torch.Tensor] = None,
            spatial_context: Optional[Dict[str, torch.Tensor]] = None,
            cell_state: Optional[Dict[str, torch.Tensor]] = None,
            original_features: Optional[torch.Tensor] = None,
            original_gene_expression: Optional[torch.Tensor] = None,
            return_dict: bool = True,
            **kwargs
    ) -> Dict:
        # Get device from input tensor
        device = input_ids.device
        
        # Move inputs to device if needed
        attention_mask = attention_mask.to(device) if attention_mask is not None else None
        token_type_ids = token_type_ids.to(device) if token_type_ids is not None else None
        spatial_coords = spatial_coords.to(device) if spatial_coords is not None else None
        features = features.to(device) if features is not None else None
        
        # Get BERT embeddings
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        
        # Get pooled output
        hidden_states = outputs.pooler_output
        
        # Incorporate spatial information if provided
        if spatial_coords is not None:
            # Normalize coordinates to [0, 1]
            spatial_coords = spatial_coords / (torch.max(spatial_coords, dim=0)[0] + 1e-6)
            # Concatenate with hidden states
            hidden_states = torch.cat([hidden_states, spatial_coords], dim=-1)
            # Project back to hidden size
            hidden_states = self.spatial_projector(hidden_states)
        
        # Make predictions
        predictions = {}
        
        # Spatial context predictions
        spatial_context_preds = self.spatial_context_predictor(hidden_states)
        predictions.update({
            'local_density': spatial_context_preds[:, 0],
            'local_heterogeneity': spatial_context_preds[:, 1],
            'border_probability': torch.sigmoid(spatial_context_preds[:, 2])
        })
        
        # Neighbor predictions
        neighbor_preds = self.neighbor_predictor(hidden_states)
        predictions.update({
            'neighbor_location': neighbor_preds[:, :2],
            'neighbor_distance': neighbor_preds[:, 2],
            'interaction_strength': torch.sigmoid(neighbor_preds[:, 3])
        })
        
        # Feature predictions
        if features is not None:
            features = self.normalize_features(features)
        feature_preds = self.feature_predictor(hidden_states)
        predictions['features'] = self.denormalize_features(feature_preds)
        
        # Cell state predictions
        cell_state_preds = self.cell_state_predictor(hidden_states)
        predictions.update({
            'cell_cycle_score': torch.sigmoid(cell_state_preds[:, 0]),
            's_score': torch.sigmoid(cell_state_preds[:, 1]),
            'g2m_score': torch.sigmoid(cell_state_preds[:, 2]),
            'stress_score': torch.sigmoid(cell_state_preds[:, 3])
        })
        
        # Gene expression predictions
        batch_size = hidden_states.shape[0]
        gene_preds = self.gene_expression_predictor(hidden_states)
        gene_preds = gene_preds.unsqueeze(1).expand(-1, self.n_cells, -1)
        predictions['gene_expression_probs'] = gene_preds
        
        # Confidence scores
        predictions['confidence_scores'] = self.confidence_predictor(hidden_states).squeeze(-1)
        
        # Add target values if provided
        if spatial_context is not None:
            predictions.update({
                'local_density_target': spatial_context['local_density'],
                'local_heterogeneity_target': spatial_context['local_heterogeneity'],
                'border_probability_target': spatial_context['border_probability']
            })
        
        if cell_state is not None:
            predictions.update({
                'cell_cycle_score_target': cell_state['cell_cycle_score'],
                's_score_target': cell_state['s_score'],
                'g2m_score_target': cell_state['g2m_score'],
                'stress_score_target': cell_state['stress_score']
            })
        
        if features is not None:
            predictions['features_target'] = features
        
        if original_gene_expression is not None:
            if isinstance(original_gene_expression, torch.Tensor):
                # Normalize gene expression to [0, 1] range
                gene_expr_min = original_gene_expression.min(dim=-1, keepdim=True)[0].min(dim=-1, keepdim=True)[0]
                gene_expr_max = original_gene_expression.max(dim=-1, keepdim=True)[0].max(dim=-1, keepdim=True)[0]
                normalized_gene_expr = (original_gene_expression - gene_expr_min) / (gene_expr_max - gene_expr_min + 1e-6)
                predictions['gene_expression_target'] = normalized_gene_expr
        
        if not return_dict:
            return (
                predictions['neighbor_location'],
                predictions['neighbor_distance'],
                predictions['interaction_strength'],
                predictions['features'],
                predictions['gene_expression_probs'],
                predictions['local_density'],
                predictions['local_heterogeneity'],
                predictions['border_probability'],
                predictions['cell_cycle_score'],
                predictions['confidence_scores']
            )
        
        return predictions

    def save_pretrained(self, save_directory: str):
        """Save model to directory"""
        # Save BERT weights
        self.bert.save_pretrained(save_directory)

        # Save additional components
        state_dict = {
            'spatial_projector': self.spatial_projector.state_dict(),
            'neighbor_predictor': self.neighbor_predictor.state_dict(),
            'feature_predictor': self.feature_predictor.state_dict(),
            'gene_expression_predictor': self.gene_expression_predictor.state_dict(),
            'confidence_predictor': self.confidence_predictor.state_dict(),
            'feature_means': self.feature_means,
            'feature_stds': self.feature_stds
        }
        torch.save(state_dict, f"{save_directory}/cell_meta_bert_heads.pt")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *model_args, **kwargs):
        """Load model from pretrained weights"""
        # Load BERT weights
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        # Load additional components
        heads_path = f"{pretrained_model_name_or_path}/cell_meta_bert_heads.pt"
        if os.path.exists(heads_path):
            device = next(model.parameters()).device
            state_dict = torch.load(heads_path, map_location=device)
            model.spatial_projector.load_state_dict(state_dict['spatial_projector'])
            model.neighbor_predictor.load_state_dict(state_dict['neighbor_predictor'])
            model.feature_predictor.load_state_dict(state_dict['feature_predictor'])
            model.gene_expression_predictor.load_state_dict(state_dict['gene_expression_predictor'])
            model.confidence_predictor.load_state_dict(state_dict['confidence_predictor'])

        return model

import torch
import torch.nn as nn
from transformers import BertModel
import logging
import os
import json

logger = logging.getLogger(__name__)

class CellMetaBERT(nn.Module):
    """
    BERT-based model for spatial transcriptomics analysis with a focus on cell-cell communication.
    Uses gene expressions as input features during training but does not predict them.
    """
    
    def __init__(
        self,
        bert_model_name="bert-base-uncased",
        feature_dim=128,
        spatial_dim=2,
        max_neighbors=10,
        gene_vocab_size=None,
        device="cuda",
        feature_names=None
    ):
        super().__init__()
        self.device = device
        self.max_neighbors = max_neighbors
        self.feature_dim = feature_dim
        self.spatial_dim = spatial_dim
        self.feature_names = feature_names or [f"feature_{i}" for i in range(feature_dim)]
        
        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.bert_dim = self.bert.config.hidden_size
        
        # Spatial embedding layer with normalization
        self.spatial_embedding = nn.Sequential(
            nn.Linear(spatial_dim, self.bert_dim),
            nn.LayerNorm(self.bert_dim)
        )
        
        # Gene expression embedding (for input only)
        if gene_vocab_size:
            self.gene_embedding = nn.Sequential(
                nn.Linear(gene_vocab_size, self.bert_dim),
                nn.LayerNorm(self.bert_dim)
            )
        else:
            self.gene_embedding = None
            
        # Combined predictor for neighbors and features
        self.predictor = nn.Sequential(
            nn.Linear(self.bert_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 4 + feature_dim)  # [x, y, distance, interaction_strength, features...]
        )
        
        # Simple confidence scorer with improved architecture
        self.confidence_scorer = nn.Sequential(
            nn.Linear(self.bert_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Register buffers for input normalization
        self.register_buffer('spatial_mean', torch.zeros(spatial_dim))
        self.register_buffer('spatial_std', torch.ones(spatial_dim))
        if gene_vocab_size:
            self.register_buffer('gene_mean', torch.zeros(gene_vocab_size))
            self.register_buffer('gene_std', torch.ones(gene_vocab_size))
        
        self.to(device)
        
    def _validate_input(self, batch):
        """Validate input batch structure and dimensions"""
        required_keys = ['spatial_coords']
        for key in required_keys:
            if key not in batch:
                raise ValueError(f"Missing required key in batch: {key}")
        
        if batch['spatial_coords'].dim() != 3:
            raise ValueError(f"Expected spatial_coords to be 3D, got {batch['spatial_coords'].dim()}D")
        
        if batch['spatial_coords'].size(-1) != self.spatial_dim:
            raise ValueError(f"Expected spatial_coords to have size {self.spatial_dim} in last dimension")
        
        if 'gene_expression' in batch and self.gene_embedding is None:
            logger.warning("Gene expression provided but model not configured for gene expression input")
            
    def _normalize_input(self, spatial_coords, gene_expression=None):
        """Normalize input data using stored statistics"""
        spatial_coords = (spatial_coords - self.spatial_mean) / (self.spatial_std + 1e-8)
        
        if gene_expression is not None and self.gene_embedding is not None:
            gene_expression = (gene_expression - self.gene_mean) / (self.gene_std + 1e-8)
            
        return spatial_coords, gene_expression
        
    def _prepare_input_embeddings(self, spatial_coords, gene_expression=None):
        """Prepare input embeddings combining spatial and gene expression data"""
        # Normalize inputs
        spatial_coords, gene_expression = self._normalize_input(spatial_coords, gene_expression)
        
        # Embed spatial coordinates
        spatial_emb = self.spatial_embedding(spatial_coords)  # [batch_size, max_neighbors, bert_dim]
        
        if gene_expression is not None and self.gene_embedding is not None:
            # Embed gene expression data
            gene_emb = self.gene_embedding(gene_expression)  # [batch_size, max_neighbors, bert_dim]
            # Combine embeddings with learned weighting
            combined_emb = spatial_emb + gene_emb
        else:
            combined_emb = spatial_emb
            
        return combined_emb
        
    def forward(self, batch):
        """
        Forward pass returning predictions in the required schema format
        
        Args:
            batch: Dictionary containing:
                - spatial_coords: [batch_size, max_neighbors, 2]
                - gene_expression (optional): [batch_size, max_neighbors, gene_vocab_size]
                - neighbor_features: [batch_size, max_neighbors, feature_dim]
                
        Returns:
            Dictionary containing:
                - neighbor_predictions: [batch_size, max_neighbors, 4]  # [x, y, distance, interaction_strength]
                - predicted_features: [batch_size, max_neighbors, feature_dim]
                - feature_confidence: [batch_size, max_neighbors, 1]
        """
        try:
            # Validate input
            self._validate_input(batch)
            batch_size = batch['spatial_coords'].size(0)
            
            # Prepare input embeddings
            input_emb = self._prepare_input_embeddings(
                batch['spatial_coords'],
                batch.get('gene_expression', None)
            )
            
            # Get BERT embeddings
            attention_mask = torch.ones(batch_size, self.max_neighbors, device=self.device)
            bert_output = self.bert(
                inputs_embeds=input_emb,
                attention_mask=attention_mask
            )
            hidden_states = bert_output.last_hidden_state  # [batch_size, max_neighbors, bert_dim]
            
            # Generate predictions
            combined_predictions = self.predictor(hidden_states)  # [batch_size, max_neighbors, 4 + feature_dim]
            
            # Split predictions
            neighbor_predictions = combined_predictions[..., :4]  # [x, y, distance, interaction_strength]
            predicted_features = combined_predictions[..., 4:]  # [feature_dim]
            
            # Generate confidence scores
            feature_confidence = self.confidence_scorer(hidden_states)
            
            return {
                'neighbor_predictions': neighbor_predictions,
                'predicted_features': predicted_features,
                'feature_confidence': feature_confidence
            }
            
        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}")
            raise
            
    def format_predictions(self, predictions, batch):
        """Format predictions for inference time use"""
        try:
            formatted_predictions = {}
            batch_size = predictions['neighbor_predictions'].size(0)
            
            for i in range(batch_size):
                spot_predictions = {
                    'spot_id': batch.get('spot_ids', [i])[i],
                    'neighbors': []
                }
                
                for j in range(self.max_neighbors):
                    neighbor = {
                        'coordinates': {
                            'x': predictions['neighbor_predictions'][i, j, 0].item(),
                            'y': predictions['neighbor_predictions'][i, j, 1].item()
                        },
                        'distance': predictions['neighbor_predictions'][i, j, 2].item(),
                        'interaction_strength': predictions['neighbor_predictions'][i, j, 3].item(),
                        'features': {
                            self.feature_names[k]: {
                                'value': predictions['predicted_features'][i, j, k].item(),
                                'confidence': predictions['feature_confidence'][i, j, 0].item()
                            }
                            for k in range(self.feature_dim)
                        }
                    }
                    spot_predictions['neighbors'].append(neighbor)
                
                formatted_predictions[spot_predictions['spot_id']] = spot_predictions
                
            return formatted_predictions
            
        except Exception as e:
            logger.error(f"Error formatting predictions: {str(e)}")
            raise
            
    def update_normalization_stats(self, spatial_coords, gene_expression=None):
        """Update normalization statistics based on input data"""
        with torch.no_grad():
            self.spatial_mean.copy_(spatial_coords.mean(dim=(0, 1)))
            self.spatial_std.copy_(spatial_coords.std(dim=(0, 1)))
            
            if gene_expression is not None and self.gene_embedding is not None:
                self.gene_mean.copy_(gene_expression.mean(dim=(0, 1)))
                self.gene_std.copy_(gene_expression.std(dim=(0, 1)))
                
    def save_pretrained(self, save_directory: str):
        """Save model to directory"""
        os.makedirs(save_directory, exist_ok=True)
        
        # Save BERT weights
        self.bert.save_pretrained(save_directory)
        
        # Save model configuration
        config = {
            'feature_dim': self.feature_dim,
            'spatial_dim': self.spatial_dim,
            'max_neighbors': self.max_neighbors,
            'feature_names': self.feature_names,
            'normalization_stats': {
                'spatial_mean': self.spatial_mean.cpu().tolist(),
                'spatial_std': self.spatial_std.cpu().tolist()
            }
        }
        
        if self.gene_embedding is not None:
            config['normalization_stats'].update({
                'gene_mean': self.gene_mean.cpu().tolist(),
                'gene_std': self.gene_std.cpu().tolist()
            })
            
        with open(os.path.join(save_directory, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
            
        # Save model state
        torch.save({
            'spatial_embedding': self.spatial_embedding.state_dict(),
            'gene_embedding': self.gene_embedding.state_dict() if self.gene_embedding else None,
            'predictor': self.predictor.state_dict(),
            'confidence_scorer': self.confidence_scorer.state_dict()
        }, os.path.join(save_directory, 'model_state.pt'))
        
    @classmethod
    def from_pretrained(cls, model_path: str, device: str = 'cuda'):
        """Load model from pretrained weights"""
        # Load configuration
        with open(os.path.join(model_path, 'config.json'), 'r') as f:
            config = json.load(f)
            
        # Create model instance
        model = cls(
            bert_model_name=model_path,
            feature_dim=config['feature_dim'],
            spatial_dim=config['spatial_dim'],
            max_neighbors=config['max_neighbors'],
            gene_vocab_size=len(config['normalization_stats'].get('gene_mean', [])) or None,
            device=device,
            feature_names=config['feature_names']
        )
        
        # Load normalization statistics
        stats = config['normalization_stats']
        model.spatial_mean.copy_(torch.tensor(stats['spatial_mean'], device=device))
        model.spatial_std.copy_(torch.tensor(stats['spatial_std'], device=device))
        
        if model.gene_embedding is not None:
            model.gene_mean.copy_(torch.tensor(stats['gene_mean'], device=device))
            model.gene_std.copy_(torch.tensor(stats['gene_std'], device=device))
            
        # Load model state
        state_dict = torch.load(os.path.join(model_path, 'model_state.pt'), map_location=device)
        model.spatial_embedding.load_state_dict(state_dict['spatial_embedding'])
        if state_dict['gene_embedding'] and model.gene_embedding:
            model.gene_embedding.load_state_dict(state_dict['gene_embedding'])
        model.predictor.load_state_dict(state_dict['predictor'])
        model.confidence_scorer.load_state_dict(state_dict['confidence_scorer'])
        
        return model

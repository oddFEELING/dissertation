import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig


class CellMetaBERT(nn.Module):
    """BERT model adapted for metabolic activity prediction with transfer learning"""

    def __init__(
            self,
            bert_model_name="bert-base-uncased",
            n_metabolic_features=4,
            dropout=0.1
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        hidden_size = self.bert.config.hidden_size

        # Spatial position encoding (expects coordinates normalized to [0,1])
        self.spatial_encoder = nn.Sequential(
            nn.Linear(2, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size)
        )

        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.Dropout(dropout),
            nn.ReLU()
        )

        self.metabolic_predictor = nn.Linear(hidden_size // 2, n_metabolic_features)

    def unfreeze_bert_layers(self, num_layers=3):
        """Unfreeze the last n transformer layers for fine-tuning"""
        for param in self.bert.parameters():
            param.requires_grad = False

        for layer in self.bert.encoder.layer[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, input_ids, attention_mask, spatial_coords):
        # BERT embeddings
        outputs = self.bert(input_ids, attention_mask, return_dict=True)
        bert_emb = outputs.last_hidden_state[:, 0, :]

        # Spatial embeddings
        spatial_emb = self.spatial_encoder(spatial_coords)

        # Combine embeddings
        combined = torch.cat([bert_emb, spatial_emb], dim=1)

        # Extract features and predict
        features = self.feature_extractor(combined)
        predictions = self.metabolic_predictor(features)
        return predictions


def compute_metabolic_loss(predictions, true_values, reduction='mean'):
    """Compute MSE loss with optional L1 regularization"""
    mse_loss = F.mse_loss(predictions, true_values, reduction=reduction)
    return mse_loss

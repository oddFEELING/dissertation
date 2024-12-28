import torch
import logging
from rich.console import Console
from rich.logging import RichHandler
from typing import List, Dict
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()]
)
logger = logging.getLogger("SpatialCancerBERT")
console = Console()


class DirectionalPredictor(nn.Module):
    """Predicts neighbor features in a specific direction"""

    def __init__(self, hidden_size: int, n_features: int, n_cancer_genes: int):
        """
        Initialize directional predictor
        :param hidden_size: Size of input features
        :param n_features: Number of features to predict
        :param n_cancer_genes: Number of cancer genes to predict expression for
        """
        super().__init__()

        # Validate inputs
        if hidden_size <= 0 or n_features <= 0 or n_cancer_genes <= 0:
            raise ValueError(
                f"Invalid dimensions: hidden_size={hidden_size}, "
                f"n_features={n_features}, n_cancer_genes={n_cancer_genes}"
            )

        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Prediction heads
        self.existence_prob = nn.Linear(hidden_size // 2, 1)
        self.feature_predictor = nn.Linear(hidden_size // 2, n_features)
        self.gene_predictor = nn.Linear(hidden_size // 2, n_cancer_genes)
        self.confidence = nn.Linear(hidden_size // 2, 1)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for directional prediction"""
        try:
            hidden = self.predictor(features)

            return {
                'probability': torch.sigmoid(self.existence_prob(hidden)),
                'features': self.feature_predictor(hidden),
                'gene_expression': torch.sigmoid(self.gene_predictor(hidden)),
                'confidence': torch.sigmoid(self.confidence(hidden))
            }
        except Exception as e:
            logger.error(f"Error in DirectionalPredictor forward pass: {str(e)}")
            raise


class CellMetaBERT(nn.Module):
    """BERT model for spatial transcriptomics and cancer prediction"""

    def __init__(
            self,
            bert_model_name: str = "bert-base-uncased",
            n_features: int = 7,
            n_cancer_genes: int = 100,  # Set a default value
            dropout: float = 0.1,
            debug: bool = False
    ):
        """
        Initialize SpatialCancerBERT

        :param bert_model_name: Name of pretrained BERT model
        :param n_features: Number of features to predict
        :param n_cancer_genes: Number of cancer genes to predict
        :param dropout: Dropout rate
        :param debug: Enable debug logging
        """
        super().__init__()

        # Set logging level
        if debug:
            logger.setLevel(logging.DEBUG)

        logger.info(f"Initializing SpatialCancerBERT with {n_features} features and {n_cancer_genes} cancer genes")

        # Initialize BERT
        try:
            self.bert = BertModel.from_pretrained(bert_model_name)
            hidden_size = self.bert.config.hidden_size
            logger.debug(f"Loaded BERT model with hidden size {hidden_size}")
        except Exception as e:
            logger.error(f"Failed to load BERT model: {str(e)}")
            raise

        # Spatial position encoding
        self.spatial_encoder = nn.Sequential(
            nn.Linear(2, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size)
        )

        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.ReLU()
        )

        # Directional predictors
        self.directional_predictors = nn.ModuleDict({
            direction: DirectionalPredictor(hidden_size, n_features, n_cancer_genes)
            for direction in ['north', 'south', 'east', 'west']
        })

        # Border cell classifier
        self.border_classifier = nn.Linear(hidden_size, 1)

        # Define feature names
        self.feature_names = [
            'cancer_score',
            'total_counts',
            'n_genes',
            'pct_counts_mito',
            'pct_counts_ribo',
            'total_counts_mito',
            'total_counts_ribo'
        ]

        logger.info("SpatialCancerBERT initialization complete")

    def _validate_inputs(self, input_ids, attention_mask, spatial_coords, spot_ids):
        """Validate input dimensions and types"""
        batch_size = input_ids.size(0)
        assert attention_mask.size(0) == batch_size, "Attention mask batch size mismatch"
        assert spatial_coords.size(0) == batch_size, "Spatial coordinates batch size mismatch"
        assert len(spot_ids) == batch_size, "Spot IDs length mismatch"
        assert spatial_coords.size(1) == 2, "Spatial coordinates should be 2D"

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            spatial_coords: torch.Tensor,
            spot_ids: List[int]
    ) -> Dict:
        """
        Forward pass predicting neighbor features and gene expression

        :param input_ids: Tokenized input sequences
        :param attention_mask: Attention mask for tokens
        :param spatial_coords: Normalized spatial coordinates
        :param spot_ids: List of spot IDs for reference
        :return Dictionary of predictions for each spot
        """
        try:
            logger.debug("Starting forward pass")

            # Validate inputs
            self._validate_inputs(input_ids, attention_mask, spatial_coords, spot_ids)

            # Get BERT embeddings
            outputs = self.bert(input_ids, attention_mask, return_dict=True)
            bert_emb = outputs.last_hidden_state[:, 0, :]
            logger.debug(f"BERT embeddings shape: {bert_emb.shape}")

            # Get spatial embeddings
            spatial_coords = spatial_coords.to(bert_emb.device)
            spatial_emb = self.spatial_encoder(spatial_coords)
            logger.debug(f"Spatial embeddings shape: {spatial_emb.shape}")

            # Combine embeddings
            combined_emb = torch.cat([bert_emb, spatial_emb], dim=1)

            # Extract features
            features = self.feature_extractor(combined_emb)
            logger.debug(f"Extracted features shape: {features.shape}")

            # Border cell prediction
            is_border = torch.sigmoid(self.border_classifier(features))

            # Prepare predictions for each spot
            predictions = {}

            for idx, spot_id in enumerate(spot_ids):
                logger.debug(f"Processing spot {spot_id}")

                # Get predictions for each direction
                spot_features = features[idx].unsqueeze(0)
                neighbor_preds = {}

                for direction, predictor in self.directional_predictors.items():
                    predictor = predictor.to(spot_features.device)
                    neighbor_preds[direction] = predictor(spot_features)

                # Format prediction according to schema
                # TODO: make the schema on google docs into a pydantic one
                try:
                    predictions[spot_id] = {
                        'spot_id': spot_id,
                        'spot_location': {
                            'x': float(spatial_coords[idx, 0].item()),
                            'y': float(spatial_coords[idx, 1].item())
                        },
                        'is_border_cell': bool(is_border[idx] > 0.5),
                        'neighbors': {
                            direction: {
                                'probability': float(preds['probability'].item()),
                                'features': {
                                    name: float(value.item())
                                    for name, value in zip(self.feature_names, preds['features'].squeeze())
                                },
                                'gene_expression': {
                                    'expression_probabilities': [
                                        float(prob) for prob in preds['gene_expression'].squeeze().tolist()
                                    ]
                                }
                            }
                            for direction, preds in neighbor_preds.items()
                        },
                        'prediction_confidence': {
                            direction: float(preds['confidence'].item())
                            for direction, preds in neighbor_preds.items()
                        }
                    }

                    # Add border info if it's a border cell
                    if predictions[spot_id]['is_border_cell']:
                        max_direction = max(
                            neighbor_preds.items(),
                            key=lambda x: x[1]['probability'].item()
                        )[0]
                        predictions[spot_id]['border_info'] = {
                            'boundary_direction': max_direction
                        }

                except Exception as e:
                    logger.error(f"Error formatting prediction for spot {spot_id}: {str(e)}")
                    raise

            logger.debug("Forward pass complete")
            return predictions

        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}")
            raise


def compute_prediction_loss(
        predictions: Dict,
        targets: Dict,
        feature_weight: float = 1.0,
        gene_weight: float = 1.0,
        direction_weight: float = 0.5
) -> Dict:
    """
    Compute loss for predictions

    :param predictions: Model predictions
    :param targets: Target values
    :param feature_weight: Weight for feature prediction loss
    :param gene_weight: Weight for gene expression prediction loss
    :param direction_weight: Weight for directional prediction loss
    :return Dictionary of loss components
    """
    try:
        feature_loss = 0
        gene_loss = 0
        direction_loss = 0

        for spot_id in predictions:
            pred = predictions[spot_id]
            target = targets[spot_id]

            for direction in ['north', 'south', 'east', 'west']:
                if direction in target['neighbors']:
                    # Feature prediction loss
                    feature_loss += F.mse_loss(
                        torch.tensor([pred['neighbors'][direction]['features'][name]
                                      for name in pred['neighbors'][direction]['features']]),
                        torch.tensor([target['neighbors'][direction]['features'][name]
                                      for name in target['neighbors'][direction]['features']])
                    )

                    # Gene expression prediction loss
                    gene_loss += F.binary_cross_entropy(
                        torch.tensor(pred['neighbors'][direction]['gene_expression']['expression_probabilities']),
                        torch.tensor(target['neighbors'][direction]['gene_expression']['expression_probabilities'])
                    )

                # Direction prediction loss
                direction_loss += F.binary_cross_entropy(
                    torch.tensor([pred['neighbors'][direction]['probability']]),
                    torch.tensor([float(direction in target['neighbors'])])
                )

        total_loss = (
                feature_weight * feature_loss +
                gene_weight * gene_loss +
                direction_weight * direction_loss
        )

        return {
            'total_loss': total_loss,
            'feature_loss': feature_loss,
            'gene_loss': gene_loss,
            'direction_loss': direction_loss
        }

    except Exception as e:
        logger.error(f"Error computing loss: {str(e)}")
        raise

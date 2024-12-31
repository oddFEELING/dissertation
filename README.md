# Spatial Transcriptomics BERT Model

A BERT-based model for analyzing spatial transcriptomics data, predicting cell-cell communication, and understanding
spatial gene expression patterns.

## Table of Contents

- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Pipeline Overview](#pipeline-overview)
- [Usage Guide](#usage-guide)
    - [Data Ingestion](#data-ingestion)
    - [Data Preprocessing](#data-preprocessing)
    - [Tokenization](#tokenization)
    - [Training](#training)
- [Model Architecture](#model-architecture)
- [Output Format](#output-format)
- [Neighbor Predictions](#neighbor-predictions)

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On Unix/MacOS
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Data Preparation

1. Create the following directory structure:

```
project/
├── data/
│   ├── raw/                 # Place your raw data files here
│   └── processed/           # Processed data will be stored here
├── src/
│   ├── ingest/             # Data ingestion scripts
│   ├── bert/               # BERT model implementation
│   └── ...
└── checkpoints/            # Model checkpoints will be saved here
```

2. Place your spatial transcriptomics data in the `data/raw/` directory. Supported formats:

- H5AD files
- 10x Genomics Visium data
- Spatial transcriptomics matrices with coordinates

## Pipeline Overview

The pipeline consists of four main stages:

1. Data Ingestion: Convert raw data into standardized format
2. Data Preprocessing: Prepare features and spatial information
3. Tokenization: Convert biological data into BERT-compatible tokens
4. Training: Train the BERT model on the processed data

## Usage Guide

### Data Ingestion

The ingestion pipeline converts your raw data into a standardized H5AD format with required annotations.

```python
from src.ingest.ingest import process_data

# Process raw data
process_data(
    input_path='data/raw/your_data.h5ad',
    output_path='data/processed/processed_data.h5ad',
    tissue_type='brain'  # Specify tissue type
)
```

Required annotations that will be added:

- `tissue_type`: Type of tissue
- `n_genes_by_counts`: Number of genes expressed
- `total_counts`: Total RNA counts
- `pct_counts_mito`: Percentage of mitochondrial genes
- `pct_counts_ribo`: Percentage of ribosomal genes
- `cancer_score`: Cancer likelihood score (if applicable)

### Data Preprocessing

Prepare the data for the BERT model:

```python
from src.bert.data_prep import TransformerDataPrep

# Initialize data preparation
data_prep = TransformerDataPrep(
    adata_path='data/processed/processed_data.h5ad'
)

# Prepare data
combined_tensor, distances, feature_info = data_prep.prepare_data()

# Create processed data dictionary
processed_data = {
    "combined_features": combined_tensor,
    "distance_info": distances,
    "feature_info": feature_info
}
```

This step:

- Computes spatial distances between spots
- Creates feature matrices
- Normalizes gene expression data
- Computes connectivity information

### Tokenization

Convert biological data into tokens for BERT:

```python
from src.bert.tokenizer.tokeniser import IncrementalTissueTokenizer

# Initialize tokenizer
tokenizer = IncrementalTissueTokenizer()

# Add tissue data to tokenizer
tokenizer.add_tissue(
    tissue_type='brain',
    adata=data_prep.adata,
    processed_data=processed_data
)

# Save tokenizer for future use
tokenizer.save()
```

The tokenizer creates sequences that include:

- Spatial location tokens
- Tissue type information
- Gene expression levels
- Cell state indicators
- Neighborhood information

### Training

Train the BERT model:

```python
from src.bert.main import train_model
from src.bert.dataloader import create_st_dataloaders

# Create dataloaders
train_loader, val_loader = create_st_dataloaders(
    adata=data_prep.adata,
    processed_data=processed_data,
    tokenizer=tokenizer,
    batch_size=32
)

# Initialize and train model
model = CellMetaBERT(
    feature_info=processed_data['feature_info'],
    debug=True
)

trainer = SpatialTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    learning_rate=1e-4,
    num_epochs=10,
    save_dir="checkpoints/run_1"
)

trainer.train()
```

Training parameters can be adjusted:

- `batch_size`: Number of spots per batch
- `learning_rate`: Learning rate for optimization
- `num_epochs`: Number of training epochs
- `save_dir`: Directory to save checkpoints

## Model Architecture

The model uses a BERT-based architecture with:

- Pre-trained BERT base model
- Spatial encoding layers
- Direction-specific prediction heads
- Gene expression prediction modules

Key features:

- Attention to spatial relationships
- Cell-cell communication modeling
- Gene expression pattern recognition
- Border cell handling

## Output Format

The model produces predictions in the following format:

```python
prediction = {
    'spot_id': 123,
    'spot_location': {'x': 0.45, 'y': 0.67},
    'is_border_cell': True,
    'neighbors': [  # Array of all predicted neighbors
        {
            'spot_id': 124,
            'location': {'x': 0.48, 'y': 0.70},  # Spatial coordinates
            'distance': 45.2,
            'interaction_strength': 0.85,
            'features': {
                'cancer_score': 0.76,
                'total_counts': 5230,
                'n_genes': 245,
                'pct_counts_mito': 12.3,
                'pct_counts_ribo': 8.7
            },
            'gene_expression': {
                'high_expression_genes': ['TP53', 'BRCA1', 'MYC'],
                'expression_probabilities': {
                    'TP53': 0.85,
                    'BRCA1': 0.72,
                    'MYC': 0.91
                }
            }
        },
        {
            'spot_id': 125,
            'location': {'x': 0.42, 'y': 0.64},
            # ... same structure as above
        },
        # ... array continues for all predicted neighbors
    ]
}
```

For each spot, the model predicts:

- An array of all potential neighbors
- Each neighbor's spatial coordinates
- Feature values and gene expressions
- Interaction strengths
- Border cell characteristics

This array-based approach:

1. Simplifies the output structure
2. Makes it easier to process neighbor relationships
3. Provides direct spatial information
4. Allows for flexible number of neighbors

## Neighbor Predictions

The model predicts all possible neighbors within a configurable radius for each spot, using a radial approach rather
than fixed directions. This allows for more natural and comprehensive cell-cell interaction modeling.

### Neighborhood Definition

For each spot, neighbors are determined by:

1. Spatial distance threshold (configurable)
2. K-nearest neighbors (configurable)
3. Connectivity graph based on Delaunay triangulation

Example visualization:

```
    N2  N3  N4
  N1  ↘↓↙  N5
    ← Spot →
  N8  ↗↑↖  N6
    N7  N6  N5
```

Where N1-N8 represent potential neighbors, with the actual number varying based on:

- Tissue density
- Spatial distribution
- Distance threshold
- K-nearest neighbor parameter

### Neighbor Predictions

For each detected neighbor, the model predicts:

1. Interaction strength (0-1)
2. Feature values
3. Gene expression patterns
4. Cell-cell communication likelihood

Benefits of radial neighbor prediction:

1. More natural tissue representation
2. Captures all possible cell-cell interactions
3. Better handles irregular tissue structures
4. More accurate border detection
5. Scale-invariant predictions

### Configuration Options

The neighborhood prediction can be configured through parameters:

```python
model = CellMetaBERT(
    feature_info=processed_data['feature_info'],
    max_neighbors=12,  # Maximum number of neighbors to predict
    distance_threshold=100,  # Maximum distance in microns
    min_neighbors=3,  # Minimum neighbors to consider
    use_knn=True,  # Use K-nearest neighbors for initial filtering
    debug=True
)
```

The model will:

1. Find all spots within the distance threshold
2. Use KNN to filter to the most relevant neighbors
3. Predict features and gene expressions for each neighbor
4. Return an array of predictions sorted by interaction strength

This simplified approach focuses on:

- Direct spatial relationships
- Actual physical distances
- Real coordinates instead of angles
- Flexible number of neighbors per spot

### Output Format

The model's predictions now include all neighbors:

```python
prediction = {
    'spot_id': 123,
    'spot_location': {'x': 0.45, 'y': 0.67},
    'is_border_cell': True,
    'neighbors': {
        'neighbor_1': {
            'spot_id': 124,
            'distance': 45.2,
            'angle': 45.6,  # Angle in degrees from reference
            'interaction_strength': 0.85,
            'features': {
                'cancer_score': 0.76,
                'total_counts': 5230,
                'n_genes': 245,
                'pct_counts_mito': 12.3,
                'pct_counts_ribo': 8.7
            },
            'gene_expression': {
                'high_expression_genes': ['TP53', 'BRCA1', 'MYC'],
                'expression_probabilities': {
                    'TP53': 0.85,
                    'BRCA1': 0.72,
                    'MYC': 0.91
                }
            }
        },
        'neighbor_2': {
            # Similar structure for other neighbors
        },
        # ... up to max_neighbors
    }
}
```

### Border Detection

Border cells are now identified by:

- Number of neighbors below threshold
- Spatial distribution of neighbors
- Tissue boundary analysis
- Local density patterns

This approach provides a more comprehensive understanding of:

- Cell-cell communication networks
- Tissue organization
- Local microenvironment
- Spatial gene expression patterns

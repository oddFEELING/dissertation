# Spatial Transcriptomics BERT Model

A specialized BERT-based model for analyzing spatial transcriptomics data, focusing on cell-cell communication patterns and spatial gene expression analysis. This model integrates spatial information with gene expression data to understand tissue organization and cell-cell interactions.

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Core Components](#core-components)
- [Usage Guide](#usage-guide)
- [Configuration](#configuration)
- [Model Architecture](#model-architecture)
- [Output Format](#output-format)

## Project Overview

This project implements a BERT-based model specifically designed for spatial transcriptomics data analysis. Key features include:

- Custom tokenization for biological data
- Context-aware masking strategy
- Efficient preprocessing pipeline
- Cell-cell interaction modeling
- Spatial coordinate integration
- Gene expression pattern analysis

## Project Structure

```
project/
├── src/
│   ├── bert/                           # BERT model implementation
│   │   ├── tokenizer/                  # Custom tokenization
│   │   │   ├── tokeniser.py           # Tokenizer implementation
│   │   │   ├── corpus.txt             # Training corpus
│   │   │   ├── corpus_gen.py          # Corpus generation
│   │   │   └── data_prep.json         # Tokenizer configuration
│   │   ├── experiments/               # Training experiments
│   │   │   ├── MLM/                   # Masked Language Model
│   │   │   │   ├── base.py           # Base MLM implementation
│   │   │   │   └── mask.py           # Custom masking strategy
│   │   │   └── neighbourhood_analysis/ # Spatial analysis
│   │   ├── outputs/                   # Model outputs
│   │   │   ├── final_model/          # Production model
│   │   │   └── BEST_MODEL/           # Best performing model
│   │   ├── spatial_bert_data/         # Processed data
│   │   ├── base.py                    # Base model architecture
│   │   ├── compute_loss.py            # Loss functions
│   │   ├── data_prep.py              # Data preprocessing
│   │   ├── dataloader.py             # Data loading utilities
│   │   └── main.py                   # Training entry point
│   ├── ingest/                        # Data ingestion
│   │   ├── __init__.py
│   │   └── _methods.py               # Ingestion methods
│   ├── cancer/                        # Cancer analysis
│   ├── data/                          # Data processing
│   ├── figures/                       # Output figures
│   └── pipeline.py                    # Main pipeline
├── config.yaml                        # Configuration file
├── requirements.txt                   # Dependencies
├── training_res.json                  # Training results
└── .env                              # Environment variables
```

## Installation

1. Clone the repository:

```bash
git clone <https://github.com/oddFEELING/dissertation>
cd <dissertation>
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

## Core Components

### Custom Tokenizer

The `TissueTokenizer` is specifically designed for spatial transcriptomics data:

- Special tokens for biological features:
  - `[TISSUE]`: Tissue type markers
  - `[SPATIAL]`: Spatial coordinates
  - `[CANCER]`: Cancer scores
  - `[GENE]`: Gene markers
  - `[REACT]`: Cell reactivity
  - `[MITO_HIGH/MED/LOW]`: Mitochondrial activity levels
- Preserves important biological tokens:
  - Gene symbols (e.g., `gene_TP53`)
  - Tissue types (e.g., `brain_cancer`)
  - Spatial coordinates
- Custom validation for spatial coordinates (-100 to 100 range)

### Masking Strategy

Context-aware masking with specialized probabilities:

- Special tokens: 5% masking probability
- Tissue types: 20% masking probability
- Numerical values: 15% masking probability
- Gene names: 10% masking probability
- Preserves sequence dependencies
- Maintains structural tokens (`[CLS]`, `[SEP]`, `[PAD]`)

### Data Preprocessing

Efficient preprocessing pipeline with:

- Gene expression normalization (0-1 range)
- Spatial coordinate scaling (-100 to 100)
- Cell interactivity calculations
- Memory-optimized batch processing
- Distance matrix caching
- Vectorized operations

## Usage Guide

### Data Preparation

```python
from src.bert.data_prep import DataPrep

# Initialize preprocessing
prep = DataPrep(adata_path='path/to/data.h5ad')

# Process data
processed_data = prep.prepare_data()
```

### Training

```python
from src.bert.main import train_model

# Train the model
model = train_model(
    processed_data,
    config_path='config.yaml'
)
```

### Detailed Examples

#### 1. Data Loading and Preprocessing

First, initialize the data preparation pipeline. The `DataPrep` class handles normalization of gene expressions, spatial coordinate scaling, and computation of cell-cell interactions.

```python
from src.bert.data_prep import DataPrep

# Initialize data preparation
data_prep = DataPrep(adata_path='data/processed/sample.h5ad')
```

You can customize the preprocessing parameters to match your specific dataset requirements. The following parameters are commonly adjusted:

- `normalize_expression`: Scales gene expression values
- `compute_neighbors`: Enables neighbor detection
- `n_neighbors`: Number of neighbors to consider
- `distance_threshold`: Maximum distance for neighbor relationships

```python
processed_data = data_prep.prepare_data(
    normalize_expression=True,
    compute_neighbors=True,
    n_neighbors=12,
    distance_threshold=100.0
)
```

After preprocessing, create data loaders for training. The `create_st_dataloaders` function handles batch creation and data splitting:

```python
from src.bert.dataloader import create_st_dataloaders

train_loader, val_loader = create_st_dataloaders(
    adata=data_prep.adata,
    processed_data=processed_data,
    batch_size=16,
    val_split=0.2,
    shuffle=True,
    num_workers=4
)
```

Each batch contains multiple components that can be accessed individually:

```python
for batch in train_loader:
    input_ids = batch['input_ids']           # Token IDs
    attention_mask = batch['attention_mask']  # Attention mask
    spatial_coords = batch['spatial_coords']  # Spatial coordinates
    gene_expr = batch['gene_expr']           # Gene expression values
```

#### 2. Tokenizer Usage

The custom tokenizer is designed specifically for spatial transcriptomics data. Initialize it with a base BERT model:

```python
from src.bert.tokenizer.tokeniser import TissueTokenizer

tokenizer = TissueTokenizer(base_model_name='bert-base-uncased')
```

Before processing sequences, add your dataset's gene names to the tokenizer vocabulary:

```python
gene_names = ['TP53', 'BRCA1', 'MYC', 'EGFR']  # Your gene list
tokenizer.add_gene_tokens(gene_names)
```

The tokenizer handles special tokens and formatting for tissue sequences. Here's an example of tokenizing a tissue sequence:

```python
sequence = "[TISSUE] brain_cancer [SPATIAL] -16.49 82.62 [GENE] gene_TP53 0.85"
encoded = tokenizer.tokenizer.encode(
    sequence,
    add_special_tokens=True,
    return_tensors='pt'
)
```

Always validate your token sequences to ensure proper formatting:

```python
tokenizer.validate_token_sequence(sequence)
```

Save your tokenizer for later use or load a previously saved one:

```python
# Save tokenizer
tokenizer.save_tokenizer('tokenizer/_internal')

# Load existing tokenizer
loaded_tokenizer = TissueTokenizer.load_tokenizer('tokenizer/_internal')
```

#### 3. Model Training and Inference

Initialize the model with your desired configuration. The architecture can be customized through these parameters:

```python
from src.bert.base import CellMetaBERT
import torch

model = CellMetaBERT(
    config={
        'hidden_size': 768,
        'num_hidden_layers': 12,
        'num_attention_heads': 12,
        'intermediate_size': 3072,
        'hidden_dropout_prob': 0.1,
        'attention_probs_dropout_prob': 0.1
    }
)
```

Training the model involves setting up the training loop with appropriate parameters:

```python
from src.bert.main import train_model

trainer = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    learning_rate=1e-4,
    num_epochs=10,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    checkpoint_dir='checkpoints/',
    log_interval=100
)
```

For inference, set the model to evaluation mode and prepare your input data:

```python
model.eval()
with torch.no_grad():
    input_data = {
        'input_ids': encoded,
        'attention_mask': torch.ones_like(encoded),
        'spatial_coords': torch.tensor([[-16.49, 82.62]]),
        'gene_expr': torch.tensor([[0.85, 0.0, 0.3, 0.5]])
    }

    outputs = model(**input_data)

    # Access predictions
    cell_states = outputs['cell_states']
    neighbor_preds = outputs['neighbor_predictions']
    gene_expr_preds = outputs['gene_expression_predictions']
```

Save your trained model and load it later for inference:

```python
# Save model
torch.save(model.state_dict(), 'outputs/final_model/model.pt')

# Load model
loaded_model = CellMetaBERT(config=model.config)
loaded_model.load_state_dict(torch.load('outputs/final_model/model.pt'))
```

#### 4. Complete Pipeline

The pipeline combines all components into a single workflow. First, set up your configuration:

```python
from src.pipeline import run_pipeline
from pathlib import Path

pipeline_config = {
    'data_path': 'data/raw/sample.h5ad',
    'output_dir': Path('outputs/'),
    'model_params': {
        'batch_size': 16,
        'learning_rate': 1e-4,
        'num_epochs': 10
    },
    'preprocessing': {
        'normalize': True,
        'n_neighbors': 12,
        'distance_threshold': 100.0
    }
}
```

Run the complete pipeline and access results:

```python
results = run_pipeline(
    config=pipeline_config,
    save_checkpoints=True,
    verbose=True
)

# Access results
trained_model = results['model']
evaluation_metrics = results['metrics']
predictions = results['predictions']
```

The pipeline handles all steps automatically:

1. Data preprocessing
2. Tokenization
3. Model training
4. Evaluation
5. Result collection

## Configuration

Configuration options in `config.yaml`:

```yaml
model_config:
  hidden_size: 768
  num_hidden_layers: 12
  num_attention_heads: 12
  hidden_dropout_prob: 0.1

data_prep:
  max_dist: 100.0
  max_neighbors: 12

training:
  batch_size: 16
  learning_rate: 1e-4
  num_epochs: 10
  val_split: 0.2
```

## Model Architecture

The model uses a BERT-based architecture specifically adapted for spatial transcriptomics data analysis. Here's a detailed breakdown of its components:

### Base Architecture

- **Transformer Encoder**:
  - 12 transformer layers with 768 hidden dimensions
  - 12 attention heads for multi-head attention
  - Custom positional encoding for spatial coordinates
  - Dropout rate of 0.1 for regularization

### Specialized Components

#### 1. Spatial Encoding Layer

- Converts spatial coordinates into continuous representations
- Handles both absolute and relative positions
- Integrates distance-based attention weights
- Supports 2D spatial relationships in tissue context

#### 2. Cell-Cell Communication Module

- Analyzes interactions between neighboring cells
- Features:
  - Distance-weighted attention mechanism
  - Neighbor relationship encoding
  - Interaction strength prediction
  - Border cell detection

#### 3. Gene Expression Module

- Processes gene expression data with:
  - Gene-specific embeddings
  - Expression level normalization
  - Variable gene attention mechanism
  - Tissue-specific gene patterns

#### 4. Attention Mechanisms

- **Spatial Attention**:
  - Distance-based attention weights
  - Neighborhood-aware attention
  - Tissue boundary consideration
- **Gene Attention**:
  - Expression level-based attention
  - Gene co-expression patterns
  - Tissue-specific gene importance

### Training Objectives

1. **Masked Language Modeling (MLM)**:

   - Context-aware masking strategy
   - Different masking probabilities for different token types
   - Special handling of biological tokens

2. **Spatial Prediction**:

   - Neighbor prediction task
   - Distance estimation
   - Border cell classification

3. **Gene Expression Prediction**:
   - Expression level prediction
   - Gene co-expression patterns
   - Cell type marker prediction

### Model Inputs

The model accepts multiple input types:

1. **Tokenized Sequences**:

   - Tissue markers
   - Spatial coordinates
   - Gene expressions
   - Cell state indicators

2. **Spatial Information**:

   - 2D coordinates (scaled to [-100, 100])
   - Distance matrices
   - Neighborhood graphs

3. **Expression Data**:
   - Normalized gene expressions
   - Variable genes
   - Mitochondrial activity
   - Cell reactivity scores

### Model Outputs

The model produces a comprehensive analysis:

1. **Cell-Level Predictions**:

   - Cell state classification
   - Border cell probability
   - Cell reactivity scores
   - Gene expression profiles

2. **Spatial Analysis**:

   - Neighbor predictions
   - Interaction strengths
   - Spatial patterns
   - Tissue organization

3. **Gene Expression Analysis**:
   - Gene expression predictions
   - Co-expression patterns
   - Marker gene identification
   - Expression probabilities

### Performance Optimization

- Efficient attention computation
- Cached distance calculations
- Vectorized operations
- Memory-optimized batch processing
- Custom loss functions for biological relevance

## Output Format

The model produces predictions including:

- Cell-cell interactions
- Gene expression patterns
- Spatial relationships
- Cancer scores
- Cell state indicators

Each prediction includes:

- Spot ID and location
- Neighbor information
- Feature values
- Gene expression probabilities
- Interaction strengths

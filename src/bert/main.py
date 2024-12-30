import torch
import logging
from pathlib import Path
from rich.console import Console
import scanpy as sc
from transformers import BertConfig
import wandb

from src.bert.base import CellMetaBERT
from src.bert.data_prep import TransformerDataPrep
from src.bert.new_tokeniser import IncrementalTissueTokenizer
from src.bert.dataloader import create_st_dataloaders
from src.bert.trainer import SpatialTrainer

# Setup logging
console = Console()
logger = logging.getLogger("SpatialBERT")
logger.setLevel(logging.INFO)

# Configuration
MODEL_CONFIG = {
    'hidden_size': 768,
    'num_hidden_layers': 12,
    'num_attention_heads': 12,
    'intermediate_size': 3072,
    'hidden_dropout_prob': 0.1,
    'attention_probs_dropout_prob': 0.1,
    'use_spatial_context': True,  # Enable spatial context features
    'use_gene_context': True,     # Enable gene context features
    'use_cell_state': True        # Enable cell state features
}

TRAINING_CONFIG = {
    'batch_size': 16,
    'learning_rate': 1e-4,
    'num_epochs': 10,
    'val_split': 0.2,
    'feature_weights': {
        # Basic features
        'cancer_score': 2.0,
        'total_counts': 1.0,
        'n_genes': 1.0,
        'pct_counts_mito': 1.5,
        'pct_counts_ribo': 1.5,
        
        # Spatial context features
        'local_density': 1.5,
        'local_heterogeneity': 1.5,
        'border_probability': 1.0,
        
        # Cell state features
        'cell_cycle_score': 1.5,
        's_score': 1.0,
        'g2m_score': 1.0,
        'stress_score': 1.5,
        'glycolysis_score': 1.0,
        'oxidative_phos_score': 1.0,
        
        # Gene context features
        'mean_neighbor_expr': 1.5
    },
    'loss_weights': {
        'neighbor': 1.0,
        'feature': 1.0,
        'gene_expression': 1.0,
        'spatial_context': 1.5,  # Higher weight for spatial context
        'cell_state': 1.5,      # Higher weight for cell state
        'confidence': 0.5       # Lower weight for confidence
    }
}

DATA_CONFIG = {
    'use_pca': True,
    'max_dist': 100.0,
    'max_neighbors': 12,
    'min_gene_count': 200,     # Minimum number of genes per cell
    'min_cell_count': 3,       # Minimum number of cells expressing a gene
    'normalize_method': 'log1p'  # Normalization method for gene expression
}

# Paths
DATA_PATH = '../ingest/lung_cancer_results.h5ad'  # Path to your data
CHECKPOINT_DIR = 'checkpoints/run_1'  # Where to save model checkpoints

def setup_model_config() -> BertConfig:
    """Setup BERT configuration with biological focus"""
    config = BertConfig(
        vocab_size=30522,  # Standard BERT vocab size
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        **MODEL_CONFIG
    )
    
    # Add custom attributes for biological features
    config.use_spatial_context = MODEL_CONFIG['use_spatial_context']
    config.use_gene_context = MODEL_CONFIG['use_gene_context']
    config.use_cell_state = MODEL_CONFIG['use_cell_state']
    
    return config

def train():
    """Train the model with biological focus"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    console.print(f"[yellow]Using device: {device}[/]")
    
    # Initialize wandb
    if TRAINING_CONFIG.get('use_wandb', True):
        wandb.init(
            project="spatial-bert",
            config={
                "model_config": MODEL_CONFIG,
                "training_config": TRAINING_CONFIG,
                "data_config": DATA_CONFIG
            }
        )
    
    # Load and prepare data
    console.print("[bold blue]Loading and preparing data...[/]")
    data_prep = TransformerDataPrep(
        adata_path=DATA_PATH,
        use_pca=DATA_CONFIG['use_pca'],
        max_dist=DATA_CONFIG['max_dist'],
        max_neighbors=DATA_CONFIG['max_neighbors'],
        device=device
    )
    
    # Get processed data
    combined_tensor, distance_info, feature_info = data_prep.prepare_data()
    
    processed_data = {
        "combined_features": combined_tensor,
        "distance_info": distance_info,
        "feature_info": feature_info
    }
    
    # Initialize tokenizer
    console.print("[bold blue]Initializing tokenizer...[/]")
    tokenizer = IncrementalTissueTokenizer()
    tokenizer.add_tissue(data_prep.adata.obs['tissue_type'], data_prep.adata, processed_data)
    
    # Create dataloaders
    console.print("[bold blue]Creating dataloaders...[/]")
    train_loader, val_loader = create_st_dataloaders(
        adata=data_prep.adata,
        processed_data=processed_data,
        tokenizer=tokenizer,
        batch_size=TRAINING_CONFIG['batch_size'],
        val_split=TRAINING_CONFIG['val_split']
    )
    
    # Initialize model
    console.print("[bold blue]Initializing model...[/]")
    model_config = setup_model_config()
    model = CellMetaBERT(
        config=model_config,
        feature_info=feature_info,
        debug=True  # Set to False in production
    )
    
    # Move model to device
    model = model.to(device)
    
    # Initialize trainer
    console.print("[bold blue]Setting up trainer...[/]")
    trainer = SpatialTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=TRAINING_CONFIG['learning_rate'],
        num_epochs=TRAINING_CONFIG['num_epochs'],
        save_dir=CHECKPOINT_DIR,
        feature_weights=TRAINING_CONFIG['feature_weights'],
        loss_weights=TRAINING_CONFIG['loss_weights'],
        use_wandb=TRAINING_CONFIG.get('use_wandb', True)
    )
    
    # Train model
    console.print("[bold blue]Starting training...[/]")
    trainer.train()
    
    # Save final model
    console.print("[bold blue]Saving model...[/]")
    model.save_pretrained(CHECKPOINT_DIR)
    tokenizer.save(CHECKPOINT_DIR)
    
    console.print("[bold green]Training complete![/]")

def evaluate(model_path: str):
    """Evaluate a trained model"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    console.print(f"[yellow]Using device: {device}[/]")
    
    # Load data
    console.print("[bold blue]Loading data for evaluation...[/]")
    data_prep = TransformerDataPrep(
        adata_path=DATA_PATH,
        use_pca=DATA_CONFIG['use_pca'],
        max_dist=DATA_CONFIG['max_dist'],
        max_neighbors=DATA_CONFIG['max_neighbors']
    )
    combined_tensor, distances, feature_info = data_prep.prepare_data()

    processed_data = {
        "combined_features": combined_tensor,
        "distance_info": distances,
        "feature_info": feature_info
    }

    # Initialize tokenizer and model
    console.print("[bold blue]Loading model and tokenizer...[/]")
    tokenizer = IncrementalTissueTokenizer()
    tokenizer.load()  # Load existing tokenizer

    model_config = setup_model_config()
    model = CellMetaBERT.from_pretrained(
        model_path,
        config=model_config,
        feature_info=feature_info
    )
    
    # Move model to device
    model = model.to(device)

    # Create evaluation dataloader
    _, eval_loader = create_st_dataloaders(
        adata=data_prep.adata,
        processed_data=processed_data,
        tokenizer=tokenizer,
        batch_size=TRAINING_CONFIG['batch_size'],
        val_split=1.0  # Use all data for evaluation
    )

    # Initialize trainer for evaluation
    trainer = SpatialTrainer(
        model=model,
        train_loader=None,
        val_loader=eval_loader,
        save_dir=CHECKPOINT_DIR,
        use_wandb=False
    )

    # Run evaluation
    console.print("[bold blue]Running evaluation...[/]")
    metrics = trainer.validate()
    
    # Display results
    console.print("\n[bold green]Evaluation Results:[/]")
    trainer.display_metrics(metrics, "Evaluation")

def main():
    """Main function to either train or evaluate the model"""
    try:
        # Set this to True to evaluate instead of train
        EVALUATE_MODE = False
        
        if EVALUATE_MODE:
            # Specify the path to your trained model
            MODEL_PATH = f"{CHECKPOINT_DIR}/best_model.pt"
            evaluate(MODEL_PATH)
        else:
            train()
            
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/]")
        logger.exception("An error occurred")
        raise

if __name__ == "__main__":
    # Required for Windows to avoid recursive imports with multiprocessing
    import multiprocessing
    multiprocessing.freeze_support()
    main()

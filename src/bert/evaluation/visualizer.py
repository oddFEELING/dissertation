"""
Visualization module for BERT model evaluation.
Provides complex visualizations and examples for model analysis,
including attention patterns, embeddings, and prediction examples.
"""

import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Any
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import networkx as nx
from tqdm import tqdm

class Visualizer:
    def __init__(self, model, tokenizer, data_module, device):
        self.model = model
        self.tokenizer = tokenizer
        self.data_module = data_module
        self.device = device
        self.model.eval()

    def visualize_attention_patterns(self, num_samples: int = 5) -> Dict[str, Any]:
        """
        Generate attention pattern visualizations.
        
        Args:
            num_samples: Number of examples to visualize
            
        Returns:
            Dictionary containing attention visualizations
        """
        figures = {}
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.data_module.val_dataloader()):
                if batch_idx >= num_samples:
                    break
                    
                outputs = self.model(
                    input_ids=batch['input_ids'].to(self.device),
                    attention_mask=batch['attention_mask'].to(self.device),
                    token_type_ids=batch['token_type_ids'].to(self.device),
                    output_attentions=True
                )
                
                # Get attention weights
                attention = outputs.attentions[-1]  # Last layer attention
                
                # Create attention heatmap
                for head in range(attention.size(1)):
                    fig = go.Figure()
                    
                    # Get tokens for the sequence
                    tokens = self.tokenizer.convert_ids_to_tokens(
                        batch['input_ids'][0].tolist()
                    )
                    
                    # Create heatmap
                    fig.add_trace(go.Heatmap(
                        z=attention[0, head].cpu().numpy(),
                        x=tokens,
                        y=tokens,
                        colorscale='Viridis'
                    ))
                    
                    fig.update_layout(
                        title=f'Attention Pattern - Head {head}',
                        xaxis_title='Target Tokens',
                        yaxis_title='Source Tokens',
                        template='plotly_dark'
                    )
                    
                    figures[f'attention_head_{batch_idx}_{head}'] = fig
        
        return figures

    def visualize_embeddings(self, num_samples: int = 1000) -> Dict[str, Any]:
        """
        Generate embedding space visualizations using dimensionality reduction.
        
        Args:
            num_samples: Number of tokens to visualize
            
        Returns:
            Dictionary containing embedding visualizations
        """
        figures = {}
        embeddings = []
        tokens = []
        
        with torch.no_grad():
            for batch in tqdm(self.data_module.val_dataloader(), desc="Collecting embeddings"):
                outputs = self.model(
                    input_ids=batch['input_ids'].to(self.device),
                    attention_mask=batch['attention_mask'].to(self.device),
                    token_type_ids=batch['token_type_ids'].to(self.device),
                    output_hidden_states=True
                )
                
                # Get last layer embeddings
                batch_embeddings = outputs.hidden_states[-1]
                
                # Collect embeddings and tokens
                for i in range(len(batch['input_ids'])):
                    mask = batch['attention_mask'][i] == 1
                    embeddings.extend(batch_embeddings[i][mask].cpu().numpy())
                    tokens.extend(
                        self.tokenizer.convert_ids_to_tokens(
                            batch['input_ids'][i][mask].tolist()
                        )
                    )
                    
                if len(embeddings) >= num_samples:
                    break
        
        # Truncate to num_samples
        embeddings = embeddings[:num_samples]
        tokens = tokens[:num_samples]
        
        # Apply dimensionality reduction
        # 1. t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=embeddings_2d[:, 0],
            y=embeddings_2d[:, 1],
            mode='markers+text',
            text=tokens,
            textposition="top center",
            hoverinfo='text'
        ))
        
        fig.update_layout(
            title='Token Embeddings (t-SNE)',
            template='plotly_dark',
            showlegend=False
        )
        figures['tsne_embeddings'] = fig
        
        # 2. PCA
        pca = PCA(n_components=3)
        embeddings_3d = pca.fit_transform(embeddings)
        
        fig = go.Figure(data=[go.Scatter3d(
            x=embeddings_3d[:, 0],
            y=embeddings_3d[:, 1],
            z=embeddings_3d[:, 2],
            mode='markers+text',
            text=tokens,
            marker=dict(
                size=5,
                color=embeddings_3d[:, 2],
                colorscale='Viridis',
            )
        )])
        
        fig.update_layout(
            title='Token Embeddings (PCA)',
            template='plotly_dark'
        )
        figures['pca_embeddings'] = fig
        
        return figures

    def visualize_prediction_examples(self, num_examples: int = 10) -> Dict[str, Any]:
        """
        Generate visualizations of interesting prediction examples.
        
        Args:
            num_examples: Number of examples to visualize
            
        Returns:
            Dictionary containing prediction visualizations
        """
        figures = {}
        examples = {
            'high_confidence_correct': [],
            'high_confidence_wrong': [],
            'low_confidence_correct': [],
            'low_confidence_wrong': []
        }
        
        with torch.no_grad():
            for batch in tqdm(self.data_module.val_dataloader(), desc="Finding examples"):
                outputs = self.model(
                    input_ids=batch['input_ids'].to(self.device),
                    attention_mask=batch['attention_mask'].to(self.device),
                    token_type_ids=batch['token_type_ids'].to(self.device)
                )
                
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)
                confidences = torch.max(probs, dim=-1)[0]
                
                # Find interesting examples
                for i in range(len(batch['input_ids'])):
                    mask = batch['attention_mask'][i] == 1
                    seq_preds = predictions[i][mask]
                    seq_conf = confidences[i][mask]
                    seq_true = batch['labels'][i][mask]
                    
                    for j in range(len(seq_preds)):
                        correct = seq_preds[j] == seq_true[j]
                        conf = seq_conf[j].item()
                        
                        example = {
                            'token': self.tokenizer.decode([batch['input_ids'][i][j]]),
                            'prediction': self.tokenizer.decode([seq_preds[j]]),
                            'true': self.tokenizer.decode([seq_true[j]]),
                            'confidence': conf,
                            'context': self.tokenizer.decode(batch['input_ids'][i])
                        }
                        
                        if correct and conf > 0.9:
                            examples['high_confidence_correct'].append(example)
                        elif not correct and conf > 0.9:
                            examples['high_confidence_wrong'].append(example)
                        elif correct and conf < 0.5:
                            examples['low_confidence_correct'].append(example)
                        elif not correct and conf < 0.5:
                            examples['low_confidence_wrong'].append(example)
        
        # Create visualization for each category
        for category, category_examples in examples.items():
            if category_examples:
                # Sort by confidence
                category_examples = sorted(
                    category_examples, 
                    key=lambda x: x['confidence'],
                    reverse=True
                )[:num_examples]
                
                # Create table
                fig = go.Figure(data=[go.Table(
                    header=dict(
                        values=['Token', 'Prediction', 'True', 'Confidence', 'Context'],
                        fill_color='darkblue',
                        align='left'
                    ),
                    cells=dict(
                        values=[
                            [ex['token'] for ex in category_examples],
                            [ex['prediction'] for ex in category_examples],
                            [ex['true'] for ex in category_examples],
                            [f"{ex['confidence']:.3f}" for ex in category_examples],
                            [ex['context'] for ex in category_examples]
                        ],
                        fill_color='darkslategray',
                        align='left'
                    )
                )])
                
                fig.update_layout(
                    title=f'Prediction Examples - {category.replace("_", " ").title()}',
                    template='plotly_dark'
                )
                figures[f'examples_{category}'] = fig
        
        return figures

    def visualize_attention_graph(self, sequence: str) -> Dict[str, Any]:
        """
        Generate graph visualization of attention relationships.
        
        Args:
            sequence: Input sequence to visualize
            
        Returns:
            Dictionary containing attention graph visualization
        """
        # Tokenize input
        inputs = self.tokenizer(
            sequence,
            return_tensors='pt',
            padding=True,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_attentions=True
            )
        
        # Get attention weights from last layer
        attention = outputs.attentions[-1][0]  # [num_heads, seq_len, seq_len]
        
        # Average across attention heads
        avg_attention = attention.mean(dim=0).cpu().numpy()
        
        # Create graph
        G = nx.DiGraph()
        
        # Add nodes
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        for i, token in enumerate(tokens):
            G.add_node(i, token=token)
        
        # Add edges (attention connections)
        for i in range(len(tokens)):
            for j in range(len(tokens)):
                if avg_attention[i, j] > 0.1:  # Threshold for visibility
                    G.add_edge(i, j, weight=float(avg_attention[i, j]))
        
        # Create plot
        fig = plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=1000)
        
        # Draw edges with varying thickness
        edge_weights = [G[u][v]['weight'] * 5 for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=edge_weights, edge_color='gray', arrows=True)
        
        # Add labels
        labels = nx.get_node_attributes(G, 'token')
        nx.draw_networkx_labels(G, pos, labels)
        
        plt.title('Attention Graph Visualization')
        plt.axis('off')
        
        return {'attention_graph': fig}

    def create_interactive_dashboard(self, all_results: Dict[str, Any]) -> go.Figure:
        """
        Create an interactive dashboard combining various visualizations.
        
        Args:
            all_results: Dictionary containing all analysis results
            
        Returns:
            Plotly figure containing the dashboard
        """
        # Create subplot grid
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Token Embeddings',
                'Attention Patterns',
                'Prediction Confidence',
                'Layer Dynamics',
                'Error Analysis',
                'Performance Overview'
            ),
            specs=[[{'type': 'scatter3d'}, {'type': 'heatmap'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}],
                   [{'type': 'bar'}, {'type': 'table'}]]
        )
        
        # Add visualizations to subplots
        # (Add specific visualization code here based on all_results)
        
        fig.update_layout(
            height=1200,
            width=1600,
            title_text="BERT Model Analysis Dashboard",
            showlegend=True,
            template='plotly_dark'
        )
        
        return fig 
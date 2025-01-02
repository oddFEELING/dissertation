"""
Linguistic analysis module for BERT model evaluation.
Provides detailed analysis of linguistic patterns, semantic relationships,
and contextual understanding.
"""

import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from tqdm import tqdm
import nltk
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import spacy

class LinguisticAnalyzer:
    def __init__(self, model, tokenizer, data_module, device):
        self.model = model
        self.tokenizer = tokenizer
        self.data_module = data_module
        self.device = device
        self.model.eval()
        
        # Load spaCy model for advanced linguistic analysis
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            print("Downloading spaCy model...")
            import os
            os.system('python -m spacy download en_core_web_sm')
            self.nlp = spacy.load('en_core_web_sm')

    def analyze_pos_performance(self, num_samples: int = 1000) -> Dict[str, Any]:
        """
        Analyze model performance across different parts of speech.
        
        :param num_samples Number of samples to analyze
            
        :returns Dictionary containing POS-based performance metrics
        """
        pos_predictions = defaultdict(list)
        pos_confidences = defaultdict(list)
        pos_counts = defaultdict(int)
        
        with torch.no_grad():
            for batch in tqdm(self.data_module.val_dataloader(), desc="Analyzing POS performance"):
                # Get model predictions
                outputs = self.model(
                    input_ids=batch['input_ids'].to(self.device),
                    attention_mask=batch['attention_mask'].to(self.device),
                    token_type_ids=batch['token_type_ids'].to(self.device)
                )
                
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)
                confidences = torch.max(probs, dim=-1)[0]
                
                # Process each sequence in the batch
                for i in range(len(batch['input_ids'])):
                    # Decode tokens
                    tokens = self.tokenizer.decode(batch['input_ids'][i], skip_special_tokens=True)
                    
                    # Get POS tags
                    pos_tags = pos_tag(word_tokenize(tokens))
                    
                    # Match predictions with POS tags
                    for j, (word, pos) in enumerate(pos_tags):
                        if j < len(predictions[i]):
                            pos_predictions[pos].append(
                                (predictions[i][j] == batch['labels'][i][j]).item()
                            )
                            pos_confidences[pos].append(confidences[i][j].item())
                            pos_counts[pos] += 1
        
        # Compile results
        results = {
            'pos_accuracy': {
                pos: np.mean(preds) for pos, preds in pos_predictions.items()
            },
            'pos_confidence': {
                pos: np.mean(confs) for pos, confs in pos_confidences.items()
            },
            'pos_counts': pos_counts,
            'pos_error_rate': {
                pos: 1 - np.mean(preds) for pos, preds in pos_predictions.items()
            }
        }
        
        return results

    def analyze_semantic_similarity(self, num_samples: int = 100) -> Dict[str, Any]:
        """
        Analyze semantic relationships between predictions and true tokens.
        
        :param num_samples Number of samples to analyze
        :returns Dictionary containing semantic similarity metrics
        """
        similarities = []
        context_effects = []
        
        with torch.no_grad():
            for batch in tqdm(self.data_module.val_dataloader(), desc="Analyzing semantic similarity"):
                # Get model predictions and embeddings
                outputs = self.model(
                    input_ids=batch['input_ids'].to(self.device),
                    attention_mask=batch['attention_mask'].to(self.device),
                    token_type_ids=batch['token_type_ids'].to(self.device),
                    output_hidden_states=True
                )
                
                # Get last layer embeddings
                embeddings = outputs.hidden_states[-1]
                
                # Calculate semantic similarity
                for i in range(len(batch['input_ids'])):
                    # Get non-padded tokens
                    mask = batch['attention_mask'][i] == 1
                    seq_embeddings = embeddings[i][mask]
                    
                    # Calculate pairwise similarities
                    sim_matrix = cosine_similarity(
                        seq_embeddings.cpu().numpy()
                    )
                    
                    # Store average similarity
                    similarities.append(np.mean(sim_matrix))
                    
                    # Analyze context effect
                    if len(sim_matrix) > 2:
                        context_effect = np.mean(sim_matrix[1:-1]) - np.mean(
                            [sim_matrix[0].mean(), sim_matrix[-1].mean()]
                        )
                        context_effects.append(context_effect)
        
        return {
            'semantic_similarity': {
                'mean': np.mean(similarities),
                'std': np.std(similarities),
                'distribution': list(np.histogram(similarities, bins=20)[0])
            },
            'context_effect': {
                'mean': np.mean(context_effects),
                'std': np.std(context_effects),
                'distribution': list(np.histogram(context_effects, bins=20)[0])
            }
        }

    def analyze_contextual_patterns(self, num_samples: int = 100) -> Dict[str, Any]:
        """
        Analyze how context affects prediction patterns.
        
        :param num_samples Number of samples to analyze
        :returns Dictionary containing contextual pattern analysis
        """
        context_scores = defaultdict(list)
        attention_patterns = []
        
        with torch.no_grad():
            for batch in tqdm(self.data_module.val_dataloader(), desc="Analyzing contextual patterns"):
                # Get model predictions with attention
                outputs = self.model(
                    input_ids=batch['input_ids'].to(self.device),
                    attention_mask=batch['attention_mask'].to(self.device),
                    token_type_ids=batch['token_type_ids'].to(self.device),
                    output_attentions=True
                )
                
                # Analyze attention patterns
                attention = outputs.attentions[-1]  # Last layer attention
                attention_patterns.extend(attention.mean(dim=1).cpu().numpy())
                
                # Process each sequence
                for i in range(len(batch['input_ids'])):
                    # Decode sequence
                    tokens = self.tokenizer.decode(
                        batch['input_ids'][i],
                        skip_special_tokens=True
                    )
                    
                    # Analyze with spaCy
                    doc = self.nlp(tokens)
                    
                    # Analyze dependencies
                    for token in doc:
                        context_scores[token.dep_].append(
                            attention[i, :, token.i].mean().item()
                        )
        
        return {
            'dependency_attention': {
                dep: np.mean(scores) for dep, scores in context_scores.items()
            },
            'attention_patterns': {
                'mean': np.mean(attention_patterns),
                'std': np.std(attention_patterns),
                'distribution': list(np.histogram(
                    np.array(attention_patterns).flatten(), 
                    bins=20
                )[0])
            }
        }

    def plot_linguistic_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate visualizations for linguistic analysis results.
        
        :param results: Dictionary containing linguistic analysis results
        :returns Dictionary containing figure objects
        """
        figures = {}
        
        # 1. POS Performance Plot
        if 'pos_accuracy' in results:
            pos_data = pd.DataFrame({
                'POS': list(results['pos_accuracy'].keys()),
                'Accuracy': list(results['pos_accuracy'].values()),
                'Confidence': list(results['pos_confidence'].values()),
                'Count': list(results['pos_counts'].values())
            })
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=pos_data['POS'],
                y=pos_data['Accuracy'],
                name='Accuracy',
                marker_color='blue'
            ))
            fig.add_trace(go.Bar(
                x=pos_data['POS'],
                y=pos_data['Confidence'],
                name='Confidence',
                marker_color='red'
            ))
            fig.update_layout(
                title='Performance by Part of Speech',
                barmode='group',
                xaxis_title='Part of Speech',
                yaxis_title='Score',
                template='plotly_dark'
            )
            figures['pos_performance'] = fig
        
        # 2. Semantic Similarity Distribution
        if 'semantic_similarity' in results:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=results['semantic_similarity']['distribution'],
                nbinsx=20,
                name='Semantic Similarity'
            ))
            fig.update_layout(
                title='Distribution of Semantic Similarities',
                xaxis_title='Similarity Score',
                yaxis_title='Count',
                template='plotly_dark'
            )
            figures['semantic_similarity'] = fig
        
        # 3. Contextual Patterns Plot
        if 'dependency_attention' in results:
            dep_data = pd.DataFrame({
                'Dependency': list(results['dependency_attention'].keys()),
                'Attention': list(results['dependency_attention'].values())
            })
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=dep_data['Dependency'],
                y=dep_data['Attention'],
                marker_color='green'
            ))
            fig.update_layout(
                title='Attention by Dependency Type',
                xaxis_title='Dependency Type',
                yaxis_title='Average Attention Score',
                template='plotly_dark'
            )
            figures['dependency_attention'] = fig
        
        return figures 
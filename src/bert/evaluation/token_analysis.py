"""
Token-level analysis module for BERT model evaluation.
Provides detailed analysis of token-level predictions, confidence scores,
and error patterns.
"""

import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
from collections import defaultdict
from typing import Dict, List, Tuple, Any
from tqdm import tqdm

class TokenAnalyzer:
    def __init__(self, model, tokenizer, data_module, device):
        self.model = model
        self.tokenizer = tokenizer
        self.data_module = data_module
        self.device = device
        self.model.eval()

    def analyze_token_predictions(self, num_samples: int = 1000) -> Dict[str, Any]:
        """
        Analyze token predictions including accuracy, confidence, and error patterns.
        
        :param num_samples Number of samples to analyze
            
        :returns Dictionary containing various token-level metrics
        """
        results = {}
        predictions = []
        confidences = []
        true_labels = []
        token_positions = []
        
        # Collect predictions
        dataloader = self.data_module.val_dataloader()
        samples_processed = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Analyzing tokens"):
                if samples_processed >= num_samples:
                    break
                    
                outputs = self.model(
                    input_ids=batch['input_ids'].to(self.device),
                    attention_mask=batch['attention_mask'].to(self.device),
                    token_type_ids=batch['token_type_ids'].to(self.device)
                )
                
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                
                # Get predictions and confidences
                batch_preds = torch.argmax(logits, dim=-1)
                batch_conf = torch.max(probs, dim=-1)[0]
                
                # Collect data for non-padded tokens
                mask = batch['attention_mask'] == 1
                
                predictions.extend(batch_preds[mask].cpu().numpy())
                confidences.extend(batch_conf[mask].cpu().numpy())
                true_labels.extend(batch['labels'][mask].numpy())
                
                # Track token positions
                positions = torch.arange(batch['input_ids'].size(1)).expand(batch['input_ids'].size(0), -1)
                token_positions.extend(positions[mask].numpy())
                
                samples_processed += batch['input_ids'].size(0)

        # Convert to numpy arrays
        predictions = np.array(predictions)
        confidences = np.array(confidences)
        true_labels = np.array(true_labels)
        token_positions = np.array(token_positions)

        # Calculate basic metrics
        results['accuracy'] = (predictions == true_labels).mean()
        results['mean_confidence'] = confidences.mean()
        results['confidence_std'] = confidences.std()
        
        # Analyze confidence distribution
        results['confidence_distribution'] = {
            'bins': list(np.linspace(0, 1, 20)),
            'counts': list(np.histogram(confidences, bins=20)[0])
        }
        
        # Position-based analysis
        position_accuracy = defaultdict(list)
        position_confidence = defaultdict(list)
        
        for pos, pred, true, conf in zip(token_positions, predictions, true_labels, confidences):
            position_accuracy[pos].append(pred == true)
            position_confidence[pos].append(conf)
            
        results['position_analysis'] = {
            'position_accuracy': {pos: np.mean(accs) for pos, accs in position_accuracy.items()},
            'position_confidence': {pos: np.mean(confs) for pos, confs in position_confidence.items()}
        }
        
        # Error analysis
        errors_mask = predictions != true_labels
        results['error_analysis'] = {
            'total_errors': errors_mask.sum(),
            'error_rate': errors_mask.mean(),
            'high_confidence_errors': ((confidences > 0.9) & errors_mask).sum(),
            'low_confidence_correct': ((confidences < 0.5) & ~errors_mask).sum()
        }
        
        return results

    def generate_confusion_matrix(self, top_k: int = 20) -> Tuple[np.ndarray, List[str]]:
        """
        Generate confusion matrix for top-k most common tokens.
        
        :param top_k Number of most frequent tokens to include
        :returns Tuple of (confusion matrix, token labels)
        """
        predictions = []
        true_labels = []
        
        # Collect predictions
        dataloader = self.data_module.val_dataloader()
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Generating confusion matrix"):
                outputs = self.model(
                    input_ids=batch['input_ids'].to(self.device),
                    attention_mask=batch['attention_mask'].to(self.device),
                    token_type_ids=batch['token_type_ids'].to(self.device)
                )
                
                batch_preds = torch.argmax(outputs.logits, dim=-1)
                
                # Collect non-padded tokens
                mask = batch['attention_mask'] == 1
                predictions.extend(batch_preds[mask].cpu().numpy())
                true_labels.extend(batch['labels'][mask].numpy())
        
        # Convert to numpy arrays
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        
        # Get top-k most common tokens
        unique_tokens, counts = np.unique(true_labels, return_counts=True)
        top_k_tokens = unique_tokens[np.argsort(counts)[-top_k:]]
        
        # Filter for top-k tokens
        mask = np.isin(true_labels, top_k_tokens)
        filtered_preds = predictions[mask]
        filtered_true = true_labels[mask]
        
        # Generate confusion matrix
        conf_matrix = confusion_matrix(filtered_true, filtered_preds)
        
        # Get token labels
        token_labels = [self.tokenizer.decode([token]) for token in top_k_tokens]
        
        return conf_matrix, token_labels

    def analyze_rare_words(self, rarity_threshold: int = 10) -> Dict[str, Any]:
        """
        Analyze model performance on rare words.
        
        :param rarity_threshold Threshold for considering a word rare
            
        :returns Dictionary containing rare word analysis metrics
        """
        # Count token frequencies in training data
        token_counts = defaultdict(int)
        
        for batch in tqdm(self.data_module.train_dataloader(), desc="Counting token frequencies"):
            for seq in batch['input_ids']:
                for token in seq:
                    token_counts[token.item()] += 1
        
        # Identify rare tokens
        rare_tokens = {token: count for token, count in token_counts.items() 
                      if count < rarity_threshold}
        
        # Analyze performance on rare tokens
        rare_correct = 0
        rare_total = 0
        rare_confidences = []
        
        with torch.no_grad():
            for batch in tqdm(self.data_module.val_dataloader(), desc="Analyzing rare words"):
                outputs = self.model(
                    input_ids=batch['input_ids'].to(self.device),
                    attention_mask=batch['attention_mask'].to(self.device),
                    token_type_ids=batch['token_type_ids'].to(self.device)
                )
                
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)
                confidences = torch.max(probs, dim=-1)[0]
                
                # Identify rare tokens in batch
                rare_mask = torch.tensor([[token.item() in rare_tokens 
                                         for token in seq] 
                                        for seq in batch['input_ids']])
                
                # Update statistics
                rare_correct += ((predictions == batch['labels']) & rare_mask).sum().item()
                rare_total += rare_mask.sum().item()
                rare_confidences.extend(confidences[rare_mask].cpu().numpy())
        
        return {
            'rare_token_count': len(rare_tokens),
            'rare_token_accuracy': rare_correct / rare_total if rare_total > 0 else 0,
            'rare_token_confidence': {
                'mean': np.mean(rare_confidences),
                'std': np.std(rare_confidences),
                'distribution': list(np.histogram(rare_confidences, bins=20)[0])
            },
            'rare_token_examples': [
                {
                    'token': self.tokenizer.decode([token]),
                    'count': count
                }
                for token, count in list(rare_tokens.items())[:10]
            ]
        }

    def plot_token_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate visualizations for token analysis results.
        
        :param results: Dictionary containing token analysis results
            
        :returns Dictionary containing figure objects
        """
        figures = {}
        
        # 1. Confidence Distribution Plot
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=np.array(results['confidence_distribution']['bins'][:-1]) + 0.025,
            y=results['confidence_distribution']['counts'],
            name='Confidence Distribution'
        ))
        fig.update_layout(
            title='Token Prediction Confidence Distribution',
            xaxis_title='Confidence',
            yaxis_title='Count',
            template='plotly_dark'
        )
        figures['confidence_distribution'] = fig
        
        # 2. Position Analysis Plot
        pos_data = results['position_analysis']
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(pos_data['position_accuracy'].keys()),
            y=list(pos_data['position_accuracy'].values()),
            name='Accuracy',
            mode='lines+markers'
        ))
        fig.add_trace(go.Scatter(
            x=list(pos_data['position_confidence'].keys()),
            y=list(pos_data['position_confidence'].values()),
            name='Confidence',
            mode='lines+markers'
        ))
        fig.update_layout(
            title='Token Performance by Position',
            xaxis_title='Position in Sequence',
            yaxis_title='Score',
            template='plotly_dark'
        )
        figures['position_analysis'] = fig
        
        # 3. Error Analysis Plot
        error_data = results['error_analysis']
        fig = go.Figure(data=[
            go.Bar(
                x=['Total Errors', 'High Confidence Errors', 'Low Confidence Correct'],
                y=[error_data['total_errors'], 
                   error_data['high_confidence_errors'],
                   error_data['low_confidence_correct']],
                text=[error_data['total_errors'], 
                      error_data['high_confidence_errors'],
                      error_data['low_confidence_correct']],
                textposition='auto',
            )
        ])
        fig.update_layout(
            title='Error Analysis Overview',
            yaxis_title='Count',
            template='plotly_dark'
        )
        figures['error_analysis'] = fig
        
        return figures 
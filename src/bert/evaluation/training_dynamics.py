"""
Training dynamics analysis module for BERT model evaluation.
Provides detailed analysis of model training behavior, including learning curves,
gradient statistics, and layer-wise learning patterns.
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

class TrainingDynamicsAnalyzer:
    def __init__(self, model, optimizer, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.gradient_history = defaultdict(list)
        self.layer_metrics = defaultdict(lambda: defaultdict(list))

    def analyze_gradients(self) -> Dict[str, Any]:
        """
        Analyze gradient statistics across model parameters.
        
        Returns:
            Dictionary containing gradient analysis metrics
        """
        gradient_stats = defaultdict(dict)
        
        # Collect gradients for each parameter
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad = param.grad.cpu().numpy()
                
                gradient_stats[name] = {
                    'mean': float(np.mean(grad)),
                    'std': float(np.std(grad)),
                    'norm': float(np.linalg.norm(grad)),
                    'max': float(np.max(np.abs(grad))),
                    'sparsity': float(np.mean(grad == 0)),
                    'histogram': list(np.histogram(grad, bins=50)[0])
                }
                
                # Track history
                self.gradient_history[name].append(gradient_stats[name])
        
        return gradient_stats

    def analyze_layer_dynamics(self) -> Dict[str, Any]:
        """
        Analyze learning dynamics at different layers of the model.
        
        Returns:
            Dictionary containing layer-wise analysis metrics
        """
        layer_stats = defaultdict(dict)
        
        # Group parameters by layer
        for name, param in self.model.named_parameters():
            layer_name = name.split('.')[0]  # Get the main layer name
            
            if param.grad is not None:
                # Calculate statistics
                values = param.data.cpu().numpy()
                grads = param.grad.cpu().numpy()
                
                layer_stats[layer_name].update({
                    'weight_norm': float(np.linalg.norm(values)),
                    'gradient_norm': float(np.linalg.norm(grads)),
                    'weight_mean': float(np.mean(values)),
                    'weight_std': float(np.std(values)),
                    'gradient_mean': float(np.mean(grads)),
                    'gradient_std': float(np.std(grads))
                })
                
                # Track metrics history
                for metric, value in layer_stats[layer_name].items():
                    self.layer_metrics[layer_name][metric].append(value)
        
        return layer_stats

    def analyze_attention_dynamics(self) -> Dict[str, Any]:
        """
        Analyze the dynamics of attention mechanisms during training.
        
        Returns:
            Dictionary containing attention analysis metrics
        """
        attention_stats = {}
        
        # Collect attention-related parameters
        attention_params = {
            name: param for name, param in self.model.named_parameters()
            if 'attention' in name
        }
        
        for name, param in attention_params.items():
            if param.grad is not None:
                values = param.data.cpu().numpy()
                grads = param.grad.cpu().numpy()
                
                attention_stats[name] = {
                    'weight_stats': {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'norm': float(np.linalg.norm(values))
                    },
                    'gradient_stats': {
                        'mean': float(np.mean(grads)),
                        'std': float(np.std(grads)),
                        'norm': float(np.linalg.norm(grads))
                    },
                    'pattern_analysis': {
                        'sparsity': float(np.mean(np.abs(values) < 1e-4)),
                        'max_attention': float(np.max(np.abs(values)))
                    }
                }
        
        return attention_stats

    def analyze_learning_rate(self) -> Dict[str, Any]:
        """
        Analyze learning rate behavior if scheduler is available.
        
        Returns:
            Dictionary containing learning rate analysis
        """
        if not self.scheduler:
            return {}
            
        lr_stats = {
            'current_lr': self.scheduler.get_last_lr()[0],
            'initial_lr': self.optimizer.defaults['lr']
        }
        
        # If using AdamW, analyze optimizer parameters
        if isinstance(self.optimizer, torch.optim.AdamW):
            lr_stats.update({
                'beta1': self.optimizer.defaults['betas'][0],
                'beta2': self.optimizer.defaults['betas'][1],
                'weight_decay': self.optimizer.defaults['weight_decay'],
                'epsilon': self.optimizer.defaults['eps']
            })
        
        return lr_stats

    def plot_training_dynamics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate visualizations for training dynamics analysis.
        
        :param results: Dictionary containing training dynamics results
            
        :returns Dictionary containing figure objects
        """
        figures = {}
        
        # 1. Gradient Norm Evolution
        if self.gradient_history:
            fig = go.Figure()
            for param_name, history in self.gradient_history.items():
                norms = [entry['norm'] for entry in history]
                fig.add_trace(go.Scatter(
                    y=norms,
                    name=param_name,
                    mode='lines'
                ))
            fig.update_layout(
                title='Gradient Norm Evolution',
                xaxis_title='Training Step',
                yaxis_title='Gradient Norm',
                template='plotly_dark'
            )
            figures['gradient_evolution'] = fig
        
        # 2. Layer-wise Learning Dynamics
        if self.layer_metrics:
            # Create subplots for different metrics
            metrics = ['weight_norm', 'gradient_norm', 'weight_std', 'gradient_std']
            fig = go.Figure()
            
            for layer_name, metrics_dict in self.layer_metrics.items():
                for metric in metrics:
                    if metric in metrics_dict:
                        fig.add_trace(go.Scatter(
                            y=metrics_dict[metric],
                            name=f"{layer_name}-{metric}",
                            mode='lines'
                        ))
            
            fig.update_layout(
                title='Layer-wise Learning Dynamics',
                xaxis_title='Training Step',
                yaxis_title='Metric Value',
                template='plotly_dark'
            )
            figures['layer_dynamics'] = fig
        
        # 3. Attention Weight Distribution
        if 'attention_dynamics' in results:
            attention_data = []
            for layer_name, stats in results['attention_dynamics'].items():
                attention_data.append({
                    'layer': layer_name,
                    'weight_norm': stats['weight_stats']['norm'],
                    'gradient_norm': stats['gradient_stats']['norm'],
                    'sparsity': stats['pattern_analysis']['sparsity']
                })
            
            if attention_data:
                df = pd.DataFrame(attention_data)
                fig = go.Figure()
                
                # Add traces for different metrics
                for metric in ['weight_norm', 'gradient_norm', 'sparsity']:
                    fig.add_trace(go.Bar(
                        x=df['layer'],
                        y=df[metric],
                        name=metric
                    ))
                
                fig.update_layout(
                    title='Attention Layer Analysis',
                    xaxis_title='Layer',
                    yaxis_title='Metric Value',
                    barmode='group',
                    template='plotly_dark'
                )
                figures['attention_analysis'] = fig
        
        return figures 
"""
Main evaluator module that integrates all analysis components.
"""

from typing import Dict, Any
from pathlib import Path
import json
import torch
from tqdm import tqdm

from .token_analysis import TokenAnalyzer
from .linguistic_analysis import LinguisticAnalyzer
from .training_dynamics import TrainingDynamicsAnalyzer
from .visualizer import Visualizer

class BertEvaluator:
    def __init__(self, 
                 model, 
                 tokenizer, 
                 data_module, 
                 optimizer=None,
                 scheduler=None,
                 device: str = None,
                 output_dir: str = 'results'):
        """
        Initialize the BERT model evaluator.
        
        
        :param model: The trained BERT model
        :param tokenizer: BERT tokenizer
        :param data_module: DataModule containing the datasets
        :param optimizer: Optional optimizer for training dynamics analysis
        :param scheduler: Optional scheduler for training dynamics analysis
        :param device: Device to run evaluations on ('cuda' or 'cpu')
        :param output_dir: Base directory for saving results
        """
        self.model = model
        self.tokenizer = tokenizer
        self.data_module = data_module
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path(output_dir)
        
        # Initialize analysis components
        self.token_analyzer = TokenAnalyzer(model, tokenizer, data_module, self.device)
        self.linguistic_analyzer = LinguisticAnalyzer(model, tokenizer, data_module, self.device)
        self.training_analyzer = TrainingDynamicsAnalyzer(model, optimizer, scheduler)
        self.visualizer = Visualizer(model, tokenizer, data_module, self.device)
        
        # Create output directories
        self.json_dir = self.output_dir / 'result_json'
        self.fig_dir = self.output_dir / 'result_figures'
        
        for dir_path in [self.json_dir, self.fig_dir]:
            for subdir in ['token_analysis', 'linguistic', 'training', 'visualization']:
                (dir_path / subdir).mkdir(parents=True, exist_ok=True)

    def run_comprehensive_evaluation(self, 
                                  num_samples: int = 1000,
                                  save_results: bool = True) -> Dict[str, Any]:
        """
        Run all evaluations and generate visualizations.
        
        :param num_samples: Number of samples to use for analysis
        :param save_results: Whether to save results to disk
            
        :eturns Dictionary containing all evaluation results
        """
        results = {}
        
        print("\n=== Starting Comprehensive Model Evaluation ===")
        
        # 1. Token-Level Analysis
        print("\n1. Running Token-Level Analysis...")
        token_results = self.token_analyzer.analyze_token_predictions(num_samples)
        results['token_analysis'] = token_results
        
        # Generate confusion matrix
        conf_matrix, token_labels = self.token_analyzer.generate_confusion_matrix()
        results['token_analysis']['confusion_matrix'] = {
            'matrix': conf_matrix.tolist(),
            'labels': token_labels
        }
        
        # Analyze rare words
        rare_word_results = self.token_analyzer.analyze_rare_words()
        results['token_analysis']['rare_words'] = rare_word_results
        
        # 2. Linguistic Analysis
        print("\n2. Running Linguistic Analysis...")
        results['linguistic'] = {
            'pos_performance': self.linguistic_analyzer.analyze_pos_performance(num_samples),
            'semantic_similarity': self.linguistic_analyzer.analyze_semantic_similarity(num_samples // 10),
            'contextual_patterns': self.linguistic_analyzer.analyze_contextual_patterns(num_samples // 10)
        }
        
        # 3. Training Dynamics Analysis
        if self.optimizer:
            print("\n3. Analyzing Training Dynamics...")
            results['training'] = {
                'gradients': self.training_analyzer.analyze_gradients(),
                'layer_dynamics': self.training_analyzer.analyze_layer_dynamics(),
                'attention_dynamics': self.training_analyzer.analyze_attention_dynamics(),
                'learning_rate': self.training_analyzer.analyze_learning_rate()
            }
        
        # 4. Generate Visualizations
        print("\n4. Generating Visualizations...")
        
        # Token analysis visualizations
        token_figures = self.token_analyzer.plot_token_analysis(results['token_analysis'])
        
        # Linguistic analysis visualizations
        linguistic_figures = self.linguistic_analyzer.plot_linguistic_analysis(
            results['linguistic']
        )
        
        # Training dynamics visualizations
        if self.optimizer:
            training_figures = self.training_analyzer.plot_training_dynamics(
                results['training']
            )
        
        # Complex visualizations
        attention_figures = self.visualizer.visualize_attention_patterns()
        embedding_figures = self.visualizer.visualize_embeddings(num_samples)
        example_figures = self.visualizer.visualize_prediction_examples()
        
        # Create interactive dashboard
        dashboard = self.visualizer.create_interactive_dashboard(results)
        
        if save_results:
            self._save_results(results, locals())
        
        print("\n=== Evaluation Complete ===")
        return results

    def _save_results(self, results: Dict[str, Any], figures: Dict[str, Any]):
        """Save all results and figures to disk."""
        # Save JSON results
        for category, data in results.items():
            self.save_json(data, f"{category}_results", category)
        
        # Save figures
        figure_categories = [
            ('token_figures', 'token_analysis'),
            ('linguistic_figures', 'linguistic'),
            ('training_figures', 'training'),
            ('attention_figures', 'visualization'),
            ('embedding_figures', 'visualization'),
            ('example_figures', 'visualization')
        ]
        
        for fig_var, subdir in figure_categories:
            if fig_var in figures and figures[fig_var]:
                for name, fig in figures[fig_var].items():
                    self.save_figure(fig, name, subdir)
        
        # Save dashboard
        if 'dashboard' in figures:
            self.save_figure(
                figures['dashboard'],
                'complete_dashboard',
                'visualization'
            )

    def save_json(self, data: Dict, filename: str, subdir: str = None) -> None:
        """Save data as JSON file."""
        save_path = self.json_dir
        if subdir:
            save_path = save_path / subdir
        save_path.mkdir(parents=True, exist_ok=True)
        
        with open(save_path / f"{filename}.json", 'w') as f:
            json.dump(data, f, indent=2)

    def save_figure(self, fig, filename: str, subdir: str = None) -> None:
        """Save matplotlib or plotly figure."""
        save_path = self.fig_dir
        if subdir:
            save_path = save_path / subdir
        save_path.mkdir(parents=True, exist_ok=True)
        
        if hasattr(fig, 'write_html'):  # Plotly figure
            fig.write_html(save_path / f"{filename}.html")
            fig.write_image(save_path / f"{filename}.png")
        else:  # Matplotlib figure
            fig.savefig(save_path / f"{filename}.png", dpi=300, bbox_inches='tight')
            plt.close(fig) 
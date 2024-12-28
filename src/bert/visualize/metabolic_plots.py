class MetabolicVisualizer:
    def __init__(self, adata, predictions, true_values, feature_names):
        self.adata = adata
        self.predictions = predictions
        self.true_values = true_values
        self.feature_names = feature_names
        
    def plot_feature_correlations(self, save_path=None):
        """Plot correlation heatmap between predicted and true values"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        correlations = np.zeros((len(self.feature_names), len(self.feature_names)))
        for i, feat1 in enumerate(self.feature_names):
            for j, feat2 in enumerate(self.feature_names):
                correlations[i, j] = np.corrcoef(
                    self.predictions[:, i],
                    self.true_values[:, j]
                )[0, 1]
        
        sns.heatmap(
            correlations,
            xticklabels=self.feature_names,
            yticklabels=self.feature_names,
            annot=True,
            cmap='coolwarm',
            ax=ax
        )
        
        plt.title('Metabolic Feature Correlations')
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_spatial_metabolic_patterns(self, feature_idx, save_path=None):
        """Plot spatial patterns of metabolic features"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # True values
        scatter1 = ax1.scatter(
            self.adata.obsm['spatial'][:, 0],
            self.adata.obsm['spatial'][:, 1],
            c=self.true_values[:, feature_idx],
            cmap='viridis',
            s=50
        )
        ax1.set_title(f'True {self.feature_names[feature_idx]}')
        plt.colorbar(scatter1, ax=ax1)
        
        # Predicted values
        scatter2 = ax2.scatter(
            self.adata.obsm['spatial'][:, 0],
            self.adata.obsm['spatial'][:, 1],
            c=self.predictions[:, feature_idx],
            cmap='viridis',
            s=50
        )
        ax2.set_title(f'Predicted {self.feature_names[feature_idx]}')
        plt.colorbar(scatter2, ax=ax2)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_prediction_accuracy(self, save_dir=None):
        """Plot scatter plots for each metabolic feature"""
        n_features = len(self.feature_names)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten()
        
        for i, feature in enumerate(self.feature_names):
            ax = axes[i]
            ax.scatter(
                self.true_values[:, i],
                self.predictions[:, i],
                alpha=0.5
            )
            ax.plot(
                [self.true_values[:, i].min(), self.true_values[:, i].max()],
                [self.true_values[:, i].min(), self.true_values[:, i].max()],
                'r--'
            )
            ax.set_title(feature)
            ax.set_xlabel('True Values')
            ax.set_ylabel('Predicted Values')
            
        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'prediction_accuracy.png'))
        plt.show() 
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from rich.console import Console

console = Console()


class MetabolicTrainer:
    def __init__(
            self,
            model,
            train_dataset,
            val_dataset,
            device="cuda" if torch.cuda.is_available() else "cpu",
            learning_rate=1e-4,
            num_epochs=10,
            warmup_epochs=2,
            checkpoint_dir="checkpoints/metabolic"
    ):
        self.model = model.to(device)
        self.device = device
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.num_epochs = num_epochs
        self.warmup_epochs = warmup_epochs
        self.checkpoint_dir = checkpoint_dir

        self.optimizer = AdamW(
            [{'params': [p for n, p in self.model.named_parameters() if
                         not any(nd in n for nd in ['bias', 'LayerNorm.weight'])],
              'weight_decay': 0.01},
             {'params': [p for n, p in self.model.named_parameters() if
                         any(nd in n for nd in ['bias', 'LayerNorm.weight'])],
              'weight_decay': 0.0}],
            lr=learning_rate
        )

        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=learning_rate,
            epochs=num_epochs,
            steps_per_epoch=len(train_dataset),
            pct_start=warmup_epochs / num_epochs
        )

        self.best_val_loss = float('inf')

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_dataset)

        with console.status("[bold blue]Training...") as status:
            for batch_idx, batch in enumerate(self.train_dataset):
                self.optimizer.zero_grad()

                # Move all batch items to device at once
                batch = {k: v.to(self.device) for k, v in batch.items()}

                predictions = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    spatial_coords=batch['spatial_coords']
                )

                loss = torch.nn.functional.mse_loss(
                    predictions,
                    batch['metabolic_features']
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()
                self.scheduler.step()

                total_loss += loss.item()

                # Update progress every 10 batches
                if batch_idx % 10 == 0:
                    avg_loss = total_loss / (batch_idx + 1)
                    status.update(f"[cyan]Batch {batch_idx}/{num_batches} "
                                f"| Loss: {avg_loss:.4f} "
                                f"| LR: {self.scheduler.get_last_lr()[0]:.2e}[/]")

            if epoch == self.warmup_epochs:
                console.print("[yellow]🔓 Unfreezing BERT layers...[/]")
                self.model.unfreeze_bert_layers()

        return total_loss / num_batches

    def validate(self):
        self.model.eval()
        total_loss = 0
        predictions_list = []
        targets_list = []

        with torch.no_grad():
            for batch in self.val_dataset:
                predictions = self.model(
                    input_ids=batch['input_ids'].to(self.device),
                    attention_mask=batch['attention_mask'].to(self.device),
                    spatial_coords=batch['spatial_coords'].to(self.device)
                )

                targets = batch['metabolic_features'].to(self.device)
                loss = torch.nn.functional.mse_loss(predictions, targets)

                total_loss += loss.item()
                predictions_list.append(predictions.cpu())
                targets_list.append(targets.cpu())

        all_preds = torch.cat(predictions_list)
        all_targets = torch.cat(targets_list)

        return {
            'val_loss': total_loss / len(self.val_dataset),
            'r2_score': self._calculate_r2(all_preds, all_targets),
            'correlations': self._calculate_correlations(all_preds, all_targets)
        }

    def _calculate_r2(self, predictions, targets):
        r2_scores = {}
        for i in range(predictions.shape[1]):
            ss_tot = torch.sum((targets[:, i] - targets[:, i].mean()) ** 2)
            ss_res = torch.sum((targets[:, i] - predictions[:, i]) ** 2)
            r2_scores[f'feature_{i}'] = 1 - (ss_res / ss_tot).item()
        return r2_scores

    def _calculate_correlations(self, predictions, targets):
        correlations = {}
        for i in range(predictions.shape[1]):
            correlation = torch.corrcoef(torch.stack([predictions[:, i], targets[:, i]]))[0, 1].item()
            correlations[f'feature_{i}'] = correlation
        return correlations

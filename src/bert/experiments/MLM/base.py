import torch
from transformers import BertForMaskedLM, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.nn import functional as F
import os
from pathlib import Path
from tqdm import tqdm
import random
import numpy as np
from src.bert.experiments.MLM.mask import MaskingStrategy


class MLMTrainer:
    def __init__(self,
                 model_name: str,
                 data_module,
                 device: str = 'cuda',
                 lr: float = 2e-5
                 ):
        print('\n\n------------ MLM trainer module started  --\n')
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'--> Using device: {self.device}')

        # Initialize model and move to device
        self.model = BertForMaskedLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.resize_token_embeddings(len(data_module.train_dataset.tokenizer.tokenizer))

        self.data_module = data_module
        self.lr = lr
        
        # Initialize masking strategy with mask token ID from tokenizer
        self.masking_strategy = MaskingStrategy(
            data_module.train_dataset.tokenizer.tokenizer,
            data_module.train_dataset.tokenizer.tokenizer.mask_token_id
        )

    def prepare_batch(self, batch):
        """Apply custom masking to the batch"""
        # Move input tensors to the same device as the model
        input_ids = batch['input_ids'].clone().to(self.device)
        attention_mask = batch['attention_mask'].clone().to(self.device)
        token_type_ids = batch['token_type_ids'].clone().to(self.device)
        
        masked_input_ids = []
        labels = []
        
        # Apply masking to each sequence in the batch
        for sequence in input_ids:
            masked_sequence, sequence_labels = self.masking_strategy.apply_masking(sequence)
            masked_input_ids.append(masked_sequence)
            labels.append(sequence_labels)
            
        return {
            'input_ids': torch.stack(masked_input_ids),  # Already on correct device
            'attention_mask': attention_mask,            # Already on correct device
            'token_type_ids': token_type_ids,           # Already on correct device
            'labels': torch.stack(labels).to(self.device)
        }

    def train(self, num_epochs: int = 10, output_dir: str = 'outputs', save_steps: int = 1000, resume_from_checkpoint: str = None):
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get data loaders
        train_dataloader = self.data_module.train_dataloader()
        val_dataloader = self.data_module.val_dataloader()

        # Setup optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        total_steps = len(train_dataloader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_steps // 10,
            num_training_steps=total_steps
        )

        # Load checkpoint if specified
        start_epoch = 0
        step = 0
        best_val_loss = float('inf')
        if resume_from_checkpoint:
            print(f'\n--> Resuming from checkpoint: {resume_from_checkpoint}')
            checkpoint_path = Path(resume_from_checkpoint)
            if checkpoint_path.exists():
                # Extract step number from checkpoint directory name
                step = int(checkpoint_path.name.split('-')[1])
                # Calculate starting epoch
                start_epoch = step // len(train_dataloader)
                
                # Load model state
                self.model = BertForMaskedLM.from_pretrained(checkpoint_path)
                self.model.to(self.device)
                
                print(f'--> Resuming from step {step} (epoch {start_epoch + 1})')
            else:
                raise ValueError(f"Checkpoint path {resume_from_checkpoint} does not exist")

        print('\n\n------------ Starting Training Loop  --\n')
        print('--------------------------------------------')
        print(f'--> Total training steps: {total_steps}')
        print(f'--> Warmup steps: {total_steps // 10}')
        print(f'--> Initial learning rate: {self.lr}')
        print(f'--> Number of parameters: {sum(p.numel() for p in self.model.parameters())}')
        print('--------------------------------------------\n')
        
        best_val_loss = float('inf')
        step = 0
        
        # Track metrics
        epoch_metrics = []

        for epoch in range(num_epochs):
            print(f'\n=== Epoch {epoch + 1}/{num_epochs} ===')
            epoch_start_lr = scheduler.get_last_lr()[0]
            print(f'--> Current learning rate: {epoch_start_lr:.2e}')

            # Training
            self.model.train()
            train_losses = []
            train_bar = tqdm(train_dataloader, desc='Training', leave=False)
            
            # Track masking statistics
            total_tokens = 0
            masked_tokens = 0

            for batch in train_bar:
                # Apply custom masking
                masked_batch = self.prepare_batch(batch)
                
                # Update masking statistics
                total_tokens += torch.sum(masked_batch['attention_mask']).item()
                masked_tokens += torch.sum(masked_batch['labels'] != -100).item()
                
                # Forward pass
                outputs = self.model(
                    input_ids=masked_batch['input_ids'],
                    attention_mask=masked_batch['attention_mask'],
                    labels=masked_batch['labels'],
                    token_type_ids=masked_batch['token_type_ids'],
                )
                loss = outputs.loss
                train_losses.append(loss.item())

                # Backward pass
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # Update progress bar with current loss and learning rate
                train_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}"
                })
                step += 1

                if step % save_steps == 0:
                    checkpoint_dir = output_dir / f'checkpoint-{step}'
                    self.save_model(checkpoint_dir)
                    print(f'\n--> Checkpoint saved at step {step}')

            # Calculate training statistics
            avg_train_loss = sum(train_losses) / len(train_losses)
            masking_percentage = (masked_tokens / total_tokens) * 100

            # Validation
            val_loss, val_metrics = self.evaluate(dataloader=val_dataloader)
            
            # Log epoch statistics
            print('\nEpoch Statistics:')
            print('------------------')
            print(f'Training:')
            print(f'  - Average loss: {avg_train_loss:.4f}')
            print(f'  - Tokens masked: {masked_tokens:,} / {total_tokens:,} ({masking_percentage:.2f}%)')
            print(f'  - Learning rate: {scheduler.get_last_lr()[0]:.2e}')
            print(f'Validation:')
            print(f'  - Loss: {val_loss:.4f}')
            print(f'  - Perplexity: {torch.exp(torch.tensor(val_loss)):.2f}')
            print(f'  - Accuracy: {val_metrics["accuracy"]:.2f}%')
            print(f'  - Masked token accuracy: {val_metrics["masked_accuracy"]:.2f}%')

            # Track best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(output_dir / "BEST_MODEL")
                print(f'--> New best model saved! (validation loss: {val_loss:.4f})')
            
            # Store epoch metrics
            epoch_metrics.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
                'val_perplexity': torch.exp(torch.tensor(val_loss)).item(),
                'val_accuracy': val_metrics["accuracy"],
                'val_masked_accuracy': val_metrics["masked_accuracy"],
                'masking_percentage': masking_percentage,
                'learning_rate': scheduler.get_last_lr()[0]
            })

        # Save final model and training metrics
        self.save_model(output_dir / 'final_model')
        
        # Save training metrics
        import json
        with open(output_dir / 'training_metrics.json', 'w') as f:
            json.dump(epoch_metrics, f, indent=2)
        
        print('\n\n------------ Training Loop Finished  --\n')
        print(f'Best validation loss: {best_val_loss:.4f}')
        print(f'Final learning rate: {scheduler.get_last_lr()[0]:.2e}')
        print(f'Training metrics saved to {output_dir/"training_metrics.json"}')

    def evaluate(self, dataloader):
        """Evaluate the model on the provided dataloader"""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_masked_correct = 0
        total_tokens = 0
        total_masked_tokens = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Evaluating', leave=False):
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels'],
                    token_type_ids=batch['token_type_ids'],
                )
                
                loss = outputs.loss
                logits = outputs.logits
                labels = batch['labels']
                
                # Calculate accuracy
                predictions = torch.argmax(logits, dim=-1)
                mask = labels != -100
                
                # Overall accuracy
                correct = (predictions == labels) & mask
                total_correct += correct.sum().item()
                total_tokens += mask.sum().item()
                
                # Masked token accuracy (using tokenizer's mask_token_id)
                mask_token_id = self.data_module.train_dataset.tokenizer.tokenizer.mask_token_id
                masked_correct = correct & (batch['input_ids'] == mask_token_id)
                total_masked_correct += masked_correct.sum().item()
                total_masked_tokens += (batch['input_ids'] == mask_token_id).sum().item()
                
                total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        accuracy = (total_correct / total_tokens) * 100 if total_tokens > 0 else 0
        masked_accuracy = (total_masked_correct / total_masked_tokens) * 100 if total_masked_tokens > 0 else 0

        return avg_loss, {
            "accuracy": accuracy,
            "masked_accuracy": masked_accuracy
        }

    def save_model(self, output_dir: str):
        """Saves a model to specified path"""
        self.model.save_pretrained(output_dir)
        print(f'--> model saved to {output_dir}')

    @classmethod
    def load_model(cls, model_path: str, data_module, device: str = None):
        """Loads a saved model from the specified path"""
        trainer = cls('bert-base-uncased', data_module, device)
        trainer.model = BertForMaskedLM.from_pretrained(model_path)
        trainer.model.to(device)
        return trainer

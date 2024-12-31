import torch
from transformers import BertForMaskedLM, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.nn import functional as F
import os
from pathlib import Path
from tqdm import tqdm


class MLMTrainer:
    def __init__(self,
                 model_name: str,
                 data_module,
                 device: str = 'cuda',
                 lr: float = 2e-5
                 ):
        # Setup device

        print('\n\n------------ MLM trainer module started  --\n')
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'--> Using device: {self.device}')

        # Initialize model and move to device
        self.model = BertForMaskedLM.from_pretrained(model_name)
        self.model.to(self.device)

        self.data_module = data_module
        self.lr = lr  # learning rate

    def train(self, num_epochs: int = 10, output_dir: str = 'outputs', save_steps: int = 1000):
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

        # Training loop
        print('\n\n------------ Starting Training Loop  --\n')
        print('--------------------------------------------\n')
        best_val_loss = float('inf')
        step = 0

        for epoch in range(num_epochs):
            print(f'--> Epoch {epoch + 1}/{num_epochs}')

            # training
            self.model.train()
            train_loss = 0.0
            train_bar = tqdm(train_dataloader, desc='Training')

            for batch in train_bar:
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                loss = outputs.loss
                train_loss = loss.item()

                # Backward pass
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # Update the progress bar
                train_bar.set_postfix({"loss": f"{train_loss:.4f}"})
                step += 1

                # Save checkpoint
                if step % save_steps == 0:
                    checkpoint_dir = output_dir / f'checkpoint-{step}'
                    self.save_model(checkpoint_dir)

            avg_train_loss = train_loss / len(train_dataloader)

            # Validation
            val_loss = self.evaluate(val_dataloader=val_dataloader)

            print(f'Average training loss: {avg_train_loss:.4f}')
            print(f'Validation loss: {val_loss:.4f}')

            # save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(output_dir / "BEST_MODEL")

        # Save final model
        self.save_model(output_dir / 'final_model')
        print('\n\n------------ Training Loop Finished  --\n')

    def evaluate(self, dataloader):
        """Evaluate the model on the provided dataloader"""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Evaluating'):
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                loss = outputs.loss
                total_loss += loss.item()

        return total_loss / len(dataloader)

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

"""
Training loop implementations for fluProfiler.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from .early_stopping import EarlyStopping


class Trainer:
    """
    Basic trainer class for fluProfiler models.
    """
    def __init__(self, model, optimizer, scheduler=None, device='cuda', mixed_precision=False):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.mixed_precision = mixed_precision

        if mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()

        self.model.to(device)

    def train_epoch(self, train_loader):
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch in tqdm(train_loader, desc='Training'):
            self.optimizer.zero_grad()

            # Move batch to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                    for k, v in batch.items()}

            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**batch)
                    loss = outputs[0]

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(**batch)
                loss = outputs[0]
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if self.scheduler:
                self.scheduler.step()

        return total_loss / num_batches

    def validate(self, val_loader):
        """
        Validate model on validation set.

        Args:
            val_loader: Validation data loader

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validating'):
                # Move batch to device
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                        for k, v in batch.items()}

                outputs = self.model(**batch)
                loss = outputs[0]

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def train(self, train_loader, val_loader, epochs, early_stopping=None, save_path=None):
        """
        Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
            early_stopping: EarlyStopping instance (optional)
            save_path: Path to save best model (optional)

        Returns:
            Training history
        """
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")

            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            print(".4f")
            print(".4f")

            # Save best model
            if val_loss < best_val_loss and save_path:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_path)
                print(f"Saved best model with validation loss: {val_loss:.4f}")

            # Early stopping
            if early_stopping:
                early_stopping(val_loss, self.model)
                if early_stopping.early_stop:
                    print("Early stopping triggered")
                    break

        return history
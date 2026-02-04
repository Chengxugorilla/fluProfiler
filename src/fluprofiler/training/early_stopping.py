"""
Early stopping implementation for fluProfiler training.
"""

import torch
import numpy as np
from datetime import datetime


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.
    """
    def __init__(self, patience=7, verbose=True, delta=0, trace_func=print, save_dir=None):
        """
        Initialize early stopping.

        Args:
            patience (int): Number of epochs to wait before early stopping
            verbose (bool): Whether to print messages
            delta (float): Minimum change to qualify as improvement
            trace_func: Function to use for printing messages
            save_dir: Directory to save model checkpoints
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = np.inf
        self.early_stop = False
        self.delta = delta
        self.save_dir = save_dir
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        """
        Check if early stopping should be triggered.

        Args:
            val_loss (float): Current validation loss
            model: Model to save if validation improves
        """
        if self.best_score > val_loss:
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            self.best_score = val_loss
        else:
            self.counter += 1
            self.trace_func(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model):
        """
        Save model checkpoint when validation loss decreases.

        Args:
            val_loss (float): Current validation loss
            model: Model to save
        """
        if self.verbose:
            self.trace_func(
                f'Validation MSE decrease ({self.best_score:.6f} --> {val_loss:.6f}).  Saving model ...')

        current_time = datetime.now()

        # Format current time as string: 2024-06-09_12-34-56
        checkpoint_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        torch.save(model, self.save_dir + checkpoint_name + '.pth')
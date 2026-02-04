"""
Optimizers and learning rate schedulers for fluProfiler.
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ExponentialLR


def create_optimizer(model, optimizer_name='adam', lr=1e-3, weight_decay=1e-4, **kwargs):
    """
    Create optimizer for model parameters.

    Args:
        model: PyTorch model
        optimizer_name: Name of optimizer ('adam', 'adamw', 'sgd')
        lr: Learning rate
        weight_decay: Weight decay coefficient
        **kwargs: Additional optimizer arguments

    Returns:
        Optimizer instance
    """
    if optimizer_name.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    elif optimizer_name.lower() == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    elif optimizer_name.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def create_scheduler(optimizer, scheduler_name='cosine', **kwargs):
    """
    Create learning rate scheduler.

    Args:
        optimizer: PyTorch optimizer
        scheduler_name: Name of scheduler ('cosine', 'step', 'exponential')
        **kwargs: Scheduler-specific arguments

    Returns:
        Scheduler instance
    """
    if scheduler_name.lower() == 'cosine':
        return CosineAnnealingLR(optimizer, **kwargs)
    elif scheduler_name.lower() == 'step':
        return StepLR(optimizer, **kwargs)
    elif scheduler_name.lower() == 'exponential':
        return ExponentialLR(optimizer, **kwargs)
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")
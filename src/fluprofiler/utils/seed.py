"""
Random seed management utilities.
"""

import torch
import numpy as np
import random
import os


def set_seed(seed=42):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variable for python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_seed():
    """
    Get current random seed.

    Returns:
        Current random seed
    """
    return os.environ.get('PYTHONHASHSEED', 'Not set')
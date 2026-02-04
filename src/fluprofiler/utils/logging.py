"""
Logging utilities for fluProfiler.
"""

import logging
import os
from datetime import datetime


def setup_logger(name='fluProfiler', log_file=None, level=logging.INFO):
    """
    Set up logger with file and console handlers.

    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_experiment_logger(experiment_name, log_dir='logs'):
    """
    Create experiment-specific logger.

    Args:
        experiment_name: Name of the experiment
        log_dir: Directory to store logs

    Returns:
        Experiment logger
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'{experiment_name}_{timestamp}.log')

    return setup_logger(f'exp_{experiment_name}', log_file)
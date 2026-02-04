"""
Plotting utilities for fluProfiler.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path


def set_plotting_style(style='default'):
    """
    Set matplotlib and seaborn plotting style.

    Args:
        style: Plotting style ('default', 'seaborn', 'ggplot')
    """
    if style == 'seaborn':
        sns.set_style('whitegrid')
        sns.set_palette('husl')
    elif style == 'ggplot':
        plt.style.use('ggplot')
    else:
        plt.style.use('default')

    # Set default figure size
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12


def plot_scatter_predictions(y_true, y_pred, title='Predictions vs Actual',
                           xlabel='Actual Values', ylabel='Predicted Values',
                           save_path=None):
    """
    Create scatter plot of predictions vs actual values.

    Args:
        y_true: Actual values
        y_pred: Predicted values
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Path to save plot (optional)
    """
    plt.figure(figsize=(8, 6))

    # Calculate correlation
    corr = np.corrcoef(y_true, y_pred)[0, 1]

    plt.scatter(y_true, y_pred, alpha=0.6, s=50)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)],
             'r--', linewidth=2, label='Perfect prediction')

    plt.title('.3f')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_training_history(history, save_path=None):
    """
    Plot training and validation loss curves.

    Args:
        history: Dictionary with 'train_loss' and 'val_loss' keys
        save_path: Path to save plot (optional)
    """
    plt.figure(figsize=(10, 6))

    epochs = range(1, len(history['train_loss']) + 1)

    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)

    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_seasonal_performance(seasonal_data, metric='MAE', save_path=None):
    """
    Plot performance metrics by season.

    Args:
        seasonal_data: DataFrame with seasonal performance data
        metric: Metric to plot
        save_path: Path to save plot (optional)
    """
    plt.figure(figsize=(12, 6))

    seasons = seasonal_data.index
    values = seasonal_data[metric]

    plt.bar(seasons, values, alpha=0.7, color='skyblue', edgecolor='navy')

    plt.title(f'{metric} by Season')
    plt.xlabel('Season')
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, v in enumerate(values):
        plt.text(i, v + max(values) * 0.01, '.3f',
                ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
"""
Performance metrics for fluProfiler evaluation.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr


def calculate_metrics(observation, prediction, print_result=True):
    """
    Calculate comprehensive performance metrics.

    Args:
        observation: Ground truth values
        prediction: Predicted values
        print_result: Whether to print results

    Returns:
        Dictionary of metrics (MAE, MSE, Pearson, Spearman, R2)
    """
    O = np.array(observation)
    P = np.array(prediction)

    MAE, MSE, Pearson, Spearman, R2 = (
        mean_absolute_error(O, P),
        mean_squared_error(O, P),
        pearsonr(O, P),
        spearmanr(O, P),
        r2_score(O, P)
    )

    if print_result:
        print(".5f")
        print(".5f")
        print(".5f")
        print(".5f")
        print(".5f")

    return {
        'MAE': MAE,
        'MSE': MSE,
        'Pearson_r': Pearson[0],
        'Pearson_p': Pearson[1],
        'Spearman_r': Spearman[0],
        'Spearman_p': Spearman[1],
        'R2': R2
    }


def print_exams(Observation, Prediction, print_result=True):
    """
    Legacy function for backwards compatibility.
    """
    return calculate_metrics(Observation, Prediction, print_result)
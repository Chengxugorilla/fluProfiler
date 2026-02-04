"""
Statistical analysis utilities for fluProfiler evaluation.

Includes bootstrap analysis, confidence intervals, and group statistics.
"""

import numpy as np
from scipy.stats import bootstrap
import pandas as pd


def bootstrap_confidence_interval(data, statistic_func, n_resamples=1000, confidence_level=0.95):
    """
    Calculate bootstrap confidence intervals.

    Args:
        data: Input data array
        statistic_func: Function to compute statistic
        n_resamples: Number of bootstrap resamples
        confidence_level: Confidence level (0-1)

    Returns:
        Confidence interval bounds
    """
    def statistic(x):
        return statistic_func(x)

    res = bootstrap((data,), statistic, n_resamples=n_resamples,
                   confidence_level=confidence_level, method='percentile')

    return res.confidence_interval


def grouped_statistics(dataframe, group_column, value_column, statistic_funcs=None):
    """
    Calculate statistics grouped by a column.

    Args:
        dataframe: Input DataFrame
        group_column: Column to group by
        value_column: Column with values to analyze
        statistic_funcs: List of statistic functions to apply

    Returns:
        DataFrame with grouped statistics
    """
    if statistic_funcs is None:
        statistic_funcs = [np.mean, np.std, np.median, len]

    results = []
    for name, group in dataframe.groupby(group_column):
        stats = {'group': name}
        for func in statistic_funcs:
            if func == len:
                stats['count'] = func(group[value_column])
            else:
                stats[func.__name__] = func(group[value_column])
        results.append(stats)

    return pd.DataFrame(results)


def seasonal_analysis(dataframe, season_column, value_column):
    """
    Analyze performance by flu season.

    Args:
        dataframe: DataFrame with season and value columns
        season_column: Column containing season information
        value_column: Column with values to analyze

    Returns:
        Seasonal analysis summary
    """
    seasonal_stats = dataframe.groupby(season_column)[value_column].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).round(4)

    return seasonal_stats


def reverse_test_evaluation(predictions, actuals, test_season):
    """
    Evaluate reverse testing performance.

    Args:
        predictions: Predicted values
        actuals: Actual values
        test_season: Season being tested

    Returns:
        Evaluation metrics for reverse testing
    """
    from .metrics import calculate_metrics

    metrics = calculate_metrics(actuals, predictions, print_result=False)
    metrics['test_season'] = test_season

    return metrics


def extrapolation_evaluation(predictions, actuals, train_seasons, test_season):
    """
    Evaluate extrapolation performance.

    Args:
        predictions: Predicted values
        actuals: Actual values
        train_seasons: Seasons used for training
        test_season: Season being extrapolated to

    Returns:
        Evaluation metrics for extrapolation
    """
    from .metrics import calculate_metrics

    metrics = calculate_metrics(actuals, predictions, print_result=False)
    metrics['train_seasons'] = train_seasons
    metrics['test_season'] = test_season

    return metrics
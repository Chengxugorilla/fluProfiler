"""
Scoring functions for active learning sample selection.
"""

import numpy as np
from typing import Dict, List
from scipy.stats import entropy


def uncertainty_score(predictions: np.ndarray, method: str = 'least_confident') -> np.ndarray:
    """
    Calculate uncertainty scores for predictions.

    Args:
        predictions: Model prediction probabilities
        method: Uncertainty calculation method

    Returns:
        Uncertainty scores
    """
    if method == 'least_confident':
        return 1 - np.max(predictions, axis=1)
    elif method == 'entropy':
        return entropy(predictions.T)
    elif method == 'margin':
        sorted_preds = np.sort(predictions, axis=1)
        return 1 - (sorted_preds[:, -1] - sorted_preds[:, -2])
    else:
        raise ValueError(f"Unknown uncertainty method: {method}")


def diversity_score(embeddings: np.ndarray, selected_indices: List[int]) -> np.ndarray:
    """
    Calculate diversity scores based on embeddings.

    Args:
        embeddings: Data embeddings
        selected_indices: Indices of already selected samples

    Returns:
        Diversity scores for each sample
    """
    if len(selected_indices) == 0:
        return np.ones(len(embeddings))

    selected_embeddings = embeddings[selected_indices]
    similarities = np.dot(embeddings, selected_embeddings.T)
    max_similarities = np.max(similarities, axis=1)

    return 1 - max_similarities


def representativeness_score(embeddings: np.ndarray,
                            cluster_centers: np.ndarray) -> np.ndarray:
    """
    Calculate representativeness scores based on cluster centers.

    Args:
        embeddings: Data embeddings
        cluster_centers: Cluster center embeddings

    Returns:
        Representativeness scores
    """
    similarities = np.dot(embeddings, cluster_centers.T)
    max_similarities = np.max(similarities, axis=1)

    return max_similarities


def combined_score(uncertainty_scores: np.ndarray,
                  diversity_scores: np.ndarray,
                  representativeness_scores: np.ndarray,
                  weights: Dict[str, float] = None) -> np.ndarray:
    """
    Combine multiple scoring criteria.

    Args:
        uncertainty_scores: Uncertainty scores
        diversity_scores: Diversity scores
        representativeness_scores: Representativeness scores
        weights: Weights for each criterion

    Returns:
        Combined scores
    """
    if weights is None:
        weights = {'uncertainty': 0.5, 'diversity': 0.3, 'representativeness': 0.2}

    combined = (weights['uncertainty'] * uncertainty_scores +
               weights['diversity'] * diversity_scores +
               weights['representativeness'] * representativeness_scores)

    return combined
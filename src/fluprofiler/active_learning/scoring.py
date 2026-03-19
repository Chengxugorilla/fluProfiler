"""
Scoring functions for active learning sample selection.

Enhanced from Section4_ActiveLearning.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.stats import entropy
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA


def uncertainty_score(
    predictions: np.ndarray,
    method: str = 'least_confident'
) -> np.ndarray:
    """
    Calculate uncertainty scores for predictions.
    
    Args:
        predictions: Model prediction probabilities (n_samples, n_classes)
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


def diversity_score(
    embeddings: np.ndarray,
    selected_indices: List[int]
) -> np.ndarray:
    """
    Calculate diversity scores based on embeddings.
    
    Higher score = more diverse from already selected samples.
    
    Args:
        embeddings: Data embeddings (n_samples, embedding_dim)
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


def representativeness_score(
    embeddings: np.ndarray,
    cluster_centers: Optional[np.ndarray] = None,
    method: str = 'distance'
) -> np.ndarray:
    """
    Calculate representativeness scores based on cluster centers.
    
    Args:
        embeddings: Data embeddings
        cluster_centers: Cluster center embeddings
        method: Scoring method ('distance' or 'density')
        
    Returns:
        Representativeness scores
    """
    if cluster_centers is None:
        # Compute cluster centers using PCA
        n_components = min(10, embeddings.shape[1], embeddings.shape[0] - 1)
        pca = PCA(n_components=n_components)
        pca.fit(embeddings)
        cluster_centers = pca.components_.T
    
    distances = cdist(embeddings, cluster_centers)
    min_distances = np.min(distances, axis=1)
    
    if method == 'distance':
        # Lower distance = higher representativeness
        return -min_distances
    else:
        # Use density estimation
        return 1 / (min_distances + 1e-6)


def combined_score(
    uncertainty_scores: np.ndarray,
    diversity_scores: np.ndarray,
    representativeness_scores: np.ndarray,
    weights: Optional[Dict[str, float]] = None
) -> np.ndarray:
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
    
    # Normalize scores to [0, 1]
    uncertainty_scores = _normalize(uncertainty_scores)
    diversity_scores = _normalize(diversity_scores)
    representativeness_scores = _normalize(representativeness_scores)
    
    combined = (
        weights.get('uncertainty', 0.5) * uncertainty_scores +
        weights.get('diversity', 0.3) * diversity_scores +
        weights.get('representativeness', 0.2) * representativeness_scores
    )
    
    return combined


def antigenic_distance_score(
    embeddings: np.ndarray,
    antigenic_centers: Optional[Dict[str, np.ndarray]] = None
) -> np.ndarray:
    """
    Calculate antigenic distance score for vaccine strain selection.
    
    Args:
        embeddings: Strain embeddings
        antigenic_centers: Pre-defined antigenic center embeddings
                          for different virus types (H1N1, H3N2, etc.)
        
    Returns:
        Antigenic distance scores
    """
    if antigenic_centers is None:
        # Default: return random scores
        return np.random.random(len(embeddings))
    
    scores = np.zeros(len(embeddings))
    for virus_type, center in antigenic_centers.items():
        distances = np.linalg.norm(embeddings - center, axis=1)
        scores += distances
    
    return scores / len(antigenic_centers)


def evolutionary_distance_score(
    embeddings: np.ndarray,
    reference_strains: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Calculate evolutionary distance score.
    
    Measures distance from reference strains (e.g., vaccine strains).
    
    Args:
        embeddings: Strain embeddings
        reference_strains: Reference strain embeddings
        
    Returns:
        Evolutionary distance scores
    """
    if reference_strains is None or len(reference_strains) == 0:
        return np.zeros(len(embeddings))
    
    distances = cdist(embeddings, reference_strains)
    min_distances = np.min(distances, axis=1)
    
    return min_distances


def batch_bald_score(
    predictions: np.ndarray,
    n_classes: int = 2,
    n_mc_samples: int = 10
) -> np.ndarray:
    """
    Batch BALD (Bayesian Active Learning by Disagreement) score.
    
    Args:
        predictions: Monte Carlo dropout predictions (n_samples, n_mc_samples, n_classes)
        n_classes: Number of classes
        n_mc_samples: Number of Monte Carlo samples
        
    Returns:
        Batch BALD scores
    """
    if predictions.ndim == 2:
        # Single predictions, not MC samples
        return uncertainty_score(predictions, method='entropy')
    
    # Calculate mutual information
    mean_probs = np.mean(predictions, axis=1)
    entropy_mean = entropy(mean_probs.T)
    
    mean_entropy = np.mean([
        entropy(predictions[:, i, :].T)
        for i in range(predictions.shape[1])
    ], axis=0)
    
    bald_scores = entropy_mean - mean_entropy
    return bald_scores


def cnic_score(
    embeddings: np.ndarray,
    cnic_representatives: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    CNIC (China National Influenza Center) specific scoring.
    
    Prioritizes strains that are well-represented in CNIC data.
    
    Args:
        embeddings: Strain embeddings
        cnic_representatives: CNIC representative strain embeddings
        
    Returns:
        CNIC-specific scores
    """
    if cnic_representatives is None:
        return np.random.random(len(embeddings))
    
    # Calculate similarity to CNIC representatives
    similarities = np.dot(embeddings, cnic_representatives.T)
    max_similarities = np.max(similarities, axis=1)
    
    return max_similarities


def cdc_score(
    embeddings: np.ndarray,
    cdc_representatives: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    CDC (US Centers for Disease Control) specific scoring.
    
    Prioritizes strains that are well-represented in CDC data.
    
    Args:
        embeddings: Strain embeddings
        cdc_representatives: CDC representative strain embeddings
        
    Returns:
        CDC-specific scores
    """
    if cdc_representatives is None:
        return np.random.random(len(embeddings))
    
    # Calculate similarity to CDC representatives
    similarities = np.dot(embeddings, cdc_representatives.T)
    max_similarities = np.max(similarities, axis=1)
    
    return max_similarities


def coverage_score(
    embeddings: np.ndarray,
    existing_indices: List[int],
    method: str = 'cluster_coverage'
) -> np.ndarray:
    """
    Calculate coverage score for selected samples.
    
    Measures how well the selected samples cover the data distribution.
    
    Args:
        embeddings: Data embeddings
        existing_indices: Already selected indices
        method: Coverage method ('cluster_coverage' or 'density_coverage')
        
    Returns:
        Coverage scores
    """
    n_samples = len(embeddings)
    
    if len(existing_indices) == 0:
        return np.ones(n_samples)
    
    # Calculate distances from each point to nearest selected point
    selected_embeddings = embeddings[existing_indices]
    distances = cdist(embeddings, selected_embeddings)
    min_distances = np.min(distances, axis=1)
    
    if method == 'cluster_coverage':
        # Higher score = more uncovered (need to select)
        return min_distances
    else:
        # Inverse: higher score = already covered
        return 1 / (min_distances + 1e-6)


def _normalize(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize scores to [0, 1]."""
    min_val = np.min(scores)
    max_val = np.max(scores)
    if max_val - min_val == 0:
        return np.zeros_like(scores)
    return (scores - min_val) / (max_val - min_val)


def compute_all_scores(
    embeddings: np.ndarray,
    predictions: Optional[np.ndarray] = None,
    selected_indices: Optional[List[int]] = None,
    cluster_centers: Optional[np.ndarray] = None,
    weights: Optional[Dict[str, float]] = None
) -> Dict[str, np.ndarray]:
    """
    Compute all available scoring functions.
    
    Args:
        embeddings: Data embeddings
        predictions: Model predictions
        selected_indices: Already selected indices
        cluster_centers: Cluster centers
        weights: Weights for combined score
        
    Returns:
        Dictionary of all computed scores
    """
    scores = {}
    
    # Uncertainty scores
    if predictions is not None:
        scores['uncertainty_least_confident'] = uncertainty_score(predictions, 'least_confident')
        scores['uncertainty_entropy'] = uncertainty_score(predictions, 'entropy')
        scores['uncertainty_margin'] = uncertainty_score(predictions, 'margin')
    
    # Diversity scores
    if selected_indices is not None:
        scores['diversity'] = diversity_score(embeddings, selected_indices)
    
    # Representativeness scores
    scores['representativeness'] = representativeness_score(embeddings, cluster_centers)
    
    # Coverage scores
    if selected_indices is not None:
        scores['coverage'] = coverage_score(embeddings, selected_indices)
    
    # Combined score
    if predictions is not None and selected_indices is not None:
        scores['combined'] = combined_score(
            scores.get('uncertainty_least_confident', np.zeros(len(embeddings))),
            scores.get('diversity', np.zeros(len(embeddings))),
            scores['representativeness'],
            weights
        )
    
    return scores

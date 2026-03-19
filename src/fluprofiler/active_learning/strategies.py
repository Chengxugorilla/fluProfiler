"""
Active learning sampling strategies.

Enhanced from Section4_ActiveLearning with hybrid strategies.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from abc import ABC, abstractmethod
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist


class SamplingStrategy(ABC):
    """Abstract base class for sampling strategies."""
    
    @abstractmethod
    def select_samples(
        self,
        embeddings: np.ndarray,
        num_samples: int,
        model_predictions: Optional[Dict] = None,
        **kwargs
    ) -> List[int]:
        """
        Select samples for labeling.
        
        Args:
            embeddings: Data embeddings
            num_samples: Number of samples to select
            model_predictions: Model predictions (optional)
            **kwargs: Additional strategy-specific parameters
            
        Returns:
            Indices of selected samples
        """
        pass


class RandomSampling(SamplingStrategy):
    """Random sampling strategy."""
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize random sampling.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
    
    def select_samples(
        self,
        embeddings: np.ndarray,
        num_samples: int,
        model_predictions: Optional[Dict] = None,
        **kwargs
    ) -> List[int]:
        """Randomly select samples."""
        n_total = len(embeddings)
        if num_samples >= n_total:
            return list(range(n_total))
        return np.random.choice(n_total, num_samples, replace=False).tolist()


class UncertaintySampling(SamplingStrategy):
    """
    Uncertainty-based sampling strategy.
    
    Supports multiple uncertainty measures:
    - least_confident: 1 - max probability
    - entropy: entropy of prediction distribution
    - margin: difference between top two probabilities
    """
    
    def __init__(
        self,
        uncertainty_measure: str = 'least_confident',
        temperature: float = 1.0
    ):
        """
        Initialize uncertainty sampling.
        
        Args:
            uncertainty_measure: Type of uncertainty measure
            temperature: Temperature for softmax scaling
        """
        self.uncertainty_measure = uncertainty_measure
        self.temperature = temperature
    
    def select_samples(
        self,
        embeddings: np.ndarray,
        num_samples: int,
        model_predictions: Optional[Dict] = None,
        **kwargs
    ) -> List[int]:
        """Select samples based on uncertainty."""
        if model_predictions is None:
            raise ValueError("Model predictions required for uncertainty sampling")
        
        uncertainties = self._calculate_uncertainty(model_predictions)
        # Select samples with highest uncertainty
        indices = np.argsort(uncertainties)[-num_samples:]
        return indices.tolist()
    
    def _calculate_uncertainty(self, predictions: Dict) -> np.ndarray:
        """Calculate uncertainty scores."""
        logits = predictions.get('logits')
        if logits is not None:
            # Apply temperature scaling
            logits = logits / self.temperature
            probs = self._softmax(logits)
        else:
            probs = predictions.get('probabilities', np.random.random((len(predictions), 2)))
        
        if self.uncertainty_measure == 'least_confident':
            return 1 - np.max(probs, axis=1)
        elif self.uncertainty_measure == 'entropy':
            return -np.sum(probs * np.log(probs + 1e-10), axis=1)
        elif self.uncertainty_measure == 'margin':
            sorted_preds = np.sort(probs, axis=1)
            return 1 - (sorted_preds[:, -1] - sorted_preds[:, -2])
        else:
            raise ValueError(f"Unknown uncertainty measure: {self.uncertainty_measure}")
    
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Compute softmax values."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)


class DiversitySampling(SamplingStrategy):
    """
    Diversity-based sampling strategy using clustering.
    
    Selects samples that are diverse within each cluster.
    """
    
    def __init__(
        self,
        n_clusters: int = 10,
        method: str = 'kmeans',
        use_gpu: bool = False
    ):
        """
        Initialize diversity sampling.
        
        Args:
            n_clusters: Number of clusters
            method: Clustering method ('kmeans' or 'mini_batch_kmeans')
            use_gpu: Whether to use GPU acceleration
        """
        self.n_clusters = n_clusters
        self.method = method
        self.use_gpu = use_gpu
        self.clusterer = None
        
    def select_samples(
        self,
        embeddings: np.ndarray,
        num_samples: int,
        model_predictions: Optional[Dict] = None,
        **kwargs
    ) -> List[int]:
        """Select diverse samples using clustering."""
        # Fit clustering
        if self.method == 'kmeans':
            self.clusterer = KMeans(
                n_clusters=self.n_clusters,
                random_state=42,
                n_init=10
            )
        else:
            self.clusterer = MiniBatchKMeans(
                n_clusters=self.n_clusters,
                random_state=42,
                n_init=10
            )
        
        cluster_labels = self.clusterer.fit_predict(embeddings)
        
        # Select most representative from each cluster
        selected_indices = []
        for cluster_id in range(self.n_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            if len(cluster_indices) == 0:
                continue
                
            # Get samples closest to cluster center
            cluster_embeddings = embeddings[cluster_indices]
            center = self.clusterer.cluster_centers_[cluster_id]
            distances = np.linalg.norm(cluster_embeddings - center, axis=1)
            
            # Select closest to center for diversity
            n_select = max(1, num_samples // self.n_clusters)
            closest_indices = cluster_indices[np.argsort(distances)[:n_select]]
            selected_indices.extend(closest_indices)
            
        # Fill remaining slots with random if needed
        if len(selected_indices) < num_samples:
            remaining = set(range(len(embeddings))) - set(selected_indices)
            additional = np.random.choice(
                list(remaining),
                num_samples - len(selected_indices),
                replace=False
            )
            selected_indices.extend(additional)
            
        return selected_indices[:num_samples]


class RepresentativeSampling(SamplingStrategy):
    """
    Representative sampling based on data distribution.
    
    Selects samples closest to cluster centers.
    """
    
    def __init__(
        self,
        n_representatives: int = 10,
        method: str = 'kmeans'
    ):
        """
        Initialize representative sampling.
        
        Args:
            n_representatives: Number of representative clusters
            method: Clustering method
        """
        self.n_representatives = n_representatives
        self.method = method
        
    def select_samples(
        self,
        embeddings: np.ndarray,
        num_samples: int,
        model_predictions: Optional[Dict] = None,
        cluster_centers: Optional[np.ndarray] = None,
        **kwargs
    ) -> List[int]:
        """Select representative samples."""
        if cluster_centers is None:
            # Compute cluster centers if not provided
            if self.method == 'kmeans':
                clusterer = KMeans(
                    n_clusters=self.n_representatives,
                    random_state=42,
                    n_init=10
                )
            else:
                clusterer = MiniBatchKMeans(
                    n_clusters=self.n_representatives,
                    random_state=42,
                    n_init=10
                )
            clusterer.fit(embeddings)
            cluster_centers = clusterer.cluster_centers_
        
        # Select samples closest to each center
        distances = cdist(embeddings, cluster_centers)
        min_distances = np.min(distances, axis=1)
        
        # Select samples with minimum distance to any center
        indices = np.argsort(min_distances)[:num_samples]
        return indices.tolist()


class HybridStrategy(SamplingStrategy):
    """
    Hybrid strategy combining uncertainty, diversity, and representativeness.
    
    This is the main strategy used in Section4_ActiveLearning.
    """
    
    def __init__(
        self,
        uncertainty_weight: float = 0.5,
        diversity_weight: float = 0.3,
        representativeness_weight: float = 0.2,
        n_clusters: int = 10,
        uncertainty_measure: str = 'least_confident'
    ):
        """
        Initialize hybrid strategy.
        
        Args:
            uncertainty_weight: Weight for uncertainty score
            diversity_weight: Weight for diversity score
            representativeness_weight: Weight for representativeness score
            n_clusters: Number of clusters for diversity
            uncertainty_measure: Uncertainty measure type
        """
        self.uncertainty_weight = uncertainty_weight
        self.diversity_weight = diversity_weight
        self.representativeness_weight = representativeness_weight
        self.n_clusters = n_clusters
        self.uncertainty_measure = uncertainty_measure
        
        # Initialize sub-strategies
        self.uncertainty_sampler = UncertaintySampling(uncertainty_measure)
        self.diversity_sampler = DiversitySampling(n_clusters)
        self.representative_sampler = RepresentativeSampling(n_clusters)
        
    def select_samples(
        self,
        embeddings: np.ndarray,
        num_samples: int,
        model_predictions: Optional[Dict] = None,
        selected_indices: Optional[List[int]] = None,
        **kwargs
    ) -> List[int]:
        """
        Select samples using hybrid scoring.
        
        Args:
            embeddings: Data embeddings
            num_samples: Number of samples to select
            model_predictions: Model predictions for uncertainty
            selected_indices: Already selected indices for diversity
            **kwargs: Additional parameters
            
        Returns:
            Selected sample indices
        """
        if model_predictions is None:
            # Fallback to random if no predictions
            return np.random.choice(
                len(embeddings), num_samples, replace=False
            ).tolist()
        
        # Calculate individual scores
        uncertainty_scores = self._calculate_uncertainty_scores(model_predictions)
        diversity_scores = self._calculate_diversity_scores(
            embeddings, selected_indices
        )
        representativeness_scores = self._calculate_representativeness_scores(
            embeddings
        )
        
        # Normalize scores
        uncertainty_scores = self._normalize(uncertainty_scores)
        diversity_scores = self._normalize(diversity_scores)
        representativeness_scores = self._normalize(representativeness_scores)
        
        # Combine scores
        combined_scores = (
            self.uncertainty_weight * uncertainty_scores +
            self.diversity_weight * diversity_scores +
            self.representativeness_weight * representativeness_scores
        )
        
        # Select top samples
        indices = np.argsort(combined_scores)[-num_samples:]
        return indices.tolist()
    
    def _calculate_uncertainty_scores(self, predictions: Dict) -> np.ndarray:
        """Calculate uncertainty scores."""
        logits = predictions.get('logits')
        if logits is not None:
            probs = self.uncertainty_sampler._softmax(logits)
        else:
            probs = predictions.get('probabilities', np.random.random((len(predictions), 2)))
            
        if self.uncertainty_measure == 'least_confident':
            return 1 - np.max(probs, axis=1)
        elif self.uncertainty_measure == 'entropy':
            return -np.sum(probs * np.log(probs + 1e-10), axis=1)
        else:
            return 1 - np.max(probs, axis=1)
    
    def _calculate_diversity_scores(
        self,
        embeddings: np.ndarray,
        selected_indices: Optional[List[int]] = None
    ) -> np.ndarray:
        """Calculate diversity scores."""
        if selected_indices is None or len(selected_indices) == 0:
            return np.ones(len(embeddings))
        
        selected_embeddings = embeddings[selected_indices]
        similarities = np.dot(embeddings, selected_embeddings.T)
        max_similarities = np.max(similarities, axis=1)
        
        return 1 - max_similarities
    
    def _calculate_representativeness_scores(
        self,
        embeddings: np.ndarray
    ) -> np.ndarray:
        """Calculate representativeness scores."""
        # Use PCA to find representative samples
        n_components = min(self.n_clusters, embeddings.shape[1], embeddings.shape[0] - 1)
        pca = PCA(n_components=n_components)
        pca.fit(embeddings)
        
        # Calculate reconstruction error as representativeness
        projected = pca.transform(embeddings)
        reconstructed = pca.inverse_transform(projected)
        reconstruction_error = np.linalg.norm(embeddings - reconstructed, axis=1)
        
        # Lower error = more representative
        return -reconstruction_error
    
    @staticmethod
    def _normalize(scores: np.ndarray) -> np.ndarray:
        """Min-max normalize scores to [0, 1]."""
        min_val = np.min(scores)
        max_val = np.max(scores)
        if max_val - min_val == 0:
            return np.zeros_like(scores)
        return (scores - min_val) / (max_val - min_val)


class AdaptiveStrategy(SamplingStrategy):
    """
    Adaptive strategy that adjusts weights based on query round.
    
    Starts with more exploration (diversity) and shifts to exploitation
    (uncertainty) as the model improves.
    """
    
    def __init__(
        self,
        initial_exploration_weight: float = 0.7,
        final_exploration_weight: float = 0.2,
        n_rounds: int = 20
    ):
        """
        Initialize adaptive strategy.
        
        Args:
            initial_exploration_weight: Initial diversity weight
            final_exploration_weight: Final diversity weight
            n_rounds: Total number of active learning rounds
        """
        self.initial_exploration_weight = initial_exploration_weight
        self.final_exploration_weight = final_exploration_weight
        self.n_rounds = n_rounds
        self.current_round = 0
        
        self.hybrid_strategy = HybridStrategy()
        
    def select_samples(
        self,
        embeddings: np.ndarray,
        num_samples: int,
        model_predictions: Optional[Dict] = None,
        **kwargs
    ) -> List[int]:
        """Select samples with adaptive exploration-exploitation trade-off."""
        # Calculate current exploration weight
        progress = self.current_round / max(1, self.n_rounds)
        exploration_weight = (
            self.initial_exploration_weight - 
            (self.initial_exploration_weight - self.final_exploration_weight) * progress
        )
        
        # Update hybrid strategy weights
        self.hybrid_strategy.uncertainty_weight = 1 - exploration_weight
        self.hybrid_strategy.diversity_weight = exploration_weight * 0.7
        self.hybrid_strategy.representativeness_weight = exploration_weight * 0.3
        
        self.current_round += 1
        
        return self.hybrid_strategy.select_samples(
            embeddings, num_samples, model_predictions, **kwargs
        )

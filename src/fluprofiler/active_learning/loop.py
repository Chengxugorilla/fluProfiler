"""
Active learning loop implementation.

Enhanced from Section4_ActiveLearning.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Callable
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .strategies import (
    SamplingStrategy,
    UncertaintySampling,
    HybridStrategy,
    DiversitySampling,
    RepresentativeSampling
)
from .scoring import uncertainty_score, combined_score
from .dataset import FluProfilerDataset, ActiveLearningPool


class ActiveLearningLoop:
    """
    Main active learning loop implementation.
    
    Manages the iterative process of:
    1. Training model on labeled data
    2. Selecting informative samples from unlabeled pool
    3. Querying labels for selected samples
    4. Updating labeled/unlabeled pools
    """
    
    def __init__(
        self,
        dataset: FluProfilerDataset,
        strategy: Optional[SamplingStrategy] = None,
        initial_labeled_ratio: float = 0.1,
        batch_size: int = 8,
        max_rounds: int = 20,
        seed: Optional[int] = None
    ):
        """
        Initialize active learning loop.
        
        Args:
            dataset: The full dataset
            strategy: Sampling strategy to use
            initial_labeled_ratio: Ratio of initially labeled samples
            batch_size: Number of samples to query per round
            max_rounds: Maximum number of active learning rounds
            seed: Random seed
        """
        self.dataset = dataset
        self.pool = ActiveLearningPool(dataset, initial_labeled_ratio)
        self.strategy = strategy or HybridStrategy()
        self.batch_size = batch_size
        self.max_rounds = max_rounds
        self.current_round = 0
        
        if seed is not None:
            np.random.seed(seed)
        
        # History tracking
        self.history = []
        self.selected_samples_per_round = []
        
    def select_batch(
        self,
        embeddings: np.ndarray,
        model: Optional[torch.nn.Module] = None,
        predictions: Optional[np.ndarray] = None
    ) -> List[int]:
        """
        Select next batch of samples for labeling.
        
        Args:
            embeddings: Feature embeddings for all data
            model: Current model (used for prediction)
            predictions: Pre-computed predictions
            
        Returns:
            Indices of selected samples
        """
        unlabeled_indices = self.pool.get_unlabeled_data()
        
        if len(unlabeled_indices) <= self.batch_size:
            return unlabeled_indices
        
        # Get embeddings for unlabeled data
        unlabeled_embeddings = embeddings[unlabeled_indices]
        
        # Prepare model predictions
        model_predictions = None
        if predictions is not None:
            model_predictions = {'probabilities': predictions[unlabeled_indices]}
        elif model is not None:
            # This would require running inference
            # Placeholder: use random for now
            model_predictions = {'probabilities': np.random.random((len(unlabeled_indices), 2))}
        
        # Get already selected indices for diversity
        all_selected = [idx for round_indices in self.selected_samples_per_round 
                       for idx in round_indices]
        
        # Select using strategy
        selected_local_indices = self.strategy.select_samples(
            unlabeled_embeddings,
            self.batch_size,
            model_predictions,
            selected_indices=all_selected
        )
        
        # Convert to global indices
        selected_global = [unlabeled_indices[i] for i in selected_local_indices]
        
        return selected_global
    
    def update_labels(self, new_indices: List[int]):
        """
        Update labeled and unlabeled sets after getting new labels.
        
        Args:
            new_indices: Indices of newly labeled samples
        """
        self.pool.label_samples(new_indices)
        self.selected_samples_per_round.append(new_indices)
        self.current_round += 1
        
        # Record history
        self.history.append({
            'round': self.current_round,
            'total_labeled': len(self.pool.labeled_indices),
            'total_unlabeled': len(self.pool.unlabeled_indices),
            'selected_samples': new_indices
        })
        
    def get_labeled_data(
        self,
        dataloader: DataLoader,
        emb_dict: Dict[str, torch.Tensor]
    ) -> Tuple[List, List]:
        """
        Get labeled data for training.
        
        Args:
            dataloader: DataLoader for the dataset
            emb_dict: Embedding dictionary
            
        Returns:
            (features, labels) tuples
        """
        labeled_indices = self.pool.get_labeled_data()
        features = []
        labels = []
        
        for idx in labeled_indices:
            # Extract features from embeddings
            # This is dataset-specific implementation
            pass
        
        return features, labels
    
    def get_statistics(self) -> Dict:
        """
        Get current active learning statistics.
        
        Returns:
            Dictionary with statistics
        """
        pool_stats = self.pool.get_statistics()
        return {
            **pool_stats,
            'current_round': self.current_round,
            'max_rounds': self.max_rounds,
            'batch_size': self.batch_size,
            'strategy': type(self.strategy).__name__
        }
    
    def get_history(self) -> pd.DataFrame:
        """
        Get active learning history as DataFrame.
        
        Returns:
            DataFrame with iteration history
        """
        return pd.DataFrame(self.history)
    
    def should_stop(self) -> bool:
        """Check if active learning should stop."""
        if self.current_round >= self.max_rounds:
            return True
        if len(self.pool.unlabeled_indices) < self.batch_size:
            return True
        return False


class StreamingActiveLearningLoop(ActiveLearningLoop):
    """
    Active learning loop for streaming data.
    
    Handles scenarios where new data arrives continuously.
    """
    
    def __init__(
        self,
        dataset: FluProfilerDataset,
        strategy: Optional[SamplingStrategy] = None,
        initial_labeled_ratio: float = 0.1,
        batch_size: int = 8,
        max_rounds: int = 20,
        stream_buffer_size: int = 1000
    ):
        """
        Initialize streaming active learning loop.
        
        Args:
            dataset: Initial dataset
            strategy: Sampling strategy
            initial_labeled_ratio: Initial labeled ratio
            batch_size: Batch size for querying
            max_rounds: Maximum rounds
            stream_buffer_size: Size of streaming buffer
        """
        super().__init__(
            dataset, strategy, initial_labeled_ratio, 
            batch_size, max_rounds
        )
        self.stream_buffer_size = stream_buffer_size
        self.stream_buffer = []
        
    def add_stream_data(self, new_data):
        """
        Add new streaming data to buffer.
        
        Args:
            new_data: New data samples
        """
        self.stream_buffer.extend(new_data)
        
        if len(self.stream_buffer) >= self.stream_buffer_size:
            self._process_stream_buffer()
            
    def _process_stream_buffer(self):
        """Process buffered streaming data."""
        # Add buffer to unlabeled pool
        # This is a simplified implementation
        pass


class BatchActiveLearningLoop(ActiveLearningLoop):
    """
    Optimized active learning loop for batch processing.
    
    Uses vectorized operations for faster sample selection.
    """
    
    def __init__(
        self,
        dataset: FluProfilerDataset,
        strategy: Optional[SamplingStrategy] = None,
        initial_labeled_ratio: float = 0.1,
        batch_size: int = 8,
        max_rounds: int = 20,
        use_gpu: bool = True
    ):
        """
        Initialize batch active learning loop.
        
        Args:
            dataset: The full dataset
            strategy: Sampling strategy
            initial_labeled_ratio: Initial labeled ratio
            batch_size: Batch size
            max_rounds: Maximum rounds
            use_gpu: Whether to use GPU acceleration
        """
        super().__init__(
            dataset, strategy, initial_labeled_ratio,
            batch_size, max_rounds
        )
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        
    def select_batch_vectorized(
        self,
        embeddings: np.ndarray,
        predictions: Optional[np.ndarray] = None
    ) -> List[int]:
        """
        Vectorized batch selection for faster processing.
        
        Args:
            embeddings: Feature embeddings
            predictions: Model predictions
            
        Returns:
            Selected indices
        """
        unlabeled_indices = self.pool.get_unlabeled_data()
        
        if len(unlabeled_indices) <= self.batch_size:
            return unlabeled_indices
        
        # Use vectorized operations
        unlabeled_embeddings = embeddings[unlabeled_indices]
        
        # Pre-compute all scores at once
        scores = self._compute_all_scores_vectorized(
            unlabeled_embeddings,
            predictions,
            unlabeled_indices
        )
        
        # Select top-k
        selected_local = np.argsort(scores)[-self.batch_size:]
        selected_global = [unlabeled_indices[i] for i in selected_local]
        
        return selected_global
    
    def _compute_all_scores_vectorized(
        self,
        embeddings: np.ndarray,
        predictions: Optional[np.ndarray],
        indices: List[int]
    ) -> np.ndarray:
        """
        Compute all scores using vectorized operations.
        
        Args:
            embeddings: Data embeddings
            predictions: Model predictions
            indices: Data indices
            
        Returns:
            Combined scores
        """
        scores = np.zeros(len(embeddings))
        
        # Uncertainty component
        if predictions is not None:
            uncertainty = uncertainty_score(predictions, 'entropy')
            scores += 0.5 * uncertainty
        
        # Diversity component
        all_selected = [idx for round_indices in self.selected_samples_per_round 
                       for idx in round_indices]
        if len(all_selected) > 0:
            selected_embeddings = embeddings[[indices.index(i) for i in all_selected if i in indices]]
            if len(selected_embeddings) > 0:
                similarities = np.dot(embeddings, selected_embeddings.T)
                max_sim = np.max(similarities, axis=1)
                scores += 0.3 * (1 - max_sim)
        
        # Representativeness component
        scores += 0.2 * np.random.random(len(embeddings))  # Placeholder
        
        return scores

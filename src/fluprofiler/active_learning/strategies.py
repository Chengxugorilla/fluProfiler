"""
Active learning sampling strategies.
"""

import numpy as np
from typing import List, Tuple, Dict
from abc import ABC, abstractmethod


class SamplingStrategy(ABC):
    """Abstract base class for sampling strategies."""

    @abstractmethod
    def select_samples(self, unlabeled_data: np.ndarray,
                      num_samples: int,
                      model_predictions: Dict = None) -> List[int]:
        """
        Select samples for labeling.

        Args:
            unlabeled_data: Unlabeled data points
            num_samples: Number of samples to select
            model_predictions: Model predictions (optional)

        Returns:
            Indices of selected samples
        """
        pass


class RandomSampling(SamplingStrategy):
    """Random sampling strategy."""

    def select_samples(self, unlabeled_data: np.ndarray,
                      num_samples: int,
                      model_predictions: Dict = None) -> List[int]:
        """Randomly select samples."""
        n_total = len(unlabeled_data)
        return np.random.choice(n_total, num_samples, replace=False).tolist()


class UncertaintySampling(SamplingStrategy):
    """Uncertainty-based sampling strategy."""

    def __init__(self, uncertainty_measure: str = 'least_confident'):
        """
        Initialize uncertainty sampling.

        Args:
            uncertainty_measure: Type of uncertainty measure
        """
        self.uncertainty_measure = uncertainty_measure

    def select_samples(self, unlabeled_data: np.ndarray,
                      num_samples: int,
                      model_predictions: Dict = None) -> List[int]:
        """Select samples based on uncertainty."""
        if model_predictions is None:
            raise ValueError("Model predictions required for uncertainty sampling")

        uncertainties = self._calculate_uncertainty(model_predictions)
        # Select samples with highest uncertainty
        indices = np.argsort(uncertainties)[-num_samples:]
        return indices.tolist()

    def _calculate_uncertainty(self, predictions: Dict) -> np.ndarray:
        """Calculate uncertainty scores."""
        if self.uncertainty_measure == 'least_confident':
            # Use 1 - max probability as uncertainty
            probs = predictions.get('probabilities', np.random.random((len(predictions), 2)))
            return 1 - np.max(probs, axis=1)
        elif self.uncertainty_measure == 'entropy':
            # Use entropy as uncertainty measure
            probs = predictions.get('probabilities', np.random.random((len(predictions), 2)))
            return -np.sum(probs * np.log(probs + 1e-10), axis=1)
        else:
            raise ValueError(f"Unknown uncertainty measure: {self.uncertainty_measure}")


class DiversitySampling(SamplingStrategy):
    """Diversity-based sampling strategy."""

    def select_samples(self, unlabeled_data: np.ndarray,
                      num_samples: int,
                      model_predictions: Dict = None) -> List[int]:
        """Select diverse samples."""
        # Implement diversity sampling (e.g., using clustering)
        # Placeholder: random selection for now
        n_total = len(unlabeled_data)
        return np.random.choice(n_total, num_samples, replace=False).tolist()


class RepresentativeSampling(SamplingStrategy):
    """Representative sampling based on data distribution."""

    def select_samples(self, unlabeled_data: np.ndarray,
                      num_samples: int,
                      model_predictions: Dict = None) -> List[int]:
        """Select representative samples."""
        # Implement representative sampling
        # Placeholder: random selection for now
        n_total = len(unlabeled_data)
        return np.random.choice(n_total, num_samples, replace=False).tolist()
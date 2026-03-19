"""
Active Learning Module for fluProfiler.

This module contains:
- Sample selection strategies
- Uncertainty quantification
- Query strategies and scoring functions
- Active learning loop implementations
- Dataset classes for fluProfiler
"""

from .dataset import FluProfilerDataset
from .strategies import (
    SamplingStrategy,
    RandomSampling,
    UncertaintySampling,
    DiversitySampling,
    RepresentativeSampling,
    HybridStrategy,
)
from .scoring import (
    uncertainty_score,
    diversity_score,
    representativeness_score,
    combined_score,
    antigenic_distance_score,
    evolutionary_distance_score,
)
from .loop import ActiveLearningLoop
from .optimized import OptimizedActiveLearning

__all__ = [
    # Dataset
    'FluProfilerDataset',
    # Strategies
    'SamplingStrategy',
    'RandomSampling',
    'UncertaintySampling',
    'DiversitySampling',
    'RepresentativeSampling',
    'HybridStrategy',
    # Scoring
    'uncertainty_score',
    'diversity_score',
    'representativeness_score',
    'combined_score',
    'antigenic_distance_score',
    'evolutionary_distance_score',
    # Loop
    'ActiveLearningLoop',
    # Optimized
    'OptimizedActiveLearning',
]

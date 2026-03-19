"""
Dataset classes for fluProfiler active learning.
"""

import torch
from torch.utils.data import Dataset
from typing import List, Optional
import pandas as pd
import numpy as np


class FluProfilerDataset(Dataset):
    """
    fluProfiler dataset class for active learning.
    
    Handles 4-tuple sequence embeddings with strain passage categories
    and optional institute information.
    """
    
    def __init__(
        self,
        DataFrame: pd.DataFrame,
        include_institute: bool = True,
        emb_suffix: str = 'matrix_'
    ):
        """
        Initialize fluProfiler dataset.
        
        Args:
            DataFrame: DataFrame containing sequence IDs and labels
            include_institute: Whether to include institute information
            emb_suffix: Suffix for embedding file names
        """
        self.emb_file_name_a = (emb_suffix + DataFrame['seq_id_a']).tolist()
        self.emb_file_name_b = (emb_suffix + DataFrame['seq_id_b']).tolist()
        self.emb_file_name_c = (emb_suffix + DataFrame['seq_id_c']).tolist()
        self.emb_file_name_d = (emb_suffix + DataFrame['seq_id_d']).tolist()

        self.strainPassCats = self._convert_pass2tensor(
            ('<cls>' + DataFrame['serumPassCat'] + '<eos>' + 
             DataFrame['virusPassCat'] + '<eos>').tolist()
        )
        
        if include_institute and 'institute' in DataFrame.columns:
            self.institute = self._convert_institute2tensor(
                DataFrame['institute'].tolist()
            )
        else:
            self.institute = None
            
        self.labels = torch.tensor(DataFrame['label'].tolist(), dtype=torch.float32)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if self.institute is not None:
            return (
                self.emb_file_name_a[idx],
                self.emb_file_name_b[idx],
                self.emb_file_name_c[idx],
                self.emb_file_name_d[idx],
                self.strainPassCats[idx],
                self.institute[idx],
                self.labels[idx]
            )
        return (
            self.emb_file_name_a[idx],
            self.emb_file_name_b[idx],
            self.emb_file_name_c[idx],
            self.emb_file_name_d[idx],
            self.strainPassCats[idx],
            self.labels[idx]
        )
    
    @staticmethod
    def _convert_pass2tensor(pass_cats: List[str]) -> torch.Tensor:
        """
        Convert passage categories to tensor.
        
        Args:
            pass_cats: List of passage category strings
            
        Returns:
            Tensor of passage category encodings
        """
        result = [
            item.replace('<cls>', '0')
                .replace('<eos>', '1')
                .replace('<EGG>', '2')
                .replace('<CELL>', '3')
                .replace('<BOTH>', '4')
            for item in pass_cats
        ]
        result = torch.tensor([
            [int(number) for number in [char for char in item]]
            for item in result
        ])
        return result
    
    @staticmethod
    def _convert_institute2tensor(institutes: List[str]) -> torch.Tensor:
        """
        Convert institute categories to tensor.
        
        Args:
            institutes: List of institute strings
            
        Returns:
            Tensor of institute encodings
        """
        result = [
            item.replace('<Crick>', '0')
                .replace('<CDC>', '1')
                .replace('<CNIC>', '2')
            for item in institutes
        ]
        result = torch.tensor([
            [int(number) for number in [char for char in item]]
            for item in result
        ])
        return result


class ActiveLearningPool:
    """
    Active learning pool for managing labeled and unlabeled data.
    """
    
    def __init__(self, dataset: FluProfilerDataset, initial_labeled_ratio: float = 0.1):
        """
        Initialize active learning pool.
        
        Args:
            dataset: The full dataset
            initial_labeled_ratio: Ratio of initially labeled samples
        """
        self.dataset = dataset
        self.total_size = len(dataset)
        
        # Initialize indices
        all_indices = list(range(self.total_size))
        np.random.shuffle(all_indices)
        
        initial_labeled_count = int(self.total_size * initial_labeled_ratio)
        self.labeled_indices = set(all_indices[:initial_labeled_count])
        self.unlabeled_indices = set(all_indices[initial_labeled_count:])
        
    def get_labeled_data(self) -> List[int]:
        """Get list of labeled indices."""
        return list(self.labeled_indices)
    
    def get_unlabeled_data(self) -> List[int]:
        """Get list of unlabeled indices."""
        return list(self.unlabeled_indices)
    
    def label_samples(self, indices: List[int]):
        """
        Move samples from unlabeled to labeled pool.
        
        Args:
            indices: Indices to label
        """
        for idx in indices:
            if idx in self.unlabeled_indices:
                self.unlabeled_indices.remove(idx)
                self.labeled_indices.add(idx)
                
    def get_statistics(self) -> dict:
        """Get current pool statistics."""
        return {
            'total': self.total_size,
            'labeled': len(self.labeled_indices),
            'unlabeled': len(self.unlabeled_indices),
            'labeling_ratio': len(self.labeled_indices) / self.total_size
        }

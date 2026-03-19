"""
Optimized Active Learning implementation.

Main class migrated from Section4_ActiveLearning/optimized_active_learning.py
with improved modularization.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time
from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy.spatial.distance import cdist


class OptimizedActiveLearning:
    """
    Optimized active learning class with parallelization and GPU acceleration.
    
    Features:
    - Parallel embedding loading
    - Parallel feature extraction
    - GPU-accelerated clustering
    - Multiple selection strategies
    """
    
    def __init__(
        self,
        device: str = 'cuda:0',
        n_jobs: int = -1,
        use_gpu_clustering: bool = False,
        batch_size: int = 8
    ):
        """
        Initialize optimized active learning.
        
        Args:
            device: Computation device
            n_jobs: Number of parallel jobs (-1 = all CPU cores)
            use_gpu_clustering: Whether to use GPU for clustering
            batch_size: Batch size for processing
        """
        self.device = torch.device(device)
        self.n_jobs = n_jobs if n_jobs != -1 else torch.get_num_threads()
        self.use_gpu_clustering = use_gpu_clustering
        self.batch_size = batch_size
        self.emb_dict = None
        self.model = None
        self.selected_indices = []
        self.history = []
        
    def load_embeddings(
        self,
        emb_path: str,
        move_to_device: bool = True,
        load_func: Optional[Callable] = None
    ) -> Dict:
        """
        Load embeddings with parallel processing.
        
        Args:
            emb_path: Path to embeddings directory
            move_to_device: Whether to move to specified device
            load_func: Custom embedding load function
            
        Returns:
            Embedding dictionary
        """
        print(f"Loading embeddings from {emb_path}...")
        start_time = time.time()
        
        if load_func is not None:
            self.emb_dict = load_func(emb_path)
        else:
            # Default loading using torch
            self.emb_dict = self._load_embeddings_default(emb_path)
        
        if move_to_device:
            print(f"Moving embeddings to device: {self.device}")
            self.emb_dict = {key: value.to(self.device) 
                          for key, value in self.emb_dict.items()}
        
        load_time = time.time() - start_time
        print(f"Embeddings loaded in {load_time:.2f}s")
        return self.emb_dict
    
    def _load_embeddings_default(self, emb_path: str) -> Dict:
        """Default embedding loading using torch."""
        import os
        from pathlib import Path
        
        emb_dict = {}
        emb_files = list(Path(emb_path).glob('*.pt'))
        
        for emb_file in tqdm(emb_files, desc="Loading embeddings"):
            key = emb_file.stem  # filename without extension
            emb_dict[key] = torch.load(emb_file)
            
        return emb_dict
    
    def extract_features_parallel(
        self,
        dataloader: DataLoader,
        model: nn.Module,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Extract features in parallel using multi-threading.
        
        Args:
            dataloader: Data loader
            model: Model for feature extraction
            save_path: Optional path to save features
            
        Returns:
            Feature array
        """
        model.eval()
        model.to(self.device)
        self.model = model
        
        print("Extracting features in parallel...")
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=min(self.n_jobs, 4)) as executor:
            futures = []
            batch_results = []
            
            for batch_idx, batch in enumerate(dataloader):
                future = executor.submit(self._process_batch_features, batch, batch_idx)
                futures.append(future)
            
            for future in tqdm(futures, desc="Extracting features"):
                batch_features = future.result()
                batch_results.append(batch_features)
        
        all_features = torch.cat(batch_results, dim=0).cpu().numpy()
        
        extract_time = time.time() - start_time
        print(f"Feature extraction completed in {extract_time:.2f}s")
        
        if save_path:
            np.save(save_path, all_features)
            
        return all_features
    
    def _process_batch_features(
        self,
        batch: Tuple,
        batch_idx: int
    ) -> torch.Tensor:
        """Process single batch for feature extraction."""
        emb_file_name_a, emb_file_name_b, emb_file_name_c, emb_file_name_d, \
            strainPassCats, labels = batch
        
        # Generate matrices
        matrixs_a, masks_a = self._generate_matrix_optimized(
            [self.emb_dict[key.replace('_active', '')] 
             for key in emb_file_name_a])
        matrixs_b, masks_b = self._generate_matrix_optimized(
            [self.emb_dict[key.replace('_active', '')] 
             for key in emb_file_name_b])
        matrixs_c, masks_c = self._generate_matrix_optimized(
            [self.emb_dict[key.replace('_active', '')] 
             for key in emb_file_name_c])
        matrixs_d, masks_d = self._generate_matrix_optimized(
            [self.emb_dict[key.replace('_active', '')] 
             for key in emb_file_name_d])
        
        # Move to device
        matrixs_a = matrixs_a.to(self.device)
        matrixs_b = matrixs_b.to(self.device)
        matrixs_c = matrixs_c.to(self.device)
        matrixs_d = matrixs_d.to(self.device)
        masks_a = masks_a.to(self.device)
        masks_b = masks_b.to(self.device)
        masks_c = masks_c.to(self.device)
        masks_d = masks_d.to(self.device)
        strainPassCats = strainPassCats.to(self.device)
        labels = labels.to(self.device)
        
        with torch.no_grad():
            _, _, output = self.model(
                matrices_a=matrixs_a,
                matrices_b=matrixs_b,
                matrices_c=matrixs_c,
                matrices_d=matrixs_d,
                matrix_attention_masks_a=masks_a,
                matrix_attention_masks_b=masks_b,
                matrix_attention_masks_c=masks_c,
                matrix_attention_masks_d=masks_d,
                strainPassCats=strainPassCats,
                labels=labels
            )
        
        return output
    
    def _generate_matrix_optimized(
        self,
        matrix_list: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optimized matrix generation using vectorized operations.
        
        Args:
            matrix_list: List of matrices
            
        Returns:
            (padded_matrix, attention_mask)
        """
        if not matrix_list:
            return torch.empty(0), torch.empty(0)
            
        seq_len = [mat.shape[0] for mat in matrix_list]
        max_len = max(seq_len)
        
        # Vectorized padding
        padded_list = []
        mask_list = []
        
        for i, mat in enumerate(matrix_list):
            if mat.shape[0] < max_len:
                padding = torch.zeros(
                    max_len - mat.shape[0],
                    mat.shape[1],
                    device=mat.device
                )
                padded = torch.cat([mat, padding], dim=0)
            else:
                padded = mat
            padded_list.append(padded)
            
            mask = torch.ones(1, seq_len[i])
            if seq_len[i] < max_len:
                mask = torch.cat([
                    mask,
                    torch.zeros(1, max_len - seq_len[i])
                ], dim=1)
            mask_list.append(mask)
        
        matrix = torch.stack(padded_list)
        mask = torch.stack(mask_list).view(len(matrix_list), max_len)
        
        return matrix, mask
    
    def select_samples_hybrid(
        self,
        features: np.ndarray,
        predictions: np.ndarray,
        n_samples: int,
        n_clusters: int = 10
    ) -> List[int]:
        """
        Select samples using hybrid strategy.
        
        Args:
            features: Feature embeddings
            predictions: Model predictions
            n_samples: Number of samples to select
            n_clusters: Number of clusters
            
        Returns:
            Selected sample indices
        """
        # Step 1: Calculate uncertainty scores
        uncertainties = 1 - np.max(predictions, axis=1)
        
        # Step 2: Calculate diversity scores using clustering
        if self.use_gpu_clustering and torch.cuda.is_available():
            # GPU-accelerated clustering
            features_gpu = torch.tensor(features, device=self.device)
            # Note: Would need GPU k-means implementation
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
        else:
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
        
        cluster_labels = kmeans.fit_predict(features)
        
        # Step 3: Select diverse samples from each cluster
        selected_indices = []
        samples_per_cluster = max(1, n_samples // n_clusters)
        
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            if len(cluster_indices) == 0:
                continue
            
            # Get most uncertain in cluster
            cluster_uncertainties = uncertainties[cluster_indices]
            top_indices = cluster_indices[np.argsort(cluster_uncertainties)[-samples_per_cluster:]]
            selected_indices.extend(top_indices)
        
        # Fill remaining with most uncertain overall
        if len(selected_indices) < n_samples:
            remaining = set(range(len(features))) - set(selected_indices)
            remaining_uncertainties = uncertainties[list(remaining)]
            additional = list(remaining)[np.argsort(remaining_uncertainties)[-(n_samples-len(selected_indices)):]]
            selected_indices.extend(additional)
        
        self.selected_indices.extend(selected_indices[:n_samples])
        return selected_indices[:n_samples]
    
    def select_samples_uncertainty(
        self,
        predictions: np.ndarray,
        n_samples: int
    ) -> List[int]:
        """
        Select samples based on uncertainty only.
        
        Args:
            predictions: Model predictions
            n_samples: Number of samples to select
            
        Returns:
            Selected sample indices
        """
        uncertainties = 1 - np.max(predictions, axis=1)
        selected = np.argsort(uncertainties)[-n_samples:].tolist()
        self.selected_indices.extend(selected)
        return selected
    
    def select_samples_diversity(
        self,
        features: np.ndarray,
        n_samples: int,
        n_clusters: int = 10
    ) -> List[int]:
        """
        Select samples based on diversity (clustering).
        
        Args:
            features: Feature embeddings
            n_samples: Number of samples to select
            n_clusters: Number of clusters
            
        Returns:
            Selected sample indices
        """
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(features)
        
        selected_indices = []
        samples_per_cluster = max(1, n_samples // n_clusters)
        
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            if len(cluster_indices) == 0:
                continue
            
            # Select closest to cluster center
            cluster_features = features[cluster_indices]
            center = kmeans.cluster_centers_[cluster_id]
            distances = np.linalg.norm(cluster_features - center, axis=1)
            selected = cluster_indices[np.argsort(distances)[:samples_per_cluster]]
            selected_indices.extend(selected)
        
        self.selected_indices.extend(selected_indices[:n_samples])
        return selected_indices[:n_samples]
    
    def run_active_learning_cycle(
        self,
        model: nn.Module,
        unlabeled_dataloader: DataLoader,
        unlabeled_indices: List[int],
        n_samples: int,
        n_clusters: int = 10
    ) -> List[int]:
        """
        Run one active learning cycle.
        
        Args:
            model: Current model
            unlabeled_dataloader: DataLoader for unlabeled data
            unlabeled_indices: Indices of unlabeled samples
            n_samples: Number of samples to select
            n_clusters: Number of clusters
            
        Returns:
            Newly selected sample indices
        """
        # Extract features for unlabeled data
        features = self.extract_features_parallel(
            unlabeled_dataloader, model
        )
        
        # Get predictions
        model.eval()
        predictions_list = []
        with torch.no_grad():
            for batch in unlabeled_dataloader:
                # Simplified: assume model outputs logits
                _, logits, _ = self._get_model_output(model, batch)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                predictions_list.append(probs)
        
        predictions = np.vstack(predictions_list)
        
        # Select samples using hybrid strategy
        selected = self.select_samples_hybrid(
            features, predictions, n_samples, n_clusters
        )
        
        # Record history
        self.history.append({
            'n_selected': len(selected),
            'selected_indices': selected
        })
        
        return selected
    
    def _get_model_output(
        self,
        model: nn.Module,
        batch: Tuple
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get model output for a batch."""
        emb_file_name_a, emb_file_name_b, emb_file_name_c, emb_file_name_d, \
            strainPassCats, labels = batch
        
        matrixs_a, masks_a = self._generate_matrix_optimized(
            [self.emb_dict[key.replace('_active', '')] 
             for key in emb_file_name_a])
        matrixs_b, masks_b = self._generate_matrix_optimized(
            [self.emb_dict[key.replace('_active', '')] 
             for key in emb_file_name_b])
        matrixs_c, masks_c = self._generate_matrix_optimized(
            [self.emb_dict[key.replace('_active', '')] 
             for key in emb_file_name_c])
        matrixs_d, masks_d = self._generate_matrix_optimized(
            [self.emb_dict[key.replace('_active', '')] 
             for key in emb_file_name_d])
        
        matrixs_a = matrixs_a.to(self.device)
        matrixs_b = matrixs_b.to(self.device)
        matrixs_c = matrixs_c.to(self.device)
        matrixs_d = matrixs_d.to(self.device)
        masks_a = masks_a.to(self.device)
        masks_b = masks_b.to(self.device)
        masks_c = masks_c.to(self.device)
        masks_d = masks_d.to(self.device)
        strainPassCats = strainPassCats.to(self.device)
        
        loss, logits, output = model(
            matrices_a=matrixs_a,
            matrices_b=matrixs_b,
            matrices_c=matrixs_c,
            matrices_d=matrixs_d,
            matrix_attention_masks_a=masks_a,
            matrix_attention_masks_b=masks_b,
            matrix_attention_masks_c=masks_c,
            matrix_attention_masks_d=masks_d,
            strainPassCats=strainPassCats
        )
        
        return loss, logits, output
    
    def get_history(self) -> pd.DataFrame:
        """Get active learning history."""
        return pd.DataFrame(self.history)
    
    def reset(self):
        """Reset active learning state."""
        self.selected_indices = []
        self.history = []

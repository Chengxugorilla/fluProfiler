import torch
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
from typing import List, Tuple, Dict, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# 添加路徑
sys.path.append('../')
from utilities import load_embedding, print_exams

class fluProfiler_Dataset(Dataset):
    """流感分析器數據集類"""
    def __init__(self, DataFrame):
        self.emb_file_name_a = ('matrix_' + DataFrame['seq_id_a']).tolist()
        self.emb_file_name_b = ('matrix_' + DataFrame['seq_id_b']).tolist()
        self.emb_file_name_c = ('matrix_' + DataFrame['seq_id_c']).tolist()
        self.emb_file_name_d = ('matrix_' + DataFrame['seq_id_d']).tolist()

        self.strainPassCats = self._convert_Pass2tensor(('<cls>' + DataFrame['serumPassCat'] + '<eos>' + DataFrame['virusPassCat'] + '<eos>').tolist())
        self.labels = torch.tensor(DataFrame['label'].tolist())
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.emb_file_name_a[idx], self.emb_file_name_b[idx], self.emb_file_name_c[idx], self.emb_file_name_d[idx], \
               self.strainPassCats[idx], self.labels[idx]
    
    def _convert_Pass2tensor(self, pass_cats):
        """轉換傳遞類別為張量"""
        result = [
            item.replace('<cls>', '0').replace('<eos>', '1').replace('<EGG>', '2').replace('<CELL>', '3').replace('<BOTH>', '4')
            for item in pass_cats
        ]
        result = torch.tensor([[int(number) for number in [char for char in item]] for item in result])
        return result

class OptimizedActiveLearning:
    """
    優化的主動學習類，支持並行化和多種選擇策略
    """
    
    def __init__(self, device: str = 'cuda:0', n_jobs: int = -1, 
                 use_gpu_clustering: bool = False, batch_size: int = 8):
        """
        初始化優化的主動學習器
        
        Args:
            device: 計算設備
            n_jobs: 並行作業數，-1表示使用所有CPU核心
            use_gpu_clustering: 是否使用GPU加速聚類
            batch_size: 批次大小
        """
        self.device = torch.device(device)
        self.n_jobs = n_jobs if n_jobs != -1 else mp.cpu_count()
        self.use_gpu_clustering = use_gpu_clustering
        self.batch_size = batch_size
        self.emb_dict = None
        self.model = None
        
    def load_embeddings(self, emb_path: str, move_to_device: bool = True) -> Dict:
        """
        並行加載嵌入向量
        
        Args:
            emb_path: 嵌入文件路徑
            move_to_device: 是否移動到指定設備
            
        Returns:
            嵌入字典
        """
        print(f"正在並行加載嵌入向量...")
        start_time = time.time()
        
        self.emb_dict = load_embedding(emb_path)
        
        if move_to_device:
            print(f"移動嵌入向量到設備: {self.device}")
            self.emb_dict = {key: value.to(self.device) for key, value in self.emb_dict.items()}
        
        load_time = time.time() - start_time
        print(f"嵌入向量加載完成，耗時: {load_time:.2f}秒")
        return self.emb_dict
    
    def extract_features_parallel(self, dataloader: DataLoader, model: torch.nn.Module, 
                                 save_path: Optional[str] = None) -> np.ndarray:
        """
        並行提取特徵向量
        
        Args:
            dataloader: 數據加載器
            model: 模型
            save_path: 保存路徑
            
        Returns:
            特徵向量數組
        """
        model.eval()
        model.to(self.device)
        self.model = model
        
        print("開始並行特徵提取...")
        start_time = time.time()
        
        # 使用多線程並行處理批次
        with ThreadPoolExecutor(max_workers=min(self.n_jobs, 4)) as executor:
            futures = []
            batch_results = []
            
            for batch_idx, batch in enumerate(dataloader):
                future = executor.submit(self._process_batch_features, batch, batch_idx)
                futures.append(future)
            
            # 收集結果
            for future in tqdm(futures, desc="提取特徵向量"):
                batch_features = future.result()
                batch_results.append(batch_features)
        
        # 合併所有批次結果
        all_features = torch.cat(batch_results, dim=0).cpu().numpy()
        
        extract_time = time.time() - start_time
        print(f"特徵提取完成，耗時: {extract_time:.2f}秒")
        
        if save_path:
            self._save_features(all_features, save_path)
            
        return all_features
    
    def _process_batch_features(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        """
        處理單個批次的特徵提取
        
        Args:
            batch: 批次數據
            batch_idx: 批次索引
            
        Returns:
            批次特徵張量
        """
        emb_file_name_a, emb_file_name_b, emb_file_name_c, emb_file_name_d, strainPassCats, labels = batch
        
        # 生成矩陣
        matrixs_a, masks_a = self._generate_matrix_optimized(
            [self.emb_dict[key.replace('_active', '')] for key in emb_file_name_a])
        matrixs_b, masks_b = self._generate_matrix_optimized(
            [self.emb_dict[key.replace('_active', '')] for key in emb_file_name_b])
        matrixs_c, masks_c = self._generate_matrix_optimized(
            [self.emb_dict[key.replace('_active', '')] for key in emb_file_name_c])
        matrixs_d, masks_d = self._generate_matrix_optimized(
            [self.emb_dict[key.replace('_active', '')] for key in emb_file_name_d])
        
        # 移動到設備
        matrixs_a, matrixs_b, matrixs_c, matrixs_d = matrixs_a.to(self.device), matrixs_b.to(self.device), matrixs_c.to(self.device), matrixs_d.to(self.device)
        masks_a, masks_b, masks_c, masks_d = masks_a.to(self.device), masks_b.to(self.device), masks_c.to(self.device), masks_d.to(self.device)
        strainPassCats, labels = strainPassCats.to(self.device), labels.to(self.device)
        
        with torch.no_grad():
                _, _, output = self.model(
                    matrices_a=matrixs_a, matrices_b=matrixs_b, matrices_c=matrixs_c, matrices_d=matrixs_d,
                    matrix_attention_masks_a=masks_a, matrix_attention_masks_b=masks_b, 
                    matrix_attention_masks_c=masks_c, matrix_attention_masks_d=masks_d,
                    strainPassCats=strainPassCats, labels=labels
                )
        
        return output
    
    def _generate_matrix_optimized(self, matrix_list: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        優化的矩陣生成函數，使用向量化操作
        
        Args:
            matrix_list: 矩陣列表
            
        Returns:
            填充後的矩陣和掩碼
        """
        if not matrix_list:
            return torch.empty(0), torch.empty(0)
            
        seq_len = [mat.shape[0] for mat in matrix_list]
        max_len = max(seq_len)
        
        # 向量化填充
        padded_matrices = []
        masks = []
        
        for i, mat in enumerate(matrix_list):
            if mat.shape[0] < max_len:
                padding = max_len - mat.shape[0]
                padded_mat = F.pad(mat, (0, 0, 0, padding))
            else:
                padded_mat = mat
            padded_matrices.append(padded_mat)
            
            # 生成掩碼
            mask = torch.zeros(1, max_len)
            mask[0, :seq_len[i]] = 1
            masks.append(mask)
        
        matrix = torch.stack(padded_matrices)
        mask = torch.stack(masks).view(len(matrix_list), max_len)
        
        return matrix, mask
    
    def optimized_clustering_selection(self, features: np.ndarray, candidate_data: pd.DataFrame,
                                     sample_size: int = 200, n_clusters: int = 8,
                                     clustering_method: str = 'kmeans',
                                     selection_strategy: str = 'proportional') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        優化的聚類選擇算法
        
        Args:
            features: 特徵向量
            candidate_data: 候選數據
            sample_size: 樣本大小
            n_clusters: 聚類數量
            clustering_method: 聚類方法 ('kmeans', 'minibatch_kmeans')
            selection_strategy: 選擇策略 ('proportional', 'diverse', 'uncertainty')
            
        Returns:
            active_selection, random_selection
        """
        print(f"開始優化聚類選擇 (方法: {clustering_method}, 策略: {selection_strategy})...")
        start_time = time.time()
        
        # 選擇聚類算法
        if clustering_method == 'minibatch_kmeans':
            clusterer = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, 
                                      batch_size=1000, n_init=3)
        else:
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        
        # 執行聚類
        cluster_labels = clusterer.fit_predict(features)
        candidate_data = candidate_data.copy()
        candidate_data['cluster'] = cluster_labels
        
        # 根據策略選擇樣本
        if selection_strategy == 'proportional':
            active_selection = self._proportional_selection(candidate_data, sample_size)
        elif selection_strategy == 'diverse':
            active_selection = self._diverse_selection(candidate_data, features, sample_size)
        elif selection_strategy == 'uncertainty':
            active_selection = self._uncertainty_selection(candidate_data, features, sample_size)
        else:
            raise ValueError(f"未知的選擇策略: {selection_strategy}")
        
        # 生成隨機選擇作為對照
        random_selection = candidate_data.sample(n=len(active_selection), random_state=42)
        
        selection_time = time.time() - start_time
        print(f"聚類選擇完成，耗時: {selection_time:.2f}秒")
        print(f"Active selection: {len(active_selection)} 樣本")
        print(f"Random selection: {len(random_selection)} 樣本")
        
        return active_selection, random_selection
    
    def _proportional_selection(self, data: pd.DataFrame, sample_size: int) -> pd.DataFrame:
        """
        按比例選擇樣本
        
        Args:
            data: 數據框
            sample_size: 目標樣本大小
            
        Returns:
            選擇的樣本
        """
        cluster_counts = data['cluster'].value_counts()
        total_samples = len(data)
        sample_ratio = min(sample_size / total_samples, 1.0)
        
        selected_samples = []
        for cluster_id in cluster_counts.index:
            cluster_data = data[data['cluster'] == cluster_id]
            n_samples = max(1, int(len(cluster_data) * sample_ratio))
            selected = cluster_data.sample(n=n_samples, random_state=42)
            selected_samples.append(selected)
        
        return pd.concat(selected_samples, ignore_index=True)
    
    def _diverse_selection(self, data: pd.DataFrame, features: np.ndarray, sample_size: int) -> pd.DataFrame:
        """
        多樣性選擇：選擇距離聚類中心最遠的樣本
        
        Args:
            data: 數據框
            features: 特徵向量
            sample_size: 目標樣本大小
            
        Returns:
            選擇的樣本
        """
        # 計算每個樣本到其聚類中心的距離
        distances = []
        for idx, row in data.iterrows():
            cluster_id = row['cluster']
            cluster_center = features[data['cluster'] == cluster_id].mean(axis=0)
            distance = np.linalg.norm(features[idx] - cluster_center)
            distances.append(distance)
        
        data['distance_to_center'] = distances
        
        # 從每個聚類中選擇距離中心最遠的樣本
        selected_samples = []
        for cluster_id in data['cluster'].unique():
            cluster_data = data[data['cluster'] == cluster_id]
            n_samples = max(1, int(sample_size * len(cluster_data) / len(data)))
            selected = cluster_data.nlargest(n_samples, 'distance_to_center')
            selected_samples.append(selected)
        
        return pd.concat(selected_samples, ignore_index=True)
    
    def _uncertainty_selection(self, data: pd.DataFrame, features: np.ndarray, sample_size: int) -> pd.DataFrame:
        """
        不確定性選擇：選擇特徵空間中不確定的樣本
        
        Args:
            data: 數據框
            features: 特徵向量
            sample_size: 目標樣本大小
            
        Returns:
            選擇的樣本
        """
        # 計算樣本間的平均距離作為不確定性指標
        uncertainties = []
        for i in range(len(features)):
            distances = np.linalg.norm(features - features[i], axis=1)
            uncertainty = np.mean(distances)
            uncertainties.append(uncertainty)
        
        data['uncertainty'] = uncertainties
        
        # 選擇不確定性最高的樣本
        return data.nlargest(sample_size, 'uncertainty')
    
    def parallel_model_training(self, model: torch.nn.Module, train_dataloader: DataLoader,
                              valid_dataloader: DataLoader, test_dataloader: DataLoader,
                              lr_rate: float = 0.00004, epochs: int = 15,
                              use_mixed_precision: bool = True) -> Dict:
        """
        並行模型訓練
        
        Args:
            model: 模型
            train_dataloader: 訓練數據加載器
            valid_dataloader: 驗證數據加載器
            test_dataloader: 測試數據加載器
            lr_rate: 學習率
            epochs: 訓練輪數
            use_mixed_precision: 是否使用混合精度
            
        Returns:
            訓練結果字典
        """
        model.to(self.device)
        
        # 設置優化器
        no_decay = ["bias", "layernorm.weight", "layer_norm.weight", "layer.norm.weight"]
        weight_decay = 0.01
        
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() 
                          if not any(nd in n.lower() for nd in no_decay)],
                "weight_decay": weight_decay
            },
            {
                "params": [p for n, p in model.named_parameters() 
                          if any(nd in n.lower() for nd in no_decay)],
                "weight_decay": 0.0
            }
        ]
        
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr_rate, 
                         betas=[0.9, 0.99], eps=1e-08)
        
        # 混合精度訓練
        scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None
        
        # 訓練循環
        results = {
            'valid_MAE': [], 'valid_MSE': [], 'valid_Pearson': [], 'valid_Spearman': [],
            'test_MAE': [], 'test_MSE': [], 'test_Pearson': [], 'test_Spearman': []
        }
        
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            
            # 訓練階段
            train_loss = self._train_epoch(model, train_dataloader, optimizer, scaler)
            
            # 驗證階段
            valid_metrics = self._evaluate_epoch(model, valid_dataloader, 'valid')
            
            # 測試階段
            test_metrics = self._evaluate_epoch(model, test_dataloader, 'test')
            
            # 記錄結果
            for key, value in valid_metrics.items():
                results[key].append(value)
            for key, value in test_metrics.items():
                results[key].append(value)
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Valid MSE: {valid_metrics['valid_MSE']:.4f}")
            print(f"Test MSE: {test_metrics['test_MSE']:.4f}")
        
        return results
    
    def _train_epoch(self, model: torch.nn.Module, dataloader: DataLoader, 
                    optimizer: torch.optim.Optimizer, scaler: Optional[torch.cuda.amp.GradScaler]) -> float:
        """
        訓練一個epoch
        
        Args:
            model: 模型
            dataloader: 數據加載器
            optimizer: 優化器
            scaler: 混合精度縮放器
            
        Returns:
            平均損失
        """
        model.train()
        total_loss = 0.0
        
        for batch in tqdm(dataloader, desc="Training"):
            emb_file_name_a, emb_file_name_b, emb_file_name_c, emb_file_name_d, strainPassCats, labels = batch
            
            # 生成矩陣
            matrixs_a, masks_a = self._generate_matrix_optimized(
                [self.emb_dict[key.replace('_active', '') if key.endswith('_active') else key] for key in emb_file_name_a])
            matrixs_b, masks_b = self._generate_matrix_optimized(
                [self.emb_dict[key.replace('_active', '') if key.endswith('_active') else key] for key in emb_file_name_b])
            matrixs_c, masks_c = self._generate_matrix_optimized(
                [self.emb_dict[key.replace('_active', '') if key.endswith('_active') else key] for key in emb_file_name_c])
            matrixs_d, masks_d = self._generate_matrix_optimized(
                [self.emb_dict[key.replace('_active', '') if key.endswith('_active') else key] for key in emb_file_name_d])
            
            # 移動到設備
            masks_a, masks_b, masks_c, masks_d = masks_a.to(self.device), masks_b.to(self.device), masks_c.to(self.device), masks_d.to(self.device)
            strainPassCats, labels = strainPassCats.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    loss, _, _ = model(
                        matrices_a=matrixs_a, matrices_b=matrixs_b, matrices_c=matrixs_c, matrices_d=matrixs_d,
                        matrix_attention_masks_a=masks_a, matrix_attention_masks_b=masks_b,
                        matrix_attention_masks_c=masks_c, matrix_attention_masks_d=masks_d,
                        strainPassCats=strainPassCats, labels=labels
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss, _, _ = model(
                    matrices_a=matrixs_a, matrices_b=matrixs_b, matrices_c=matrixs_c, matrices_d=matrixs_d,
                    matrix_attention_masks_a=masks_a, matrix_attention_masks_b=masks_b,
                    matrix_attention_masks_c=masks_c, matrix_attention_masks_d=masks_d,
                    strainPassCats=strainPassCats, labels=labels
                )
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def _evaluate_epoch(self, model: torch.nn.Module, dataloader: DataLoader, 
                       prefix: str) -> Dict[str, float]:
        """
        評估一個epoch
        
        Args:
            model: 模型
            dataloader: 數據加載器
            prefix: 前綴
            
        Returns:
            評估指標字典
        """
        model.eval()
        predictions = []
        references = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {prefix}"):
                emb_file_name_a, emb_file_name_b, emb_file_name_c, emb_file_name_d, strainPassCats, labels = batch
                
                # 生成矩陣
                matrixs_a, masks_a = self._generate_matrix_optimized(
                    [self.emb_dict[key.replace('_active', '') if key.endswith('_active') else key] for key in emb_file_name_a])
                matrixs_b, masks_b = self._generate_matrix_optimized(
                    [self.emb_dict[key.replace('_active', '') if key.endswith('_active') else key] for key in emb_file_name_b])
                matrixs_c, masks_c = self._generate_matrix_optimized(
                    [self.emb_dict[key.replace('_active', '') if key.endswith('_active') else key] for key in emb_file_name_c])
                matrixs_d, masks_d = self._generate_matrix_optimized(
                    [self.emb_dict[key.replace('_active', '') if key.endswith('_active') else key] for key in emb_file_name_d])
                
                # 移動到設備
                matrixs_a, matrixs_b, matrixs_c, matrixs_d = matrixs_a.to(self.device), matrixs_b.to(self.device), matrixs_c.to(self.device), matrixs_d.to(self.device)
                masks_a, masks_b, masks_c, masks_d = masks_a.to(self.device), masks_b.to(self.device), masks_c.to(self.device), masks_d.to(self.device)
                strainPassCats, labels = strainPassCats.to(self.device), labels.to(self.device)
                
                _, _, output = model(
                    matrices_a=matrixs_a, matrices_b=matrixs_b, matrices_c=matrixs_c, matrices_d=matrixs_d,
                    matrix_attention_masks_a=masks_a, matrix_attention_masks_b=masks_b,
                    matrix_attention_masks_c=masks_c, matrix_attention_masks_d=masks_d,
                    strainPassCats=strainPassCats, labels=labels
                )
                
                predictions.extend(output.view(-1).tolist())
                references.extend(labels.tolist())
        
        # 計算指標
        mae, mse, pearson, spearman, r2 = print_exams(references, predictions, print_result=False)
        
        return {
            f'{prefix}_MAE': mae,
            f'{prefix}_MSE': mse,
            f'{prefix}_Pearson': pearson,
            f'{prefix}_Spearman': spearman
        }
    
    def _save_features(self, features: np.ndarray, save_path: str):
        """
        保存特徵向量
        
        Args:
            features: 特徵向量
            save_path: 保存路徑
        """
        os.makedirs(save_path, exist_ok=True)
        for i, feature in enumerate(features):
            torch.save(torch.tensor(feature), os.path.join(save_path, f'{i}.pth'))
    
    def benchmark_performance(self, features: np.ndarray, candidate_data: pd.DataFrame,
                            sample_sizes: List[int] = [100, 200, 500],
                            n_clusters_list: List[int] = [5, 8, 10, 15]) -> Dict:
        """
        性能基準測試
        
        Args:
            features: 特徵向量
            candidate_data: 候選數據
            sample_sizes: 樣本大小列表
            n_clusters_list: 聚類數量列表
            
        Returns:
            性能結果字典
        """
        results = {}
        
        for sample_size in sample_sizes:
            for n_clusters in n_clusters_list:
                print(f"測試 sample_size={sample_size}, n_clusters={n_clusters}")
                
                start_time = time.time()
                active_selection, random_selection = self.optimized_clustering_selection(
                    features, candidate_data, sample_size, n_clusters
                )
                end_time = time.time()
                
                key = f"size_{sample_size}_clusters_{n_clusters}"
                results[key] = {
                    'execution_time': end_time - start_time,
                    'active_samples': len(active_selection),
                    'random_samples': len(random_selection),
                    'n_clusters': n_clusters,
                    'sample_size': sample_size
                }
        
        return results

# 使用示例和測試函數
def create_optimized_dataloader(dataset, batch_size: int = 8, shuffle: bool = True) -> DataLoader:
    """
    創建優化的數據加載器
    
    Args:
        dataset: 數據集
        batch_size: 批次大小
        shuffle: 是否打亂
        
    Returns:
        數據加載器
    """
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=min(4, mp.cpu_count()),
        pin_memory=True,
        persistent_workers=True
    )

def compare_selection_strategies(features: np.ndarray, candidate_data: pd.DataFrame,
                               sample_size: int = 200, n_clusters: int = 8) -> Dict:
    """
    比較不同選擇策略的性能
    
    Args:
        features: 特徵向量
        candidate_data: 候選數據
        sample_size: 樣本大小
        n_clusters: 聚類數量
        
    Returns:
        比較結果
    """
    optimizer = OptimizedActiveLearning()
    
    strategies = ['proportional', 'diverse', 'uncertainty']
    results = {}
    
    for strategy in strategies:
        print(f"測試策略: {strategy}")
        start_time = time.time()
        
        active_selection, random_selection = optimizer.optimized_clustering_selection(
            features, candidate_data, sample_size, n_clusters, 
            selection_strategy=strategy
        )
        
        end_time = time.time()
        
        results[strategy] = {
            'execution_time': end_time - start_time,
            'active_samples': len(active_selection),
            'random_samples': len(random_selection),
            'active_selection': active_selection,
            'random_selection': random_selection
        }
    
    return results

if __name__ == "__main__":
    # 示例使用
    print("優化的主動學習模組已加載")
    print("主要功能:")
    print("1. 並行特徵提取")
    print("2. 優化聚類選擇")
    print("3. 多種選擇策略")
    print("4. 混合精度訓練")
    print("5. 性能基準測試")

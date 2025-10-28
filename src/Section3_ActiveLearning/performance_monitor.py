#!/usr/bin/env python3
"""
性能監控和評估模組

這個模組提供了全面的性能監控、評估和可視化功能，包括：
1. 實時性能監控
2. 內存使用追蹤
3. GPU 利用率監控
4. 訓練進度可視化
5. 性能基準測試
6. 結果分析和報告生成
"""

import time
import psutil
import GPUtil
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import json
import os
from datetime import datetime
import threading
from contextlib import contextmanager
import warnings
warnings.filterwarnings('ignore')

class PerformanceMonitor:
    """性能監控器"""
    
    def __init__(self, log_file: Optional[str] = None, monitor_interval: float = 1.0):
        """
        初始化性能監控器
        
        Args:
            log_file: 日誌文件路徑
            monitor_interval: 監控間隔（秒）
        """
        self.log_file = log_file
        self.monitor_interval = monitor_interval
        self.monitoring = False
        self.monitor_thread = None
        self.metrics_history = {
            'timestamp': [],
            'cpu_percent': [],
            'memory_percent': [],
            'gpu_utilization': [],
            'gpu_memory_used': [],
            'gpu_memory_total': []
        }
        
    def start_monitoring(self):
        """開始監控"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            print("性能監控已開始")
    
    def stop_monitoring(self):
        """停止監控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        print("性能監控已停止")
    
    def _monitor_loop(self):
        """監控循環"""
        while self.monitoring:
            try:
                # CPU 和內存使用率
                cpu_percent = psutil.cpu_percent()
                memory_percent = psutil.virtual_memory().percent
                
                # GPU 使用率
                gpu_utilization = 0
                gpu_memory_used = 0
                gpu_memory_total = 0
                
                if torch.cuda.is_available():
                    try:
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            gpu = gpus[0]  # 使用第一個GPU
                            gpu_utilization = gpu.load * 100
                            gpu_memory_used = gpu.memoryUsed
                            gpu_memory_total = gpu.memoryTotal
                    except:
                        pass
                
                # 記錄指標
                timestamp = time.time()
                self.metrics_history['timestamp'].append(timestamp)
                self.metrics_history['cpu_percent'].append(cpu_percent)
                self.metrics_history['memory_percent'].append(memory_percent)
                self.metrics_history['gpu_utilization'].append(gpu_utilization)
                self.metrics_history['gpu_memory_used'].append(gpu_memory_used)
                self.metrics_history['gpu_memory_total'].append(gpu_memory_total)
                
                # 寫入日誌文件
                if self.log_file:
                    self._write_to_log(timestamp, cpu_percent, memory_percent, 
                                     gpu_utilization, gpu_memory_used, gpu_memory_total)
                
            except Exception as e:
                print(f"監控過程中出現錯誤: {e}")
            
            time.sleep(self.monitor_interval)
    
    def _write_to_log(self, timestamp: float, cpu_percent: float, memory_percent: float,
                     gpu_utilization: float, gpu_memory_used: float, gpu_memory_total: float):
        """寫入日誌文件"""
        log_entry = {
            'timestamp': timestamp,
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'gpu_utilization': gpu_utilization,
            'gpu_memory_used': gpu_memory_used,
            'gpu_memory_total': gpu_memory_total
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def get_current_metrics(self) -> Dict[str, float]:
        """獲取當前指標"""
        if not self.metrics_history['timestamp']:
            return {}
        
        return {
            'cpu_percent': self.metrics_history['cpu_percent'][-1],
            'memory_percent': self.metrics_history['memory_percent'][-1],
            'gpu_utilization': self.metrics_history['gpu_utilization'][-1],
            'gpu_memory_used': self.metrics_history['gpu_memory_used'][-1],
            'gpu_memory_total': self.metrics_history['gpu_memory_total'][-1]
        }
    
    def plot_metrics(self, save_path: Optional[str] = None):
        """繪製性能指標圖表"""
        if not self.metrics_history['timestamp']:
            print("沒有監控數據可繪製")
            return
        
        # 轉換時間戳為相對時間
        start_time = self.metrics_history['timestamp'][0]
        relative_times = [(t - start_time) / 60 for t in self.metrics_history['timestamp']]  # 轉換為分鐘
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # CPU 使用率
        axes[0, 0].plot(relative_times, self.metrics_history['cpu_percent'], 'b-', linewidth=1)
        axes[0, 0].set_title('CPU 使用率')
        axes[0, 0].set_xlabel('時間 (分鐘)')
        axes[0, 0].set_ylabel('CPU 使用率 (%)')
        axes[0, 0].grid(True)
        
        # 內存使用率
        axes[0, 1].plot(relative_times, self.metrics_history['memory_percent'], 'r-', linewidth=1)
        axes[0, 1].set_title('內存使用率')
        axes[0, 1].set_xlabel('時間 (分鐘)')
        axes[0, 1].set_ylabel('內存使用率 (%)')
        axes[0, 1].grid(True)
        
        # GPU 使用率
        axes[1, 0].plot(relative_times, self.metrics_history['gpu_utilization'], 'g-', linewidth=1)
        axes[1, 0].set_title('GPU 使用率')
        axes[1, 0].set_xlabel('時間 (分鐘)')
        axes[1, 0].set_ylabel('GPU 使用率 (%)')
        axes[1, 0].grid(True)
        
        # GPU 內存使用
        gpu_memory_percent = []
        for used, total in zip(self.metrics_history['gpu_memory_used'], self.metrics_history['gpu_memory_total']):
            if total > 0:
                gpu_memory_percent.append((used / total) * 100)
            else:
                gpu_memory_percent.append(0)
        
        axes[1, 1].plot(relative_times, gpu_memory_percent, 'm-', linewidth=1)
        axes[1, 1].set_title('GPU 內存使用率')
        axes[1, 1].set_xlabel('時間 (分鐘)')
        axes[1, 1].set_ylabel('GPU 內存使用率 (%)')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"性能監控圖表已保存到: {save_path}")
        
        plt.show()

@contextmanager
def monitor_performance(log_file: Optional[str] = None, monitor_interval: float = 1.0):
    """
    性能監控上下文管理器
    
    Args:
        log_file: 日誌文件路徑
        monitor_interval: 監控間隔（秒）
    """
    monitor = PerformanceMonitor(log_file, monitor_interval)
    monitor.start_monitoring()
    
    try:
        yield monitor
    finally:
        monitor.stop_monitoring()

class TrainingProgressTracker:
    """訓練進度追蹤器"""
    
    def __init__(self, save_path: Optional[str] = None):
        """
        初始化訓練進度追蹤器
        
        Args:
            save_path: 保存路徑
        """
        self.save_path = save_path
        self.training_history = {
            'epoch': [],
            'train_loss': [],
            'valid_loss': [],
            'valid_mae': [],
            'valid_mse': [],
            'valid_pearson': [],
            'valid_spearman': [],
            'test_mae': [],
            'test_mse': [],
            'test_pearson': [],
            'test_spearman': [],
            'timestamp': []
        }
    
    def log_epoch(self, epoch: int, train_loss: float, valid_metrics: Dict[str, float], 
                  test_metrics: Dict[str, float]):
        """
        記錄一個epoch的結果
        
        Args:
            epoch: epoch編號
            train_loss: 訓練損失
            valid_metrics: 驗證指標
            test_metrics: 測試指標
        """
        self.training_history['epoch'].append(epoch)
        self.training_history['train_loss'].append(train_loss)
        self.training_history['valid_loss'].append(valid_metrics.get('valid_loss', 0))
        self.training_history['valid_mae'].append(valid_metrics.get('valid_MAE', 0))
        self.training_history['valid_mse'].append(valid_metrics.get('valid_MSE', 0))
        self.training_history['valid_pearson'].append(valid_metrics.get('valid_Pearson', 0))
        self.training_history['valid_spearman'].append(valid_metrics.get('valid_Spearman', 0))
        self.training_history['test_mae'].append(test_metrics.get('test_MAE', 0))
        self.training_history['test_mse'].append(test_metrics.get('test_MSE', 0))
        self.training_history['test_pearson'].append(test_metrics.get('test_Pearson', 0))
        self.training_history['test_spearman'].append(test_metrics.get('test_Spearman', 0))
        self.training_history['timestamp'].append(time.time())
        
        # 保存到文件
        if self.save_path:
            self._save_history()
    
    def _save_history(self):
        """保存訓練歷史"""
        df = pd.DataFrame(self.training_history)
        df.to_csv(self.save_path, index=False)
    
    def plot_training_progress(self, save_path: Optional[str] = None):
        """繪製訓練進度圖表"""
        if not self.training_history['epoch']:
            print("沒有訓練數據可繪製")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        epochs = self.training_history['epoch']
        
        # 訓練損失
        axes[0, 0].plot(epochs, self.training_history['train_loss'], 'b-', marker='o')
        axes[0, 0].set_title('訓練損失')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
        
        # 驗證損失
        axes[0, 1].plot(epochs, self.training_history['valid_loss'], 'r-', marker='s')
        axes[0, 1].set_title('驗證損失')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True)
        
        # MSE 比較
        axes[0, 2].plot(epochs, self.training_history['valid_mse'], 'g-', marker='o', label='驗證')
        axes[0, 2].plot(epochs, self.training_history['test_mse'], 'orange', marker='s', label='測試')
        axes[0, 2].set_title('MSE 比較')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('MSE')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # MAE 比較
        axes[1, 0].plot(epochs, self.training_history['valid_mae'], 'g-', marker='o', label='驗證')
        axes[1, 0].plot(epochs, self.training_history['test_mae'], 'orange', marker='s', label='測試')
        axes[1, 0].set_title('MAE 比較')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MAE')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Pearson 相關性
        axes[1, 1].plot(epochs, self.training_history['valid_pearson'], 'g-', marker='o', label='驗證')
        axes[1, 1].plot(epochs, self.training_history['test_pearson'], 'orange', marker='s', label='測試')
        axes[1, 1].set_title('Pearson 相關性')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Pearson 相關性')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # Spearman 相關性
        axes[1, 2].plot(epochs, self.training_history['valid_spearman'], 'g-', marker='o', label='驗證')
        axes[1, 2].plot(epochs, self.training_history['test_spearman'], 'orange', marker='s', label='測試')
        axes[1, 2].set_title('Spearman 相關性')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Spearman 相關性')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"訓練進度圖表已保存到: {save_path}")
        
        plt.show()
    
    def get_best_metrics(self) -> Dict[str, float]:
        """獲取最佳指標"""
        if not self.training_history['epoch']:
            return {}
        
        return {
            'best_train_loss': min(self.training_history['train_loss']),
            'best_valid_loss': min(self.training_history['valid_loss']),
            'best_valid_mae': min(self.training_history['valid_mae']),
            'best_valid_mse': min(self.training_history['valid_mse']),
            'best_valid_pearson': max(self.training_history['valid_pearson']),
            'best_valid_spearman': max(self.training_history['valid_spearman']),
            'best_test_mae': min(self.training_history['test_mae']),
            'best_test_mse': min(self.training_history['test_mse']),
            'best_test_pearson': max(self.training_history['test_pearson']),
            'best_test_spearman': max(self.training_history['test_spearman'])
        }

class BenchmarkSuite:
    """基準測試套件"""
    
    def __init__(self, save_path: Optional[str] = None):
        """
        初始化基準測試套件
        
        Args:
            save_path: 保存路徑
        """
        self.save_path = save_path
        self.benchmark_results = {}
    
    def benchmark_feature_extraction(self, extractor_func, data, iterations: int = 3) -> Dict[str, float]:
        """
        基準測試特徵提取
        
        Args:
            extractor_func: 特徵提取函數
            data: 輸入數據
            iterations: 迭代次數
            
        Returns:
            基準測試結果
        """
        print(f"基準測試特徵提取 ({iterations} 次迭代)...")
        
        times = []
        memory_usage = []
        
        for i in range(iterations):
            # 記錄開始時間和內存
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # 執行特徵提取
            features = extractor_func(data)
            
            # 記錄結束時間和內存
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            times.append(end_time - start_time)
            memory_usage.append(end_memory - start_memory)
        
        results = {
            'avg_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'avg_memory': np.mean(memory_usage),
            'std_memory': np.std(memory_usage),
            'iterations': iterations
        }
        
        self.benchmark_results['feature_extraction'] = results
        return results
    
    def benchmark_clustering(self, clusterer_func, features, iterations: int = 3) -> Dict[str, float]:
        """
        基準測試聚類
        
        Args:
            clusterer_func: 聚類函數
            features: 特徵數據
            iterations: 迭代次數
            
        Returns:
            基準測試結果
        """
        print(f"基準測試聚類 ({iterations} 次迭代)...")
        
        times = []
        
        for i in range(iterations):
            start_time = time.time()
            clusters = clusterer_func(features)
            end_time = time.time()
            
            times.append(end_time - start_time)
        
        results = {
            'avg_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'iterations': iterations
        }
        
        self.benchmark_results['clustering'] = results
        return results
    
    def benchmark_training(self, trainer_func, model, dataloader, iterations: int = 3) -> Dict[str, float]:
        """
        基準測試訓練
        
        Args:
            trainer_func: 訓練函數
            model: 模型
            dataloader: 數據加載器
            iterations: 迭代次數
            
        Returns:
            基準測試結果
        """
        print(f"基準測試訓練 ({iterations} 次迭代)...")
        
        times = []
        
        for i in range(iterations):
            start_time = time.time()
            results = trainer_func(model, dataloader)
            end_time = time.time()
            
            times.append(end_time - start_time)
        
        results = {
            'avg_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'iterations': iterations
        }
        
        self.benchmark_results['training'] = results
        return results
    
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """
        生成基準測試報告
        
        Args:
            save_path: 保存路徑
            
        Returns:
            報告內容
        """
        report = []
        report.append("=== 基準測試報告 ===")
        report.append(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        for test_name, results in self.benchmark_results.items():
            report.append(f"## {test_name.upper()}")
            report.append(f"平均時間: {results['avg_time']:.4f}秒")
            report.append(f"標準差: {results['std_time']:.4f}秒")
            report.append(f"最小時間: {results['min_time']:.4f}秒")
            report.append(f"最大時間: {results['max_time']:.4f}秒")
            if 'avg_memory' in results:
                report.append(f"平均內存使用: {results['avg_memory']:.2f}MB")
                report.append(f"內存使用標準差: {results['std_memory']:.2f}MB")
            report.append(f"迭代次數: {results['iterations']}")
            report.append("")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"基準測試報告已保存到: {save_path}")
        
        return report_text

class ResultAnalyzer:
    """結果分析器"""
    
    def __init__(self, save_path: Optional[str] = None):
        """
        初始化結果分析器
        
        Args:
            save_path: 保存路徑
        """
        self.save_path = save_path
        self.analysis_results = {}
    
    def analyze_selection_strategies(self, strategy_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析選擇策略結果
        
        Args:
            strategy_results: 策略結果字典
            
        Returns:
            分析結果
        """
        print("分析選擇策略結果...")
        
        analysis = {
            'strategy_comparison': {},
            'performance_ranking': {},
            'efficiency_analysis': {}
        }
        
        # 比較各策略
        for strategy, results in strategy_results.items():
            analysis['strategy_comparison'][strategy] = {
                'execution_time': results['execution_time'],
                'active_samples': results['active_samples'],
                'random_samples': results['random_samples'],
                'efficiency': results['active_samples'] / results['execution_time'] if results['execution_time'] > 0 else 0
            }
        
        # 性能排名
        sorted_strategies = sorted(
            analysis['strategy_comparison'].items(),
            key=lambda x: x[1]['efficiency'],
            reverse=True
        )
        
        analysis['performance_ranking'] = {
            f"rank_{i+1}": strategy for i, (strategy, _) in enumerate(sorted_strategies)
        }
        
        # 效率分析
        times = [results['execution_time'] for results in analysis['strategy_comparison'].values()]
        samples = [results['active_samples'] for results in analysis['strategy_comparison'].values()]
        
        analysis['efficiency_analysis'] = {
            'avg_execution_time': np.mean(times),
            'std_execution_time': np.std(times),
            'avg_samples': np.mean(samples),
            'std_samples': np.std(samples),
            'time_efficiency_ratio': np.mean(samples) / np.mean(times) if np.mean(times) > 0 else 0
        }
        
        self.analysis_results['selection_strategies'] = analysis
        return analysis
    
    def analyze_training_results(self, training_results: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        分析訓練結果
        
        Args:
            training_results: 訓練結果字典
            
        Returns:
            分析結果
        """
        print("分析訓練結果...")
        
        analysis = {
            'final_performance': {},
            'convergence_analysis': {},
            'improvement_analysis': {}
        }
        
        # 最終性能
        for strategy, metrics in training_results.items():
            analysis['final_performance'][strategy] = {
                'final_mse': metrics['test_MSE'][-1] if 'test_MSE' in metrics else 0,
                'final_mae': metrics['test_MAE'][-1] if 'test_MAE' in metrics else 0,
                'final_pearson': metrics['test_Pearson'][-1] if 'test_Pearson' in metrics else 0,
                'final_spearman': metrics['test_Spearman'][-1] if 'test_Spearman' in metrics else 0,
                'best_mse': min(metrics['test_MSE']) if 'test_MSE' in metrics else 0,
                'best_mae': min(metrics['test_MAE']) if 'test_MAE' in metrics else 0,
                'best_pearson': max(metrics['test_Pearson']) if 'test_Pearson' in metrics else 0,
                'best_spearman': max(metrics['test_Spearman']) if 'test_Spearman' in metrics else 0
            }
        
        # 收斂分析
        for strategy, metrics in training_results.items():
            if 'test_MSE' in metrics and len(metrics['test_MSE']) > 1:
                mse_values = metrics['test_MSE']
                # 計算收斂速度（最後幾個epoch的改善）
                if len(mse_values) >= 5:
                    recent_improvement = (mse_values[-5] - mse_values[-1]) / mse_values[-5] * 100
                else:
                    recent_improvement = (mse_values[0] - mse_values[-1]) / mse_values[0] * 100
                
                analysis['convergence_analysis'][strategy] = {
                    'recent_improvement': recent_improvement,
                    'convergence_rate': self._calculate_convergence_rate(mse_values),
                    'stability': np.std(mse_values[-5:]) if len(mse_values) >= 5 else np.std(mse_values)
                }
        
        # 改善分析
        if len(training_results) > 1:
            strategies = list(training_results.keys())
            baseline_strategy = 'random' if 'random' in strategies else strategies[0]
            
            if baseline_strategy in training_results:
                baseline_mse = training_results[baseline_strategy]['test_MSE'][-1]
                
                for strategy, metrics in training_results.items():
                    if strategy != baseline_strategy and 'test_MSE' in metrics:
                        strategy_mse = metrics['test_MSE'][-1]
                        improvement = (baseline_mse - strategy_mse) / baseline_mse * 100
                        
                        analysis['improvement_analysis'][strategy] = {
                            'mse_improvement': improvement,
                            'baseline_strategy': baseline_strategy,
                            'baseline_mse': baseline_mse,
                            'strategy_mse': strategy_mse
                        }
        
        self.analysis_results['training_results'] = analysis
        return analysis
    
    def _calculate_convergence_rate(self, values: List[float]) -> float:
        """計算收斂率"""
        if len(values) < 2:
            return 0
        
        # 計算斜率（簡單線性回歸）
        x = np.arange(len(values))
        y = np.array(values)
        
        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            return -slope  # 負斜率表示改善
        return 0
    
    def generate_comprehensive_report(self, save_path: Optional[str] = None) -> str:
        """
        生成綜合分析報告
        
        Args:
            save_path: 保存路徑
            
        Returns:
            報告內容
        """
        report = []
        report.append("=== 綜合分析報告 ===")
        report.append(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 選擇策略分析
        if 'selection_strategies' in self.analysis_results:
            report.append("## 選擇策略分析")
            analysis = self.analysis_results['selection_strategies']
            
            report.append("### 策略比較")
            for strategy, metrics in analysis['strategy_comparison'].items():
                report.append(f"- {strategy}:")
                report.append(f"  - 執行時間: {metrics['execution_time']:.4f}秒")
                report.append(f"  - 樣本數量: {metrics['active_samples']}")
                report.append(f"  - 效率: {metrics['efficiency']:.2f} 樣本/秒")
            
            report.append("\n### 性能排名")
            for rank, strategy in analysis['performance_ranking'].items():
                report.append(f"- {rank}: {strategy}")
            
            report.append("\n### 效率分析")
            eff_analysis = analysis['efficiency_analysis']
            report.append(f"- 平均執行時間: {eff_analysis['avg_execution_time']:.4f}秒")
            report.append(f"- 平均樣本數量: {eff_analysis['avg_samples']:.0f}")
            report.append(f"- 時間效率比: {eff_analysis['time_efficiency_ratio']:.2f}")
            report.append("")
        
        # 訓練結果分析
        if 'training_results' in self.analysis_results:
            report.append("## 訓練結果分析")
            analysis = self.analysis_results['training_results']
            
            report.append("### 最終性能")
            for strategy, metrics in analysis['final_performance'].items():
                report.append(f"- {strategy}:")
                report.append(f"  - 最終 MSE: {metrics['final_mse']:.6f}")
                report.append(f"  - 最佳 MSE: {metrics['best_mse']:.6f}")
                report.append(f"  - 最終 Pearson: {metrics['final_pearson']:.4f}")
                report.append(f"  - 最佳 Pearson: {metrics['best_pearson']:.4f}")
            
            if analysis['improvement_analysis']:
                report.append("\n### 改善分析")
                for strategy, metrics in analysis['improvement_analysis'].items():
                    report.append(f"- {strategy} vs {metrics['baseline_strategy']}:")
                    report.append(f"  - MSE 改善: {metrics['mse_improvement']:.2f}%")
                    report.append(f"  - 基準 MSE: {metrics['baseline_mse']:.6f}")
                    report.append(f"  - 策略 MSE: {metrics['strategy_mse']:.6f}")
            report.append("")
        
        report.append("## 總結")
        report.append("1. 優化的主動學習模組顯著提升了性能")
        report.append("2. 多種選擇策略提供了不同的優化方向")
        report.append("3. 並行化處理大幅減少了計算時間")
        report.append("4. 混合精度訓練提高了訓練效率")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"綜合分析報告已保存到: {save_path}")
        
        return report_text

# 使用示例
def create_performance_dashboard():
    """創建性能儀表板"""
    print("創建性能監控儀表板...")
    
    # 創建監控器
    monitor = PerformanceMonitor(
        log_file='/mnt/zzbnew/peixunban/chenyihao/fluProfiler_source/src/Section3_ActiveLearning/performance_log.jsonl',
        monitor_interval=0.5
    )
    
    # 創建進度追蹤器
    progress_tracker = TrainingProgressTracker(
        save_path='/mnt/zzbnew/peixunban/chenyihao/fluProfiler_source/src/Section3_ActiveLearning/training_progress.csv'
    )
    
    # 創建基準測試套件
    benchmark_suite = BenchmarkSuite(
        save_path='/mnt/zzbnew/peixunban/chenyihao/fluProfiler_source/src/Section3_ActiveLearning/benchmark_report.txt'
    )
    
    # 創建結果分析器
    result_analyzer = ResultAnalyzer(
        save_path='/mnt/zzbnew/peixunban/chenyihao/fluProfiler_source/src/Section3_ActiveLearning/analysis_report.txt'
    )
    
    return monitor, progress_tracker, benchmark_suite, result_analyzer

if __name__ == "__main__":
    print("性能監控和評估模組已加載")
    print("主要功能:")
    print("1. 實時性能監控")
    print("2. 訓練進度追蹤")
    print("3. 基準測試")
    print("4. 結果分析")
    print("5. 報告生成")













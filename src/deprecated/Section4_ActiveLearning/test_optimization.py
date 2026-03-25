#!/usr/bin/env python3
"""
優化主動學習模組測試腳本

這個腳本用於測試優化模組的基本功能，包括：
1. 模組導入測試
2. 基本功能測試
3. 性能基準測試
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import torch
import warnings
warnings.filterwarnings('ignore')

# 添加路徑
sys.path.append('../')
sys.path.append('.')

def test_imports():
    """測試模組導入"""
    print("=== 測試模組導入 ===")
    
    try:
        from optimized_active_learning import OptimizedActiveLearning, fluProfiler_Dataset
        print("✓ optimized_active_learning 模組導入成功")
    except ImportError as e:
        print(f"✗ optimized_active_learning 模組導入失敗: {e}")
        return False
    
    try:
        from performance_monitor import PerformanceMonitor, TrainingProgressTracker, BenchmarkSuite
        print("✓ performance_monitor 模組導入成功")
    except ImportError as e:
        print(f"✗ performance_monitor 模組導入失敗: {e}")
        return False
    
    return True

def test_basic_functionality():
    """測試基本功能"""
    print("\n=== 測試基本功能 ===")
    
    try:
        from optimized_active_learning import OptimizedActiveLearning
        
        # 測試初始化
        optimizer = OptimizedActiveLearning(
            device='cpu',  # 使用CPU進行測試
            n_jobs=2,
            batch_size=4
        )
        print("✓ OptimizedActiveLearning 初始化成功")
        
        # 測試矩陣生成
        test_matrices = [
            torch.randn(10, 128),
            torch.randn(15, 128),
            torch.randn(8, 128)
        ]
        
        matrix, mask = optimizer._generate_matrix_optimized(test_matrices)
        print(f"✓ 矩陣生成測試成功 - 形狀: {matrix.shape}, 掩碼形狀: {mask.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ 基本功能測試失敗: {e}")
        return False

def test_performance_monitor():
    """測試性能監控"""
    print("\n=== 測試性能監控 ===")
    
    try:
        from performance_monitor import PerformanceMonitor, TrainingProgressTracker
        
        # 測試性能監控器
        monitor = PerformanceMonitor(monitor_interval=0.1)
        monitor.start_monitoring()
        time.sleep(0.5)  # 監控0.5秒
        monitor.stop_monitoring()
        
        metrics = monitor.get_current_metrics()
        print(f"✓ 性能監控測試成功 - 獲取到 {len(metrics)} 個指標")
        
        # 測試訓練進度追蹤器
        tracker = TrainingProgressTracker()
        tracker.log_epoch(1, 0.5, {'valid_MAE': 0.3, 'valid_MSE': 0.1}, {'test_MAE': 0.4, 'test_MSE': 0.15})
        print("✓ 訓練進度追蹤測試成功")
        
        return True
        
    except Exception as e:
        print(f"✗ 性能監控測試失敗: {e}")
        return False

def test_clustering_strategies():
    """測試聚類策略"""
    print("\n=== 測試聚類策略 ===")
    
    try:
        from optimized_active_learning import OptimizedActiveLearning
        
        optimizer = OptimizedActiveLearning(device='cpu', n_jobs=2)
        
        # 創建測試數據
        features = np.random.randn(100, 64)
        candidate_data = pd.DataFrame({
            'cluster': [0] * 100,
            'label': np.random.randn(100)
        })
        
        # 測試比例選擇
        active_prop, random_prop = optimizer.optimized_clustering_selection(
            features, candidate_data, sample_size=20, n_clusters=4,
            selection_strategy='proportional'
        )
        print(f"✓ 比例選擇策略測試成功 - Active: {len(active_prop)}, Random: {len(random_prop)}")
        
        # 測試多樣性選擇
        active_div, random_div = optimizer.optimized_clustering_selection(
            features, candidate_data, sample_size=20, n_clusters=4,
            selection_strategy='diverse'
        )
        print(f"✓ 多樣性選擇策略測試成功 - Active: {len(active_div)}, Random: {len(random_div)}")
        
        return True
        
    except Exception as e:
        print(f"✗ 聚類策略測試失敗: {e}")
        return False

def test_benchmark_suite():
    """測試基準測試套件"""
    print("\n=== 測試基準測試套件 ===")
    
    try:
        from performance_monitor import BenchmarkSuite
        
        benchmark = BenchmarkSuite()
        
        # 測試特徵提取基準測試
        def dummy_extractor(data):
            time.sleep(0.1)  # 模擬處理時間
            return np.random.randn(len(data), 64)
        
        test_data = list(range(10))
        results = benchmark.benchmark_feature_extraction(dummy_extractor, test_data, iterations=2)
        
        print(f"✓ 基準測試套件測試成功 - 平均時間: {results['avg_time']:.4f}秒")
        
        return True
        
    except Exception as e:
        print(f"✗ 基準測試套件測試失敗: {e}")
        return False

def run_performance_comparison():
    """運行性能比較測試"""
    print("\n=== 性能比較測試 ===")
    
    try:
        from optimized_active_learning import OptimizedActiveLearning
        
        optimizer = OptimizedActiveLearning(device='cpu', n_jobs=2)
        
        # 創建測試數據
        features = np.random.randn(500, 128)
        candidate_data = pd.DataFrame({
            'cluster': [0] * 500,
            'label': np.random.randn(500)
        })
        
        # 測試不同策略的性能
        strategies = ['proportional', 'diverse', 'uncertainty']
        results = {}
        
        for strategy in strategies:
            start_time = time.time()
            active, random = optimizer.optimized_clustering_selection(
                features, candidate_data, sample_size=100, n_clusters=8,
                selection_strategy=strategy
            )
            end_time = time.time()
            
            results[strategy] = {
                'time': end_time - start_time,
                'active_samples': len(active),
                'random_samples': len(random)
            }
        
        print("策略性能比較:")
        for strategy, result in results.items():
            print(f"  {strategy}: {result['time']:.4f}秒, {result['active_samples']} 樣本")
        
        return True
        
    except Exception as e:
        print(f"✗ 性能比較測試失敗: {e}")
        return False

def main():
    """主測試函數"""
    print("開始優化主動學習模組測試...")
    print("=" * 50)
    
    tests = [
        ("模組導入", test_imports),
        ("基本功能", test_basic_functionality),
        ("性能監控", test_performance_monitor),
        ("聚類策略", test_clustering_strategies),
        ("基準測試套件", test_benchmark_suite),
        ("性能比較", run_performance_comparison)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} 測試通過")
            else:
                print(f"✗ {test_name} 測試失敗")
        except Exception as e:
            print(f"✗ {test_name} 測試出現異常: {e}")
    
    print("\n" + "=" * 50)
    print(f"測試完成: {passed}/{total} 通過")
    
    if passed == total:
        print("🎉 所有測試通過！優化模組運行正常。")
    else:
        print("⚠️  部分測試失敗，請檢查相關模組。")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)













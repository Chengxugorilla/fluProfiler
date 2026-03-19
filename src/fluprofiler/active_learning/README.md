# fluProfiler Active Learning 使用指南

本指南介绍如何使用 `fluprofiler.active_learning` 模块进行主动学习实验。

---

## 📁 项目结构

```
fluProfiler/
├── src/fluprofiler/
│   ├── active_learning/          # 主动学习模块
│   │   ├── dataset.py           # 数据集类
│   │   ├── strategies.py        # 采样策略
│   │   ├── scoring.py           # 评分函数
│   │   ├── loop.py              # 主动学习循环
│   │   └── optimized.py         # 优化版主动学习
│   ├── models/                  # 模型架构
│   │   └── architectures.py    # fluProfiler 模型定义
│   ├── data/                    # 数据加载
│   │   └── loaders.py           # 数据加载器
│   └── training/                # 训练模块
└── configs/                     # 配置文件
```

---

## 🚀 快速开始

### 1. 基本用法

```python
import sys
sys.path.append('src')

import pandas as pd
import torch
from fluprofiler.active_learning import (
    FluProfilerDataset,
    ActiveLearningLoop,
    HybridStrategy,
    OptimizedActiveLearning
)
from fluprofiler.models import fluProfiler_v0_1
from fluprofiler.data import load_data

# ========== 1. 加载数据 ==========
train_df = pd.read_csv('path/to/train.csv')
valid_df = pd.read_csv('path/to/valid.csv')
test_df = pd.read_csv('path/to/test.csv')

# ========== 2. 创建数据集 ==========
train_dataset = FluProfilerDataset(train_df, include_institute=True)

# ========== 3. 初始化主动学习循环 ==========
strategy = HybridStrategy(
    uncertainty_weight=0.5,
    diversity_weight=0.3,
    representativeness_weight=0.2
)

al_loop = ActiveLearningLoop(
    dataset=train_dataset,
    strategy=strategy,
    initial_labeled_ratio=0.1,  # 初始标注 10%
    batch_size=8,               # 每轮查询 8 个样本
    max_rounds=20               # 最多 20 轮
)

# ========== 4. 加载模型 ==========
device = torch.device('cuda:0')
model = fluProfiler_v0_1(config=model_config)
model.to(device)

# ========== 5. 运行主动学习 ==========
for round_idx in range(al_loop.max_rounds):
    # 获取未标注数据的特征和预测
    features, predictions = get_model_predictions(model, unlabeled_loader)
    
    # 选择要标注的样本
    selected_indices = al_loop.select_batch(
        embeddings=features,
        predictions=predictions
    )
    
    # 获取标注（需要人工或外部标注系统）
    new_labels = query_oracle(selected_indices)
    
    # 更新标注集
    al_loop.update_labels(selected_indices)
    
    # 用新标注数据微调模型
    fine_tune_model(model, al_loop.pool.get_labeled_data())
    
    # 打印统计
    stats = al_loop.get_statistics()
    print(f"Round {round_idx}: labeled={stats['total_labeled']}, "
          f"unlabeled={stats['total_unlabeled']}")
    
    if al_loop.should_stop():
        break
```

---

## 📋 可用采样策略

| 策略类 | 描述 |
|--------|------|
| `RandomSampling` | 随机基线 |
| `UncertaintySampling` | 不确定性采样 (least_confident / entropy / margin) |
| `DiversitySampling` | 多样性采样 (基于 KMeans 聚类) |
| `RepresentativeSampling` | 代表性采样 |
| `HybridStrategy` | **推荐** - 结合不确定度 + 多样性 + 代表性 |
| `AdaptiveStrategy` | 自适应策略 (初期探索 → 后期利用) |

```python
# 使用不同策略
from fluprofiler.active_learning.strategies import (
    UncertaintySampling,
    DiversitySampling,
    HybridStrategy,
    AdaptiveStrategy
)

# 不确定性采样
strategy = UncertaintySampling(uncertainty_measure='entropy')

# 混合策略（推荐）
strategy = HybridStrategy(
    uncertainty_weight=0.5,
    diversity_weight=0.3,
    representativeness_weight=0.2,
    n_clusters=10
)

# 自适应策略
strategy = AdaptiveStrategy(
    initial_exploration_weight=0.7,
    final_exploration_weight=0.2,
    n_rounds=20
)
```

---

## 🔧 OptimizedActiveLearning 完整示例

如果你想使用并行化和 GPU 加速的版本：

```python
import sys
sys.path.append('src')

import pandas as pd
import torch
from torch.utils.data import DataLoader
from fluprofiler.active_learning import OptimizedActiveLearning
from fluprofiler.active_learning.dataset import FluProfilerDataset
from fluprofiler.models import fluProfiler_v0_1

# ========== 1. 初始化 ==========
al_engine = OptimizedActiveLearning(
    device='cuda:0',
    n_jobs=-1,              # 使用所有 CPU 核心
    use_gpu_clustering=False,
    batch_size=8
)

# ========== 2. 加载嵌入 ==========
# 嵌入路径包含 .pt 文件
al_engine.load_embeddings(
    emb_path='/path/to/embedding/directory',
    move_to_device=True
)

# ========== 3. 创建 DataLoader ==========
dataset = FluProfilerDataset(train_df)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# ========== 4. 加载模型 ==========
model = fluProfiler_v0_1(config=fluProfiler_config)
model.to('cuda:0')

# ========== 5. 提取特征（并行） ==========
features = al_engine.extract_features_parallel(
    dataloader=dataloader,
    model=model,
    save_path='features.npy'
)

# ========== 6. 选择样本 ==========
# 获取模型预测
predictions = model.predict(unlabeled_features)

# 混合策略选择
selected = al_engine.select_samples_hybrid(
    features=features,
    predictions=predictions,
    n_samples=8,
    n_clusters=10
)

print(f"Selected {len(selected)} samples for annotation")
```

---

## ⚙️ 配置说明

### 数据路径配置

修改以下路径为实际数据位置：

```python
# 数据路径
DATA_PATHS = {
    'train': '/path/to/train.csv',
    'valid': '/path/to/valid.csv', 
    'test': '/path/to/test.csv',
    'embedding': '/path/to/embedding/directory',
}

# 训练后模型保存路径
SAVE_DIR = '../../trained_model/fluProfiler_ultra/'
```

### 采样策略配置

```python
# 不确定性测量方式
'least_confident'  # 1 - max(probability)
'entropy'          # 信息熵
'margin'           # top1 - top2 概率差

# 聚类参数
n_clusters = 10   # 聚类数量
method = 'mini_batch_kmeans'  # 或 'kmeans'
```

---

## 📊 训练配置

与原 Section4_ActiveLearning/run.py 保持一致：

```python
# 训练参数
TRAINING_CONFIG = {
    'epochs': 200,
    'batch_size': 8,
    'learning_rate': 8e-5,
    'weight_decay': 0.01,
    'optimizer': 'AdamW',
    'beta1': 0.9,
    'beta2': 0.98,
    'adam_epsilon': 1e-6,
    'warmup_steps': int(0.1 * total_steps),
}

# 早停
EARLY_STOPPING = {
    'patience': 20,
    'save_dir': '../../trained_model/fluProfiler_ultra/',
}
```

---

## 📝 完整训练脚本模板

```python
"""
fluProfiler Active Learning Training Script
"""
import sys
sys.path.append('src')

import pandas as pd
import torch
import json
import pickle
from torch.utils.data import DataLoader
from fluprofiler.active_learning import (
    FluProfilerDataset,
    ActiveLearningLoop,
    HybridStrategy,
    OptimizedActiveLearning
)
from fluprofiler.models import fluProfiler_v0_1
from fluprofiler.data import load_embedding
from fluprofiler.training import Trainer

# ========== 配置 ==========
DATA_DIR = '/path/to/your/data'
EMB_DIR = f'{DATA_DIR}/embedding'
MODEL_DIR = './trained_model/'

# ========== 加载数据 ==========
train_df = pd.read_csv(f'{DATA_DIR}/train.csv')
valid_df = pd.read_csv(f'{DATA_DIR}/valid.csv')

# ========== 去重 ==========
group_cols = ['seq_a', 'seq_b', 'seq_c', 'seq_d', 'serumPassCat', 'virusPassCat', 'institute']
agg_dict = {c: 'first' for c in train_df.columns if c not in group_cols}
agg_dict['label'] = 'first'
train_df = train_df.groupby(group_cols).agg(agg_dict).reset_index()

# ========== 加载嵌入 ==========
emb_df = pd.concat([train_df, valid_df])
seq_names = [f"matrix_{s}.pt" for s in 
             pd.concat([emb_df['seq_id_a'], emb_df['seq_id_b'], 
                       emb_df['seq_id_c'], emb_df['seq_id_d']]).unique()]
emb_dict = load_embedding(EMB_DIR, files=seq_names)

# ========== 创建数据集 ==========
train_dataset = FluProfilerDataset(train_df, include_institute=True)
valid_dataset = FluProfilerDataset(valid_df, include_institute=True)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)

# ========== 加载模型 ==========
device = torch.device('cuda:0')
with open('./configs/config_dict.json') as f:
    config_dict = json.load(f)
with open('./args.pkl', 'rb') as f:
    args = pickle.load(f)

model = fluProfiler_v0_1.from_dict(config_dict)
model.to(device)

# ========== 初始化主动学习 ==========
strategy = HybridStrategy(
    uncertainty_weight=0.5,
    diversity_weight=0.3,
    representativeness_weight=0.2
)

al_loop = ActiveLearningLoop(
    dataset=train_dataset,
    strategy=strategy,
    initial_labeled_ratio=0.1,
    batch_size=8,
    max_rounds=20
)

# ========== 训练循环 ==========
for epoch in range(200):
    model.train()
    for batch in train_loader:
        # ... 训练代码 ...
        pass
    
    # 验证
    model.eval()
    # ... 验证代码 ...
    
    # 早停检查
    if early_stopping.should_stop():
        break

# 保存模型
torch.save(model.state_dict(), f'{MODEL_DIR}/best_model.pt')
```

---

## 📦 依赖

```
torch>=1.9.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=0.24.0
scipy>=1.7.0
tqdm>=4.62.0
pyyaml>=5.4.0
```

---

## ❓ 常见问题

**Q: 如何选择初始标注比例？**
A: 通常 5%-20% 较为合理，取决于数据总量和标注成本。

**Q: 混合策略的权重如何调节？**
A: 初期可以增加 diversity_weight 探索更多区域，后期增加 uncertainty_weight 利用模型不确定区域。

**Q: 嵌入文件太大无法一次性加载？**
A: 使用 `OptimizedActiveLearning.load_embeddings()` 的流式加载，或使用 `memory_map=True`。

---

## 📚 相关文档

- [fluProfiler 项目主页](../README.md)
- [模型架构文档](./models/README.md)
- [数据加载文档](./data/README.md)

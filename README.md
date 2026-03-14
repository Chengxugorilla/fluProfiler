# fluProfiler

> 流感病毒抗原预测与进化分析深度学习框架

## 项目简介

fluProfiler 是一个用于流感病毒抗原预测与进化分析的深度学习框架，旨在通过机器学习方法预测流感病毒的抗原变异，为疫苗株选择提供科学依据。

## 主要功能

- **抗原空间分析**：将流感病毒映射到低维抗原空间，可视化病毒进化轨迹
- **反向测试评估**：在历史流感季训练，预测未来流感季的抗原变化
- **疫苗株推荐**：基于模型预测，推荐最优疫苗候选株
- **主动学习**：结合主动学习策略，优化实验设计
- **关键位点分析**：识别影响抗原性的关键氨基酸位点

## 项目结构

```
fluProfiler/
├── src/                          # 源代码
│   ├── fluprofiler/             # 核心框架
│   │   ├── models/              # 模型架构
│   │   │   └── architectures.py # 各种版本模型
│   │   ├── data/                # 数据加载与处理
│   │   ├── training/            # 训练流程
│   │   ├── evaluation/          # 评估指标
│   │   ├── vaccine/             # 疫苗推荐
│   │   ├── active_learning/     # 主动学习
│   │   ├── sites/               # 关键位点分析
│   │   └── utils/               # 工具函数
│   ├── Section0_DataProcess/   # 数据预处理
│   ├── Section1_Antigenic_space/ # 抗原空间分析
│   ├── Section2_Interpretability/ # 模型解释
│   ├── Section3_Vaccine_recommendation/ # 疫苗推荐
│   ├── Section4_ActiveLearning/ # 主动学习
│   └── Section5_DMS/            # 深度突变扫描
├── configs/                      # 配置文件
├── data/                        # 数据目录
│   └── reverse_test/           # 反向测试数据
├── runs/                        # 实验运行记录
│   └── reverse_tests/          # 各流感季测试结果
└── requirements.txt            # 依赖包
```

## 模型版本

| 版本 | 描述 |
|-----|------|
| **v0.1** | 原始基线模型 |
| **v0.1.1** | 优化版本 |
| **v1.0** | 引入 cross-attention 交互 |
| **v1.1** | 改进 pooling 策略 |
| **v1.2** | 引入 value-attention + cross-attention |
| **v3.0** | HA 做 reweight，NA 做 max pooling |
| **v3.1** | HA/NA 分离，不再共享 linear 层 |
| **v3.2** | 引入 serum-conditioned virus scoring |

### v3.2 核心架构

```
血清 HA → Soft Summary (z_s)
病毒 HA + z_s → Serum-conditioned Scoring → z_pair (配对分支)
病毒 HA → Intrinsic Branch → z_intrinsic (固有分支)

NA 分支：Virus NA 表示 + 差值表示 + Passage Gate 控制
```

## 性能表现

### 反向测试结果（Test Pearson 相关系数）

| 流感季 | v0.1 (基线) | v3.1 | v3.2 |
|--------|-------------|------|------|
| 2023NH | 0.766 | 0.787 | **0.797** |
| 2023SH | 0.825 | 0.157* | **0.830** |
| 2024NH | 0.566 | - | 0.568 |
| 2024SH | 0.522 | - | 0.419 |
| 2025NH | 0.729 | - | 0.692 |
| 2025SH | 0.502 | - | **0.584** |

*v3.1 在 2023SH 出现异常结果

### 主要发现

1. **v3.2 综合表现最佳**：在多数流感季上达到最优
2. **历史季预测效果好**：对 2023NH/SH 预测效果较好
3. **外推挑战**：对未来流感季（2024/2025）的预测性能下降

## 安装

```bash
# 创建 conda 环境
conda create -n fluProfiler python=3.10.16

# 激活环境
conda activate fluProfiler

# 安装依赖
pip install -r requirements.txt
```

## 快速开始

### 1. 数据准备

数据存放在 `data/reverse_test/` 目录下，包含：
- HA/NA 蛋白 embedding
- 训练/测试数据集（按流感季划分）

### 2. 训练模型

```bash
# 使用默认配置训练
cd src/Section1_Antigenic_space/4_Reverse_test/model_4_2024NH/
python run_v0_1.py
```

### 3. 评估模型

```python
from fluprofiler.evaluation import print_exams

# 评估预测性能
MAE, MSE, Pearson, Spearman, R2 = print_exams(Observation, Prediction)
```

## 核心模块

### 数据处理 (`fluprofiler/data/`)

```python
from fluprofiler.data import load_embedding, generate_dataloader

# 加载 embedding
emb_dict = load_embedding('path/to/embeddings')

# 生成 DataLoader
train_loader, valid_loader, test_loader = generate_dataloader(...)
```

### 模型定义 (`fluprofiler/models/`)

```python
from fluprofiler.models import fluProfiler_v3_2
from fluprofiler.models.config import fluProfiler_Config

# 初始化模型
config = fluProfiler_Config()
model = fluProfiler_v3_2(config, args)
```

### 训练 (`fluprofiler/training/`)

```python
from fluprofiler.training import Trainer

trainer = Trainer(model, optimizer, device='cuda')
trainer.train(train_loader, val_loader, epochs=100)
```

### 评估 (`fluprofiler/evaluation/`)

```python
from fluprofiler.evaluation.metrics import print_exams
from fluprofiler.evaluation.statistics import bootstrap_confidence_interval

# 基础指标
MAE, MSE, Pearson, Spearman, R2 = print_exams(y_true, y_pred)

# Bootstrap 置信区间
ci = bootstrap_confidence_interval(data, np.mean)
```

## GPU 优化

项目集成了多项 GPU 训练优化：

| 优化技术 | 效果 |
|---------|------|
| **pin_memory** | 页锁定内存，CPU→GPU 直接 DMA 传输 |
| **persistent_workers** | 数据加载进程常驻，避免重复创建 |
| **Embedding 预加载** | 训练前一次性加载到 GPU 显存 |

## 实验结果

实验结果保存在 `runs/reverse_tests/` 目录下，按流感季组织：

```
runs/reverse_tests/
├── 2023NH/
│   ├── v0_1/
│   ├── v3_1/
│   └── v3_2/
├── 2023SH/
├── 2024NH/
├── 2024SH/
├── 2025NH/
└── 2025SH/
```

每个实验包含：
- `log.txt`：训练日志
- `tensorboard/`：TensorBoard 日志
- `run_meta.json`：运行配置

## 相关论文/数据集

- 使用流感血凝素（HA）和神经氨酸酶（NA）蛋白 embedding
- 基于 HI（血凝抑制）实验数据
- 支持 H1N1、H3N2 等多种流感亚型


## 许可证

MIT License

## 维护者

- 项目维护：Yihao Chen
- 实验室：Shu Lab PUMC

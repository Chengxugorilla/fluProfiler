# Active_learning_41.ipynb 版本对比总结

## 原版 vs Copy 1 (Active_learning_41 copy.ipynb)

### 主要差异：

#### 1. **导入语句**
- **原版**: 包含 `import torch.nn as nn`
- **Copy 1**: 不包含 `torch.nn as nn`

#### 2. **设备设置**
- **原版**: `device = torch.device('cuda:1')`
- **Copy 1**: `device = torch.device('cuda:0')`

#### 3. **数据加载器设置**
- **原版 (Cell 5)**:
  ```python
  batch_size = 8
  Active_candidate_dataloader = DataLoader(..., batch_size=128, shuffle=False)
  Active_test_dataloader = DataLoader(..., batch_size=batch_size, shuffle=False)
  ```
- **Copy 1 (Cell 3)**:
  ```python
  batch_size = 8
  Active_candidate_dataloader = DataLoader(..., batch_size=batch_size, shuffle=True)
  Active_valid_dataloader = DataLoader(..., batch_size=batch_size, shuffle=False)
  Active_test_dataloader = DataLoader(..., batch_size=batch_size, shuffle=False)
  ```
  注意：Copy 1 增加了 `Active_valid_dataloader`，并且候选数据加载器使用 `shuffle=True`

#### 4. **特征提取前的测试**
- **Copy 1 (Cell 8)**: 在提取特征向量前，先进行了一次模型评估测试，输出初始性能指标
  - MAE: 0.68567, MSE: 0.75857, pearson: 0.68711, spearman: 0.70560, R2: 0.45278
- **原版**: 没有这个测试步骤

#### 5. **聚类和样本选择策略**

- **原版 (Cell 13-14)**:
  - 使用 `select_representative_viruses` 函数
  - 对病毒 embedding 进行聚类（n_clusters=sample_virus_size）
  - 使用 `unique_virus_emb` 去重
  - 固定样本大小：1300
  - 循环处理多个样本大小（在绘图部分）

- **Copy 1 (Cell 11-13)**:
  - 使用肘部法则确定聚类数
  - 使用简单的 KMeans 聚类（n_clusters=200）
  - 过滤样本数 < 5 的簇
  - 按比例采样（sample_ratio）
  - 或者使用更简单的聚类方法（n_clusters=20，选择离中心最近的样本）

#### 6. **训练函数 `multidata_trainer`**

- **原版 (Cell 13)**:
  - 只有 `test_dataloader` 参数
  - 只评估测试集
  - **有模型保存功能**（`save_path` 参数）
  - **有日志记录功能**（`log_metrics`）
  - 保存最佳模型（基于 MSE）
  
- **Copy 1 (Cell 14)**:
  - 有 `valid_dataloader` 和 `test_dataloader` 两个参数
  - 同时评估验证集和测试集
  - **没有模型保存功能**
  - **没有日志记录功能**
  - 绘制验证集和测试集的 MSE 曲线

#### 7. **训练参数**
- **原版**:
  - lr_rate=0.00001
  - epoch=60
  - 保存模型和日志到指定路径
  
- **Copy 1**:
  - lr_rate=0.00002
  - epoch=10
  - 不保存模型

#### 8. **代码结构**
- **原版**: 更完整的生产级代码，包含保存和日志功能
- **Copy 1**: 更偏向实验和探索，有多个不同的聚类尝试（Cell 12-13）

---

## 原版 vs Copy 2 (Active_learning_41 copy 2.ipynb)

### 主要差异：

#### 1. **导入语句**
- **原版**: 包含 `import torch.nn as nn`
- **Copy 2**: 包含 `import torch.nn as nn`（与原版相同）

#### 2. **设备设置**
- **原版**: `device = torch.device('cuda:1')`
- **Copy 2**: `device = torch.device('cuda:0')`

#### 3. **函数定义**
- **Copy 2 (Cell 2)**: 有额外的 `monte_carlo_preds`、`set_all_dropout_p` 和 `add_head_dropout` 函数
- **原版**: 这些函数在 Cell 1 中定义

#### 4. **Monte Carlo 不确定性采样**
- **Copy 2 (Cell 10-12)**: 
  - 加载了不同的模型（1.8_Artificial_back/final.pth）
  - 使用 Monte Carlo 方法计算不确定性（T=20）
  - 计算预测的方差作为不确定性指标
  - 创建 `uncertainty_df` 包含 var、mean、uncertainty 列
  
- **原版**: 没有这个功能

#### 5. **多种聚类选择方法**

- **Copy 2** 实现了多种样本选择策略：
  
  a. **传统方法 (Cell 17-19)**:
     - 使用 `select_representative_viruses`（n_clusters=50）
     - 保存到 CSV 文件
  
  b. **Blockwise 选择 (Cell 21, 24)**:
     - 实现 `blockwise_selection` 函数
     - 分层聚类：血清 × 病毒（8 × 15）
     - 在每个 block 内选择离中心最近的样本
     - 选中 222 个样本
  
  c. **带不确定性的 Blockwise 选择 (Cell 23)**:
     - 实现 `blockwise_selection` 函数（带 uncertainty 参数）
     - 综合不确定性和代表性进行打分
     - 使用 alpha 参数平衡不确定性和代表性

- **原版**: 只使用传统的 `select_representative_viruses` 方法

#### 6. **训练函数**

- **Copy 2** 有两个版本的 `multidata_trainer`:
  
  a. **Cell 30**: 有 valid_dataloader 的版本（与原版不同）
  b. **Cell 31**: 只有 test_dataloader 的版本（与原版类似，但有保存功能）

#### 7. **训练参数**
- **Copy 2**:
  - lr_rate=0.00001
  - epoch=30
  - 保存模型和日志到指定路径（Cell 32-33）

#### 8. **模型评估**
- **Copy 2 (Cell 34-35)**: 
  - 加载训练好的模型进行最终评估
  - 分别评估 active 和 random 选择策略的模型

#### 9. **数据加载器批次大小**
- **原版**: Active_candidate_dataloader batch_size=128
- **Copy 2**: Active_candidate_dataloader batch_size=128（相同）

---

## Copy 1 vs Copy 2 的主要区别

1. **不确定性采样**：
   - Copy 2 有完整的 Monte Carlo 不确定性计算
   - Copy 1 没有

2. **聚类策略多样性**：
   - Copy 2 实现了多种聚类选择方法（传统、blockwise、带不确定性的 blockwise）
   - Copy 1 使用简单的 KMeans 聚类

3. **模型版本**：
   - Copy 2 尝试使用不同的模型（1.8_Artificial_back/final.pth）
   - Copy 1 使用原版相同的模型

4. **训练轮数**：
   - Copy 1: epoch=10
   - Copy 2: epoch=30

5. **代码完整性**：
   - Copy 2 更完整，包含模型保存、评估等完整流程
   - Copy 1 更偏向快速实验

6. **函数定义**：
   - Copy 2 有更多辅助函数（dropout 相关）
   - Copy 1 函数较少

---

## 总结

### 原版特点：
- 生产级代码，完整的保存和日志功能
- 使用传统的病毒聚类方法
- 支持批量处理多个样本大小
- 使用 cuda:1

### Copy 1 特点：
- 实验性代码，快速验证
- 增加验证集评估
- 使用简单的 KMeans 聚类
- 使用 cuda:0
- 有初始性能测试

### Copy 2 特点：
- 最完整的实验版本
- **引入 Monte Carlo 不确定性采样**
- **实现多种聚类选择策略**（传统、blockwise、带不确定性的 blockwise）
- 完整的模型训练和评估流程
- 使用 cuda:0
- 有 dropout 相关工具函数

**关键创新点（Copy 2）**：
1. Monte Carlo 不确定性量化
2. Blockwise 分层聚类选择
3. 不确定性+代表性综合打分机制

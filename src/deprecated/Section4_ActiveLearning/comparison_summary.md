# Active_learning_CNIC.ipynb 版本对比总结

## 原版 vs Copy 1 (Active_learning_CNIC copy.ipynb)

### 主要差异：

#### 1. **导入语句**
- **原版**: 包含 `import torch.nn as nn`
- **Copy 1**: 不包含 `torch.nn as nn`

#### 2. **数据处理方式**
- **原版 (Cell 2)**: 
  - 使用 `group_and_filt(Active_data, fill_word=None)`，**不添加** `_active` 后缀
  - 直接处理数据，没有血清筛选步骤
  
- **Copy 1 (Cell 2)**: 
  - 手动实现 `group_by` 和聚合
  - **添加** `_active` 后缀到所有 seq_id
  - **新增血清筛选**：只保留出现次数 >= 20 的血清组合
  ```python
  serums = Active_final[['seq_id_a', 'seq_id_b', 'serumPassCat']].value_counts().reset_index()
  serums = serums[serums['count'] >=20]
  Active_final = serums[['seq_id_a', 'seq_id_b', 'serumPassCat']].merge(Active_final, on=['seq_id_a', 'seq_id_b', 'serumPassCat'])
  ```

#### 3. **函数定义**
- **原版**: 
  - 包含 `group_and_filt` 函数
  - 包含两个版本的 `monte_carlo_preds` 函数（Cell 2 和 Cell 3）
  - 包含 `set_all_dropout_p` 和 `add_head_dropout` 函数
  
- **Copy 1**: 
  - 没有 `group_and_filt` 函数
  - 只有一个 `monte_carlo_preds` 函数
  - 没有 dropout 相关函数

#### 4. **训练函数 `multidata_trainer`**
- **原版**: 
  - 只有 `test_dataloader` 参数
  - 只评估测试集，不保存模型
  
- **Copy 1**: 
  - 有 `valid_dataloader` 和 `test_dataloader` 两个参数
  - 同时评估验证集和测试集
  - 绘制验证集和测试集的 MSE 曲线

#### 5. **聚类和样本选择策略**
- **原版 (Cell 8-14)**: 
  - 使用 `select_representative_viruses` 函数
  - 对病毒 embedding 进行聚类（n_clusters=15）
  - 循环处理多个样本大小：[1000, 1200, 1407]
  - 使用 `unique_virus_emb` 去重
  
- **Copy 1 (Cell 8-11)**: 
  - **双重聚类策略**：分别对血清和病毒进行聚类
  - 血清聚类数：10，病毒聚类数：10
  - 在每个 (血清簇, 病毒簇) 组合块中选择代表性样本
  - 选中 69 个样本
  - 还有一个更复杂的循环选择逻辑（Cell 10），按血清分组进行聚类选择

#### 6. **训练参数**
- **原版**: 
  - 循环训练多个样本大小
  - lr_rate=0.00001, epoch=60
  - 保存模型和日志到文件
  
- **Copy 1**: 
  - 单次训练
  - lr_rate=0.00004, epoch=30 (active) 和 20 (random)
  - 不保存模型

#### 7. **数据加载器批次大小**
- **原版**: batch_size=8 (Active_candidate_dataloader)
- **Copy 1**: batch_size=256 (Active_candidate_dataloader)

---

## 原版 vs Copy 2 (Active_learning_CNIC copy 2.ipynb)

### 主要差异：

#### 1. **导入语句**
- **原版**: 包含 `import torch.nn as nn`
- **Copy 2**: 不包含 `torch.nn as nn`

#### 2. **数据处理方式**
- **Copy 2**: 与 Copy 1 相同
  - 添加 `_active` 后缀
  - 血清筛选（count >= 20）

#### 3. **聚类参数**
- **Copy 2 (Cell 9)**: 
  - 血清聚类数：**10**
  - 病毒聚类数：**20**（与 Copy 1 的 10 不同）
  - 选中 **124** 个样本（Copy 1 是 69 个）

#### 4. **训练函数**
- **Copy 2**: 与 Copy 1 相同，包含 `valid_dataloader` 参数

#### 5. **训练参数**
- **Copy 2**: 
  - lr_rate=0.00004
  - **epoch=20 (active)** 和 **epoch=30 (random)**（与 Copy 1 相反）

#### 6. **其他差异**
- **Copy 2**: 添加了 `Active_final.shape` 检查（Cell 3）
- **Copy 2**: 没有 Copy 1 中 Cell 10 的复杂循环选择逻辑

---

## Copy 1 vs Copy 2 的主要区别

1. **病毒聚类数**：
   - Copy 1: 10 个病毒簇
   - Copy 2: 20 个病毒簇

2. **选中样本数**：
   - Copy 1: 69 个
   - Copy 2: 124 个

3. **训练轮数**：
   - Copy 1: active=30, random=20
   - Copy 2: active=20, random=30

4. **代码复杂度**：
   - Copy 1 有更复杂的样本选择逻辑（Cell 10）
   - Copy 2 更简洁

---

## 总结

- **原版**：使用传统的病毒聚类方法，支持批量处理多个样本大小，有完整的模型保存和日志记录功能
- **Copy 1**：引入双重聚类策略（血清+病毒），添加血清筛选，增加验证集评估
- **Copy 2**：在 Copy 1 基础上调整聚类参数（更多病毒簇）和训练轮数，代码更简洁

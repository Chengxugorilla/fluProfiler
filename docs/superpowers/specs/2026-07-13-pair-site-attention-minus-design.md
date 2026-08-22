# Pair-Site Attention SerumGate-Minus 设计

日期：2026-07-13

## 目标

新增 V1 SerumGate-Minus 变体，使模型能直接在 serum-virus 的对齐 HA
位点对上进行推理。模型必须使用现有基础模型 embedding，提供可解释的显式
位点权重，并保持现有 homologous、heterologous 和 antigenic-distance 三类
监督契约。

## 范围

V1 包含：

- serum 和 virus HA token embedding 的共享降维
- embedding 生成后的 HA 对齐坐标映射
- per-site pair token 构造
- 两层 Transformer encoder
- 显式 softmax site-attention pooling
- 利用 passage 特征预测均值和对数方差的 head
- SerumGate-Minus 的 self/query score 相减
- 以 `train_minus_homo.py` 为模板的新训练入口

V1 不包含 mean-pooling shortcut、learned gate、entropy regularization、
entmax、生物学位点 prior、CLS pooling 或 NA embedding branch。现有的 NA
glycan mismatch 标量特征仍作为 prediction head 的上下文特征。

## Embedding 与对齐数据流

基础模型 embedding 必须由天然的、不含 gap 的蛋白序列生成；本功能不重新
计算 embedding。训练集使用 `seq_id_a` 和 `seq_id_c` 加载现有文件：

```text
matrix_<seq_id_a>.pt -> serum HA embedding [Ls, hidden_size]
matrix_<seq_id_c>.pt -> virus HA embedding [Lv, hidden_size]
```

只在 embedding 生成后做对齐。`serumHA` 和 `virusHA` 是处于同一 HA 坐标系
中的等长 gap 序列。遍历每条对齐序列时，每遇到一个非 gap 残基就消费一行
embedding；每遇到一个 gap 就插入一行零向量。映射器必须验证：

- 成对的 `serumHA` 与 `virusHA` 字符串长度相等；
- serum 非 gap 残基数等于 `Ls`；
- virus 非 gap 残基数等于 `Lv`。

映射器输出对齐后的 embedding 及独立的 serum/virus residue mask。当前
H3N2 数据中，检查到的 24,924 行均满足这些条件；对齐长度均为 329，且每个
serum ID 只有一个稳定的对齐表示。

在一个 serum task 内，仅当需要组成规则张量时才对齐后的 query matrix 做
padding。两个序列都已结束后的 batch padding 必须被 mask 掉。任一侧有残基
的对齐列仍是有效 pair site，因此插入和缺失会被模型看见。

## Pair-Site Encoder

serum 和 virus 共用一个 projection：

```text
site_projection: hidden_size -> site_proj_dim
```

投影后，每侧向量都要乘对应 residue mask。这样即使 projection 含 bias，gap
仍严格保持为零。每个对齐位点构造：

```text
x_i = [s_i, v_i, v_i - s_i, abs(v_i - s_i)]
h_i = Dropout(GELU(Linear(x_i)))
```

默认维度：

```text
site_proj_dim = 128
d_model = 256
num_layers = 2
num_heads = 4
transformer_ff_dim = 1024
site_attention_dim = 128
dropout = 0.1
```

对 pair-site token 加入可训练的绝对位置 embedding，再送入 Transformer。
最大位点长度从加载的 split alignment string 中推断，并保存到 model config。
不使用 CLS token：每个 site token 已可通过 self-attention 与所有其他位点交换
信息，最终序列表示只通过显式的 site-attention pooling 产生。

该 site-attention head 独立于 Transformer 内部 multi-head attention：

```text
score_i = Linear(tanh(Linear(H_i)))
alpha = masked_softmax(score)
pair_repr = sum(alpha_i * H_i)
```

Transformer 用于得到上下文化的位点表示；显式 attention pooling 用于从这些
位点表示中选择并汇总真正参与预测的 site。`alpha` 在纯 padding 列为零，且在
有效 pair site 上和为一。它作为模型输出，用于解释和 site occlusion 实验。

## Prediction 与 Minus Wrapper

score head 将 `pair_repr` 与现有的 query-passage、passage-pair、可选 subtype
和标量 NA glycan mismatch 特征拼接。一个单隐藏层 MLP 输出 score mean 和
log variance，log variance 沿用现有范围截断。

同一个共享 pair score model 被调用两次：

```text
query = PairScoreModel(serum HA, virus HA, query context)
self  = PairScoreModel(serum HA, serum HA, self context)

distance_mean = self.mean - query.mean
```

距离方差沿用现有的 `sum` 或 `query` 行为。query 与 self 前向完全共享参数，
self 分支不会使参数量翻倍。

对外输出保留当前 trainer 所需的键：

```text
mean, log_var
self_score, query_score
self_log_var, query_log_var
huber_loss, nll_loss, rank_loss
```

并新增：

```text
query_site_attention [batch, query_count, aligned_length]
self_site_attention  [batch, query_count, aligned_length]
```

在 evaluation mode 下，完全相同的 serum-serum pair 的 Minus distance 应为零
（允许浮点误差）。

## 训练入口

新增 `experiments/serum_gate/train_pair_site_minus_homo.py`。它复用现有的 split
加载、vocabulary 构造、embedding cache、task grouping、三目标 loss、metrics、
checkpoint selection 和输出目录结构；只替换 pair-site 模型和 alignment collate。
pair-site 专属 CLI 参数暴露上述默认维度。

新入口始终验证 `serumHA` 与 `virusHA`，并始终使用 embedding 后的对齐映射。
它不暴露旧的 HA pooling、HA pair mode、latent、calibrated metric 或 NA embedding
branch 参数。

`run_config.json` 与 `best_model.pth` 必须保存完整的 pair-site model config，
从而无需依赖 CLI 默认值即可重建模型。

## 错误处理

若 embedding 文件缺失、embedding 不是二维、成对 alignment 长度不同，或非 gap
残基数和 embedding 长度不一致，训练必须给出包含行或 sequence ID 的明确错误。
模型必须拒绝非正维度、不能整除 `d_model` 的 attention head 数，以及超过 position
embedding 表范围的输入序列。

## 测试

测试覆盖：

- gap 插入后 embedding 行被映射到正确的对齐坐标；
- 非法 alignment 或 embedding 长度报出清晰错误；
- serum 和 virus 只有一个共享 projection module；
- pair attention 在无效 padding 上为零，并在有效位点上和为一；
- 插入或缺失列仍是有效 pair site；
- forward 输出符合现有 Minus shape，且 loss 有限；
- evaluation mode 下相同 self pair 的 distance 为零；
- shared projection、Transformer、site-attention head 与 prediction head 都能收到梯度；
- 新 CLI 默认值等于 V1 维度；
- 一轮 CPU smoke run 可输出 metrics、predictions、run config 和可重建 checkpoint。

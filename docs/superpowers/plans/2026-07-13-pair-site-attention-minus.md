# Pair-Site Attention SerumGate-Minus Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现一个使用现成 HA embedding、embedding 后对齐、两层 Transformer 和显式 site-attention pooling 的 SerumGate-Minus 模型，并提供独立训练入口。

**Architecture:** 新模型复用 `SerumGateBatch` 和现有 loss helper，但使用独立的 pair-site score encoder；训练入口复用现有 SerumGate 数据、指标和训练函数，并为 pair-site 对齐提供严格验证。serum/query 共享投影，Transformer 输出由显式 attention pooling 汇总，不使用 CLS、mean 或 max shortcut。

**Tech Stack:** Python 3.10+、PyTorch、pandas、NumPy、pytest/unittest、现有 fluProfiler SerumGate 训练工具。

## Global Constraints

- 只加载现有 `matrix_<seq_id>.pt`，不重新生成 embedding。
- gap 只在 embedding 生成后根据 `serumHA`/`virusHA` 插入。
- `site_proj_dim=128`、`d_model=256`、`num_layers=2`、`num_heads=4`、`transformer_ff_dim=1024`、`site_attention_dim=128`、`dropout=0.1`。
- prediction representation 只来自显式 site-attention pooling，不使用 CLS、mean pooling、max pooling 或 gate。
- V1 不包含 NA embedding branch；保留标量 NA glycan mismatch 特征。
- 保持现有 `label`、`homo_label`、`diff_label` 三目标监督和 checkpoint 输出契约。

---

### Task 1: Pair-Site Score Encoder 与 Minus Model

**Files:**
- Create: `src/fluprofiler/models/pair_site_attention_minus_model.py`
- Create: `tests/test_pair_site_attention_minus_model.py`

**Interfaces:**
- Consumes: `SerumGateBatch`、`label_bin_weights`、`pairwise_ranking_loss`、`weighted_masked_mean` from `serum_gate_model.py`
- Produces: `PairSiteAttentionMinusConfig`、`PairSiteAttentionScoreModel`、`PairSiteAttentionMinusModel`

- [ ] **Step 1: 写配置校验和共享投影的失败测试**

```python
def tiny_config(**overrides):
    values = dict(
        hidden_size=6,
        max_site_length=5,
        site_proj_dim=4,
        d_model=8,
        num_layers=2,
        num_heads=2,
        transformer_ff_dim=16,
        site_attention_dim=4,
        predictor_hidden_dim=8,
        passage_vocab_size=3,
        passage_pair_vocab_size=9,
        subtype_vocab_size=2,
        passage_dim=2,
        subtype_dim=2,
        dropout=0.0,
    )
    values.update(overrides)
    return PairSiteAttentionMinusConfig(**values)


def test_pair_model_has_one_shared_site_projection():
    model = PairSiteAttentionMinusModel(tiny_config())
    projections = [name for name, _ in model.named_modules() if name.endswith("site_projection")]
    assert projections == ["score_model.site_projection"]


def test_config_rejects_heads_that_do_not_divide_model_dimension():
    with pytest.raises(ValueError, match="d_model must be divisible"):
        PairSiteAttentionMinusModel(tiny_config(d_model=7, num_heads=2))
```

- [ ] **Step 2: 运行测试并确认因模块不存在而失败**

Run: `conda run -n fluProfiler pytest tests/test_pair_site_attention_minus_model.py -q`

Expected: FAIL，提示 `fluprofiler.models.pair_site_attention_minus_model` 不存在。

- [ ] **Step 3: 实现配置、共享投影、Transformer 和显式 pooling**

```python
@dataclass
class PairSiteAttentionMinusConfig:
    hidden_size: int
    max_site_length: int
    site_proj_dim: int = 128
    d_model: int = 256
    num_layers: int = 2
    num_heads: int = 4
    transformer_ff_dim: int = 1024
    site_attention_dim: int = 128
    dropout: float = 0.1
    passage_vocab_size: int = 6
    passage_pair_vocab_size: int = 36
    subtype_vocab_size: int = 1
    passage_dim: int = 8
    subtype_dim: int = 8
    predictor_hidden_dim: int = 256
    min_log_var: float = -6.0
    max_log_var: float = 4.0
    label_weight_thresholds: tuple[float, ...] = (2.0, 4.0, 6.0)
    label_weight_values: tuple[float, ...] = (1.0, 1.3, 1.8, 2.5)
    rank_loss_weight: float = 0.0
    rank_loss_margin: float = 0.1
    rank_loss_min_label_delta: float = 0.0
    score_log_var_mode: str = "sum"
```

`PairSiteAttentionScoreModel._encode_pairs` 必须：

```python
reference_mask = reference_mask > 0
query_mask = query_mask > 0
pair_mask = reference_mask[:, None, :] | query_mask
reference_z = self.site_projection(reference_ha) * reference_mask.unsqueeze(-1)
query_z = self.site_projection(query_ha) * query_mask.unsqueeze(-1)
reference_z = reference_z[:, None].expand(-1, query_ha.shape[1], -1, -1)
pair_tokens = torch.cat(
    [reference_z, query_z, query_z - reference_z, torch.abs(query_z - reference_z)],
    dim=-1,
)
hidden = self.pair_projection(pair_tokens)
hidden = hidden + self.position_embedding[: hidden.shape[-2]]
hidden = self.transformer(
    hidden.flatten(0, 1),
    src_key_padding_mask=~pair_mask.flatten(0, 1),
).unflatten(0, pair_mask.shape[:2])
logits = self.site_attention(hidden).squeeze(-1)
logits = logits.masked_fill(~pair_mask, torch.finfo(logits.dtype).min)
alpha = torch.softmax(logits, dim=-1) * pair_mask
alpha = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(1e-12)
pair_repr = torch.sum(alpha.unsqueeze(-1) * hidden, dim=-2)
```

Prediction head 输入维度必须为 `d_model + passage_dim * 2 + subtype_dim + 1`，输出 `mean`、clamped `log_var` 和 `site_attention`。

- [ ] **Step 4: 增加 mask、attention 和 Minus 输出测试**

```python
def test_attention_uses_union_mask_and_normalizes_valid_sites():
    model = PairSiteAttentionMinusModel(tiny_config()).eval()
    batch = make_batch(
        reference_mask=torch.tensor([[1, 1, 0, 0, 0]]),
        query_mask=torch.tensor([[[1, 0, 1, 0, 0]]]),
    )
    out = model(batch)
    alpha = out["query_site_attention"][0, 0]
    assert torch.allclose(alpha[[3, 4]], torch.zeros(2))
    assert torch.isclose(alpha[:3].sum(), torch.tensor(1.0))
    assert alpha[2] > 0


def test_eval_self_pair_has_zero_minus_distance():
    model = PairSiteAttentionMinusModel(tiny_config()).eval()
    batch = make_self_pair_batch()
    out = model(batch)
    assert torch.allclose(out["mean"], torch.zeros_like(out["mean"]), atol=1e-6)
```

同时断言输出 shape、loss finite，并在 `out["nll_loss"].backward()` 后检查 `site_projection`、Transformer、`site_attention` 和 prediction head 梯度非空。

- [ ] **Step 5: 运行模型测试**

Run: `conda run -n fluProfiler pytest tests/test_pair_site_attention_minus_model.py -q`

Expected: PASS。

- [ ] **Step 6: 提交模型与模型测试**

```bash
git add src/fluprofiler/models/pair_site_attention_minus_model.py tests/test_pair_site_attention_minus_model.py
git commit --only -m "feat: add pair-site attention minus model" -- src/fluprofiler/models/pair_site_attention_minus_model.py tests/test_pair_site_attention_minus_model.py
```

---

### Task 2: Embedding 后对齐与 Pair-Site CLI

**Files:**
- Create: `experiments/serum_gate/train_pair_site_minus_homo.py`
- Create: `tests/test_train_pair_site_minus_homo.py`

**Interfaces:**
- Consumes: existing dataset/vocab/embedding helpers from `train_minus_homo.py`
- Produces: `validate_pair_site_alignment`、`infer_max_site_length`、`parse_args`

- [ ] **Step 1: 写严格 alignment 验证的失败测试**

```python
def test_validate_pair_site_alignment_accepts_post_embedding_gap_mapping():
    frame = pd.DataFrame([{
        "seq_id_a": "HA_REF",
        "seq_id_c": "HA_QUERY",
        "seq_a": "ABC",
        "seq_c": "ADC",
        "serumHA": "AB-C",
        "virusHA": "A-DC",
    }])
    embeddings = {
        "matrix_HA_REF": torch.ones(3, 6),
        "matrix_HA_QUERY": torch.ones(3, 6),
    }
    assert module.validate_pair_site_alignment({"train": frame}, embeddings) == 4


def test_validate_pair_site_alignment_rejects_embedding_length_mismatch():
    frame = aligned_frame()
    embeddings = {
        "matrix_HA_REF": torch.ones(2, 6),
        "matrix_HA_QUERY": torch.ones(3, 6),
    }
    with pytest.raises(ValueError, match="HA_REF.*3.*2"):
        module.validate_pair_site_alignment({"train": frame}, embeddings)
```

还要覆盖 pair alignment 长度不同、ungapped alignment 不等于 `seq_a/seq_c`，以及同一 `seq_id_a` 对应多个 `serumHA` 的错误。

- [ ] **Step 2: 运行 alignment 测试并确认失败**

Run: `conda run -n fluProfiler pytest tests/test_train_pair_site_minus_homo.py -q`

Expected: FAIL，提示新入口不存在。

- [ ] **Step 3: 实现 alignment 验证和 max length 推断**

```python
def validate_pair_site_alignment(
    frames: dict[str, pd.DataFrame],
    embeddings: dict[str, torch.Tensor],
) -> int:
    max_length = 0
    reference_alignments: dict[str, str] = {}
    for split_name, frame in frames.items():
        for row_index, row in frame.iterrows():
            serum_aligned = str(row["serumHA"])
            virus_aligned = str(row["virusHA"])
            if len(serum_aligned) != len(virus_aligned):
                raise ValueError(f"{split_name} row {row_index}: alignment lengths differ")
            for side, seq_id, raw_seq, aligned in (
                ("serum", str(row["seq_id_a"]), str(row["seq_a"]), serum_aligned),
                ("virus", str(row["seq_id_c"]), str(row["seq_c"]), virus_aligned),
            ):
                ungapped = aligned.replace("-", "")
                if ungapped != raw_seq:
                    raise ValueError(f"{split_name} row {row_index}: {side} alignment does not match {seq_id}")
                matrix = embeddings[f"matrix_{seq_id}"]
                if matrix.ndim != 2 or matrix.shape[0] != len(ungapped):
                    raise ValueError(
                        f"{seq_id}: expected {len(ungapped)} embedding rows, got {matrix.shape[0]}"
                    )
            previous = reference_alignments.setdefault(str(row["seq_id_a"]), serum_aligned)
            if previous != serum_aligned:
                raise ValueError(f"{row['seq_id_a']}: multiple serumHA alignments")
            max_length = max(max_length, len(serum_aligned))
    return max_length
```

- [ ] **Step 4: 实现仅包含 common + pair-site 参数的新 CLI**

新增参数：

```python
parser.add_argument("--site-proj-dim", type=int, default=128)
parser.add_argument("--d-model", type=int, default=256)
parser.add_argument("--num-layers", type=int, default=2)
parser.add_argument("--num-heads", type=int, default=4)
parser.add_argument("--transformer-ff-dim", type=int, default=1024)
parser.add_argument("--site-attention-dim", type=int, default=128)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--predictor-hidden-dim", type=int, default=256)
```

CLI 继续提供现有数据路径、split、loss weights、label weights、ranking loss、optimizer、scheduler、GPU cache、subtype 和 progress 参数；不添加 `--ha-pooling`、`--ha-pair-mode`、`--latent-dim`、`--predictor-arch` 或 `--na-branch`。

- [ ] **Step 5: 运行 alignment 和 CLI 测试**

Run: `conda run -n fluProfiler pytest tests/test_train_pair_site_minus_homo.py -q`

Expected: alignment 与 parse-args 测试 PASS；本任务不运行训练 smoke test。

- [ ] **Step 6: 提交 alignment、CLI 与对应测试**

```bash
git add experiments/serum_gate/train_pair_site_minus_homo.py tests/test_train_pair_site_minus_homo.py
git commit --only -m "feat: add pair-site training input pipeline" -- experiments/serum_gate/train_pair_site_minus_homo.py tests/test_train_pair_site_minus_homo.py
```

---

### Task 3: 独立训练入口与 Checkpoint

**Files:**
- Modify: `experiments/serum_gate/train_pair_site_minus_homo.py`
- Modify: `tests/test_train_pair_site_minus_homo.py`

**Interfaces:**
- Consumes: `PairSiteAttentionMinusConfig`、`PairSiteAttentionMinusModel`、existing `SerumGateTaskDataset`/collate/evaluation helpers
- Produces: `run_training(args) -> dict[str, Any]`、`main()`

- [ ] **Step 1: 写一轮 CPU smoke test**

使用两条 synthetic row，embedding shape 为 `[3, 6]`，alignment 为 `ABC`，运行：

```python
args = module.parse_args([
    "--data-dir", str(data_dir),
    "--embedding-dir", str(embedding_dir),
    "--output-dir", str(output_dir),
    "--type", "H1N1",
    "--epochs", "1",
    "--site-proj-dim", "4",
    "--d-model", "8",
    "--num-layers", "2",
    "--num-heads", "2",
    "--transformer-ff-dim", "16",
    "--site-attention-dim", "4",
    "--predictor-hidden-dim", "8",
    "--dropout", "0",
    "--skip-test-eval",
    "--no-progress",
])
result = module.run_training(args)
```

断言 `metrics.csv` 包含 `valid_diff_mse`，`predictions_valid.csv` 包含现有 score 列，checkpoint 的 `model_config` 包含 `site_proj_dim=4`、`num_layers=2` 和 `max_site_length=3`。

- [ ] **Step 2: 运行 smoke test 并确认失败**

Run: `conda run -n fluProfiler pytest tests/test_train_pair_site_minus_homo.py -q`

Expected: FAIL，提示 `run_training` 尚未定义或无输出。

- [ ] **Step 3: 实现训练流程**

训练流程按以下顺序执行：

```python
frames = base.load_fixed_split_frames(...)
base.validate_aligned_ha_columns(frames)
embedding_files = base.required_embedding_files(frames, include_na_embeddings=False)
embedding_dir = base.validate_embedding_files(args.embedding_dir, embedding_files)
embeddings = base.load_embeddings(embedding_dir, embedding_files, show_progress=args.progress)
max_site_length = validate_pair_site_alignment(frames, embeddings)
hidden_size = base.infer_hidden_size(embeddings)
vocabs = base.build_serum_gate_vocabs(frames, use_subtype_feature=args.use_subtype_feature)
loaders = {
    name: base.build_loader(..., align_ha_embeddings=True, include_na_embeddings=False)
    for name, frame in frames.items()
}
model_config = PairSiteAttentionMinusConfig(
    hidden_size=hidden_size,
    max_site_length=max_site_length,
    site_proj_dim=args.site_proj_dim,
    d_model=args.d_model,
    num_layers=args.num_layers,
    num_heads=args.num_heads,
    transformer_ff_dim=args.transformer_ff_dim,
    site_attention_dim=args.site_attention_dim,
    dropout=args.dropout,
    passage_vocab_size=len(vocabs.passage_to_id),
    passage_pair_vocab_size=len(vocabs.passage_to_id) ** 2,
    subtype_vocab_size=len(vocabs.subtype_to_id),
    passage_dim=args.passage_dim,
    subtype_dim=args.subtype_dim,
    predictor_hidden_dim=args.predictor_hidden_dim,
    label_weight_thresholds=tuple(label_weight_thresholds),
    label_weight_values=tuple(label_weight_values),
    rank_loss_weight=args.rank_loss_weight,
    rank_loss_margin=args.rank_loss_margin,
    rank_loss_min_label_delta=args.rank_loss_min_label_delta,
    score_log_var_mode=args.score_log_var_mode,
)
```

随后沿用现有 `AdamW`、可选 cosine scheduler、`base.train_one_epoch`、`base.evaluate_model`、best metric checkpoint selection 和 prediction CSV 写出逻辑。`run_config["model"]` 固定为 `SerumGate-Minus-PairSiteAttn-Homo`。

- [ ] **Step 4: 运行入口测试**

Run: `conda run -n fluProfiler pytest tests/test_train_pair_site_minus_homo.py -q`

Expected: PASS。

- [ ] **Step 5: 提交训练流程和 smoke test**

```bash
git add experiments/serum_gate/train_pair_site_minus_homo.py tests/test_train_pair_site_minus_homo.py
git commit --only -m "feat: train pair-site attention minus model" -- experiments/serum_gate/train_pair_site_minus_homo.py tests/test_train_pair_site_minus_homo.py
```

---

### Task 4: 回归验证与参数审计

**Files:**
- Modify only if a failing test reveals a scoped defect in files created above.

**Interfaces:**
- Consumes: completed model and training entry
- Produces: verified implementation with no regressions in existing SerumGate tests

- [ ] **Step 1: 运行新模型和新入口测试**

Run: `conda run -n fluProfiler pytest tests/test_pair_site_attention_minus_model.py tests/test_train_pair_site_minus_homo.py -q`

Expected: PASS。

- [ ] **Step 2: 运行现有 SerumGate 回归测试**

Run: `conda run -n fluProfiler pytest tests/test_serum_gate.py tests/test_train_minus_homo.py -q`

Expected: PASS。

- [ ] **Step 3: 审计默认参数量和输出 shape**

Run:

```bash
conda run -n fluProfiler python -c "from fluprofiler.models.pair_site_attention_minus_model import PairSiteAttentionMinusConfig,PairSiteAttentionMinusModel; c=PairSiteAttentionMinusConfig(hidden_size=2560,max_site_length=329,passage_vocab_size=4,passage_pair_vocab_size=16,subtype_dim=0); m=PairSiteAttentionMinusModel(c); print(sum(p.numel() for p in m.parameters()))"
```

Expected: 输出一个有限的约 2M 参数量整数；query/self 共享参数，参数量不会翻倍。

- [ ] **Step 4: 检查改动范围和格式**

Run: `git diff --check -- src/fluprofiler/models/pair_site_attention_minus_model.py experiments/serum_gate/train_pair_site_minus_homo.py tests/test_pair_site_attention_minus_model.py tests/test_train_pair_site_minus_homo.py`

Expected: 无输出。

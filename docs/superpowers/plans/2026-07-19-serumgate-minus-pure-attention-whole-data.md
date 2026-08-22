# SerumGate-Minus Pure-Attention Whole-Data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a schema-preserving whole CSV from the existing titer seed-0 partitions, train a final H3N2 SerumGate-Minus model with 100% low-rank attention pooling, and export per-sequence and global aligned-position attention summaries.

**Architecture:** New code is additive: a dataset builder creates `whole/all.csv`; a new `lowrank_attention_only` module leaves all old pooling modes unchanged; a dedicated no-holdout trainer consumes the whole CSV; and a separate exporter maps final pooling weights to aligned positions. Existing scripts, defaults, checkpoint formats, source CSVs, and result directories remain untouched.

**Tech Stack:** Python 3, pandas, NumPy, PyTorch, matplotlib, pytest/unittest, JSON, SHA-256.

## Global Constraints

- This is a pure computational workflow over existing CSV files, embeddings, and model weights.
- Source data is exactly `data/dataset/H1H3_HA1_v1.0/splited/20260717_164256/titer/seed_0/{train,valid,test}.csv`.
- Destination data is exactly `data/dataset/H1H3_HA1_v1.0/whole/{all.csv,manifest.json}`.
- `all.csv` retains both H1N1 and H3N2 and has exactly the source schema; do not add `source_split`.
- Training filters `--type H3N2`, uses every selected row, and has no held-out evaluation.
- New pooling mode is `lowrank_attention_only`; it computes no mean pool and owns no attention/mean gate.
- Architecture settings are latent 8, attention dimension 64, four heads, dropout 0.2, and independent HA pair encoding.
- Preserve existing `attention`, `mean`, and `lowrank_attention` behavior and old checkpoint loading.
- Do not modify unrelated dirty-worktree files. Stage explicit paths only.
- Do not launch the 200-epoch production run until separately requested.

## File Responsibility Map

- `scripts/build_whole_dataset.py`: validate and atomically concatenate the three source CSVs; write provenance manifest.
- `src/fluprofiler/models/serum_gate_model.py`: add the isolated pure-attention pooler and opt-in weight-return interface.
- `experiments/serum_gate/train_whole_minus_attention.py`: load `all.csv`, filter H3N2, and train/save a final no-holdout model.
- `src/fluprofiler/sites/serum_gate_attention.py`: build the unique-sequence registry, map embedding tokens to aligned positions, extract and aggregate attention, and plot profiles.
- `experiments/serum_gate/export_whole_attention.py`: validate checkpoint provenance and orchestrate interpretation artifact export.
- Focused tests mirror each new responsibility and lock old behavior before extension.

---

### Task 1: Whole-dataset builder

**Files:**
- Create: `scripts/build_whole_dataset.py`
- Test: `tests/test_build_whole_dataset.py`

**Interfaces:**
- Consumes: a directory containing `train.csv`, `valid.csv`, and `test.csv`.
- Produces: `build_whole_dataset(source_dir: Path, output_dir: Path, *, overwrite: bool = False) -> dict[str, object]`, plus `all.csv` and `manifest.json`.

- [ ] **Step 1: Write the failing schema-preservation test**

```python
def test_build_whole_dataset_preserves_schema_and_order(tmp_path):
    source = tmp_path / "seed_0"
    output = tmp_path / "whole"
    source.mkdir()
    columns = ["seq_id_a", "seq_id_b", "seq_id_c", "seq_id_d", "seq_a", "seq_c",
               "serumHA", "virusHA", "serumPassCat", "virusPassCat", "serumName",
               "label", "Type"]
    rows = {
        "train": [["a1", "b1", "c1", "d1", "AA", "AB", "AA", "AB", "egg", "cell", "s1", 1.0, "H1N1"]],
        "valid": [["a2", "b2", "c2", "d2", "AC", "AD", "AC", "AD", "cell", "egg", "s2", 2.0, "H3N2"]],
        "test":  [["a3", "b3", "c3", "d3", "AE", "AF", "AE", "AF", "egg", "egg", "s3", 3.0, "H3N2"]],
    }
    for split, values in rows.items():
        pd.DataFrame(values, columns=columns).to_csv(source / f"{split}.csv", index=False)

    manifest = build_whole_dataset(source, output)
    combined = pd.read_csv(output / "all.csv")

    assert combined.columns.tolist() == columns
    assert "source_split" not in combined.columns
    assert combined["seq_id_a"].tolist() == ["a1", "a2", "a3"]
    assert set(combined["Type"]) == {"H1N1", "H3N2"}
    assert manifest["row_counts"] == {"train": 1, "valid": 1, "test": 1, "total": 3}
```

- [ ] **Step 2: Run the test and confirm the intended failure**

Run: `python -m pytest tests/test_build_whole_dataset.py::test_build_whole_dataset_preserves_schema_and_order -q`

Expected: FAIL with `ModuleNotFoundError` for `scripts.build_whole_dataset`.

- [ ] **Step 3: Implement validated concatenation**

```python
SPLITS = ("train", "valid", "test")

def load_source_frames(source_dir: Path) -> dict[str, pd.DataFrame]:
    frames = {}
    for split in SPLITS:
        path = source_dir / f"{split}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing source split: {path}")
        frames[split] = pd.read_csv(path)
    schemas = [frame.columns.tolist() for frame in frames.values()]
    if any(schema != schemas[0] for schema in schemas[1:]):
        raise ValueError("Source split columns or column order differ")
    required = {"seq_id_a", "seq_id_b", "seq_id_c", "seq_id_d", "seq_a", "seq_c",
                "serumHA", "virusHA", "serumPassCat", "virusPassCat", "serumName",
                "label", "Type"}
    missing = sorted(required - set(schemas[0]))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    for split, frame in frames.items():
        pd.to_numeric(frame["label"], errors="raise")
    return frames

def concatenate_frames(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.concat([frames[name] for name in SPLITS], axis=0, ignore_index=True)
```

- [ ] **Step 4: Write failing manifest, overwrite, and atomicity tests**

Add tests asserting source/output SHA-256 values, exact columns, subtype counts, exact duplicate count, rejection of a nonempty destination, and preservation of an existing valid destination when source validation fails.

```python
def test_existing_output_requires_overwrite(tmp_path):
    source = make_valid_source(tmp_path)
    output = tmp_path / "whole"
    output.mkdir()
    (output / "all.csv").write_text("existing\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        build_whole_dataset(source, output)
    assert (output / "all.csv").read_text(encoding="utf-8") == "existing\n"
```

- [ ] **Step 5: Implement manifest and atomic replacement**

Use `hashlib.sha256` in 1 MiB chunks. Write temporary sibling files with `tempfile.NamedTemporaryFile(dir=output_dir, delete=False)`, reload and validate the temporary CSV, then use `os.replace` for `all.csv` followed by `manifest.json`. On failure, unlink only temporary files. With `overwrite=True`, replace only these two known targets; never delete the directory recursively.

The manifest keys are `created_at`, `source_dir`, `source_files`, `source_sha256`, `row_counts`, `columns`, `subtype_counts`, `exact_duplicate_rows`, `output_file`, `output_sha256`, and `options`.

- [ ] **Step 6: Add and verify the CLI**

```python
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build a whole fixed-data CSV.")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)
```

Run:

```bash
python -m pytest tests/test_build_whole_dataset.py -q
python scripts/build_whole_dataset.py --help
```

Expected: all tests PASS and help exits 0.

- [ ] **Step 7: Commit Task 1**

```bash
git add scripts/build_whole_dataset.py tests/test_build_whole_dataset.py
git commit -m "feat: add whole titer dataset builder"
```

---

### Task 2: Pure low-rank attention pooler

**Files:**
- Modify: `src/fluprofiler/models/serum_gate_model.py` near `HAPoolEncoder` and `LowRankSelfAttentionPool`
- Test: `tests/test_serum_gate.py`

**Interfaces:**
- Consumes: padded HA tensors `[batch, length, hidden_size]` and optional masks `[batch, length]`.
- Produces: `PureLowRankSelfAttentionPool.forward(..., return_attention=False)` and `HAPoolEncoder.encode_with_attention(...) -> tuple[Tensor, Tensor]`.

- [ ] **Step 1: Lock existing modes with regression tests**

```python
def test_existing_pooling_modes_keep_their_modules():
    attention = SerumGateModel(SerumGateConfig(hidden_size=8, ha_pooling="attention"))
    mean = SerumGateModel(SerumGateConfig(hidden_size=8, ha_pooling="mean"))
    mixed = SerumGateModel(SerumGateConfig(hidden_size=8, ha_pooling="lowrank_attention",
                                           ha_attention_dim=4, ha_attention_heads=2))
    assert attention.ha_encoder.attention_pooler is not None
    assert mean.ha_encoder.attention_pooler is None
    assert mixed.ha_encoder.lowrank_pooler is not None
    assert hasattr(mixed.ha_encoder.lowrank_pooler, "attention_gate_logit")
```

Run: `python -m pytest tests/test_serum_gate.py -q`

Expected: PASS before production code changes.

- [ ] **Step 2: Write failing pure-attention tests**

```python
def test_pure_lowrank_attention_returns_normalized_masked_weights():
    torch.manual_seed(7)
    pool = PureLowRankSelfAttentionPool(8, attention_dim=4, attention_heads=2, dropout=0.0)
    matrix = torch.randn(2, 4, 8, requires_grad=True)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    pooled, weights = pool(matrix, mask, return_attention=True)
    assert pooled.shape == (2, 8)
    assert weights.shape == (2, 4)
    assert torch.allclose(weights.sum(-1), torch.ones(2))
    assert torch.equal(weights[mask == 0], torch.zeros_like(weights[mask == 0]))
    assert not any("gate" in name for name, _ in pool.named_parameters())
    pooled.sum().backward()
    assert pool.input_projection.weight.grad is not None
    assert pool.context_attention.in_proj_weight.grad is not None
    assert pool.score.weight.grad is not None
```

Also test invalid dimensions and reject any sample whose mask has no valid token.

- [ ] **Step 3: Implement the isolated pure pooler**

```python
class PureLowRankSelfAttentionPool(nn.Module):
    def __init__(self, hidden_size, attention_dim=128, attention_heads=4, dropout=0.1):
        super().__init__()
        if attention_dim <= 0 or attention_heads <= 0:
            raise ValueError("attention dimension and heads must be positive")
        if attention_dim % attention_heads != 0:
            raise ValueError("ha_attention_dim must be divisible by ha_attention_heads")
        self.norm = nn.LayerNorm(hidden_size)
        self.input_projection = nn.Linear(hidden_size, attention_dim)
        self.context_attention = nn.MultiheadAttention(
            attention_dim, attention_heads, dropout=dropout, batch_first=True
        )
        self.score = nn.Linear(attention_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, matrix, mask=None, return_attention=False):
        valid = torch.ones(matrix.shape[:2], dtype=torch.bool, device=matrix.device)
        if mask is not None:
            valid = mask > 0
        if not torch.all(valid.any(dim=1)):
            raise ValueError("Every sequence must contain at least one valid token")
        context = self.input_projection(self.norm(matrix))
        context, _ = self.context_attention(
            context, context, context, key_padding_mask=~valid, need_weights=False
        )
        logits = self.score(torch.tanh(self.dropout(context))).squeeze(-1)
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1).masked_fill(~valid, 0.0)
        pooled = (matrix * weights.unsqueeze(-1)).sum(dim=1)
        return (pooled, weights) if return_attention else pooled
```

- [ ] **Step 4: Wire the new enum without changing defaults**

Extend the accepted set to include `lowrank_attention_only`. Add a separate `pure_lowrank_pooler` attribute. Standard `HAPoolEncoder.forward()` returns only the projected tensor. Add:

```python
def encode_with_attention(self, matrix, mask=None):
    if self.pure_lowrank_pooler is None:
        raise RuntimeError("Attention export requires lowrank_attention_only")
    pooled, weights = self.pure_lowrank_pooler(matrix, mask, return_attention=True)
    return self.projection(pooled), weights
```

Do not rename or repurpose `lowrank_pooler` and do not change `ha_pooling="attention"`.

- [ ] **Step 5: Verify old and new behavior**

Run:

```bash
python -m pytest tests/test_serum_gate.py -q
python -m pytest tests/test_infer_minus_checkpoint.py -q
```

Expected: all PASS.

- [ ] **Step 6: Commit Task 2**

```bash
git add src/fluprofiler/models/serum_gate_model.py tests/test_serum_gate.py
git commit -m "feat: add pure low-rank attention pooling"
```

---

### Task 3: Dedicated whole-data trainer

**Files:**
- Create: `experiments/serum_gate/train_whole_minus_attention.py`
- Test: `tests/test_train_whole_minus_attention.py`

**Interfaces:**
- Consumes: `all.csv`, HA embedding directory, H3N2 filter, training options.
- Produces: `load_whole_frame(data_file: Path, type_filter: str) -> pd.DataFrame`, `run_training(args: argparse.Namespace) -> dict[str, object]`, and a final checkpoint/config/log.

- [ ] **Step 1: Write failing loader tests**

```python
def test_load_whole_frame_filters_h3n2_without_deduplication(tmp_path):
    row_h3 = make_row(type_value="H3N2", seq_id_a="r3")
    frame = pd.DataFrame([make_row(type_value="H1N1", seq_id_a="r1"), row_h3, row_h3])
    path = tmp_path / "all.csv"
    frame.to_csv(path, index=False)
    loaded = load_whole_frame(path, "H3N2")
    assert loaded["Type"].tolist() == ["H3N2", "H3N2"]
    assert len(loaded) == 2
```

Add tests for missing columns, nonnumeric labels, empty H3N2 results, and missing task columns.

- [ ] **Step 2: Implement loader by reusing existing validators**

```python
def load_whole_frame(data_file: Path, type_filter: str) -> pd.DataFrame:
    path = Path(data_file).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Whole data file does not exist: {path}")
    frame = pd.read_csv(path)
    validate_frame_columns(frame, path)
    pd.to_numeric(frame["label"], errors="raise")
    filtered = filter_frames_by_type({"train": frame}, type_filter)["train"]
    if filtered.empty:
        raise ValueError(f"No training rows remain for --type {type_filter}")
    return filtered.reset_index(drop=True)
```

Import stable helpers from `train_zero_shot_minus.py`; do not copy dataset, collator, metric, or embedding-cache implementations.

- [ ] **Step 3: Write failing configuration and one-epoch tests**

Assert the constructed config contains latent 8, subtype dimension 0, `conditioned_mlp`, `lowrank_attention_only`, independent pair mode, attention 64/4/0.2, no NA branch, no rank loss, and variance mode sum. On a synthetic CPU dataset, assert exactly one training epoch, finite loss, no validation/test metrics, and a selected final checkpoint.

```python
assert run_config["evaluation_status"] == "train_all_no_holdout"
assert run_config["metrics_scope"] == "in_sample_training_diagnostics"
assert run_config["model_config"]["ha_pooling"] == "lowrank_attention_only"
assert checkpoint["epoch"] == 1
```

- [ ] **Step 4: Implement whole-data preparation and training**

Reuse `required_embedding_files`, `validate_embedding_files`, `load_embeddings`, `build_serum_gate_vocabs`, `build_loader`, `train_epoch`, `GpuEmbeddingCache`, and existing label-weight helpers. Construct one training loader only. Use `AdamW` and optional `CosineAnnealingLR`. Save the final epoch, never select by validation.

Output contract:

```text
run_config.json
metrics.csv
log.txt
checkpoints/final_model.pth
```

Checkpoint keys are `model_state_dict`, `model_config`, `task_cols`, `passage_to_id`, `subtype_to_id`, `epoch`, `data_file`, `data_checksum`, `type_filter`, and `evaluation_status`.

- [ ] **Step 5: Implement CLI validation and safe output handling**

Require `--data-file`, `--embedding-dir`, and `--output-dir`. Default `--type H3N2`, task columns `seq_id_a,serumPassCat,serumName`, epochs 200, latent 8, maximum queries 32, attention 64/4/0.2, cosine scheduler, minimum LR `1e-6`, and variance mode sum. Require `--ha-pooling lowrank_attention_only` if exposed. Do not hard-code a GPU. Reject nonempty output by default; overwrite only known artifacts when explicitly enabled.

- [ ] **Step 6: Verify trainer and regressions**

Run:

```bash
python -m pytest tests/test_train_whole_minus_attention.py -q
python -m pytest tests/test_serum_gate.py tests/test_infer_minus_checkpoint.py -q
python experiments/serum_gate/train_whole_minus_attention.py --help
```

Expected: all tests PASS; help exits 0.

- [ ] **Step 7: Commit Task 3**

```bash
git add experiments/serum_gate/train_whole_minus_attention.py tests/test_train_whole_minus_attention.py
git commit -m "feat: train pure-attention SerumGate on whole data"
```

---

### Task 4: Unique-sequence attention extraction and aligned mapping

**Files:**
- Create: `src/fluprofiler/sites/serum_gate_attention.py`
- Test: `tests/test_serum_gate_attention.py`

**Interfaces:**
- Consumes: filtered whole dataframe, unique HA embeddings, a pure-attention HA encoder.
- Produces: `build_sequence_registry(frame) -> list[HASequenceRecord]`, `map_tokens_to_alignment(record, embedding_length) -> list[tuple[int, int, str]]`, and `extract_attention_rows(...) -> pd.DataFrame`.

- [ ] **Step 1: Write failing registry tests**

```python
def test_registry_deduplicates_ids_and_tracks_roles():
    frame = pd.DataFrame([
        make_row(seq_id_a="shared", seq_a="ABC", serumHA="A-BC",
                 seq_id_c="query", seq_c="ABD", virusHA="A-BD"),
        make_row(seq_id_a="ref", seq_a="ABE", serumHA="A-BE",
                 seq_id_c="shared", seq_c="ABC", virusHA="A-BC"),
    ])
    registry = {record.seq_id: record for record in build_sequence_registry(frame)}
    assert registry["shared"].role == "both"
    assert set(registry) == {"shared", "query", "ref"}
```

Add a test that one `seq_id` with conflicting raw or aligned sequences raises `ValueError` containing that ID.

- [ ] **Step 2: Write failing mapping tests**

```python
def test_token_mapping_skips_alignment_gaps():
    record = HASequenceRecord("x", "both", "ABC", "A-BC", "matrix_x")
    assert map_tokens_to_alignment(record, 3) == [
        (0, 1, "A"), (1, 3, "B"), (2, 4, "C")
    ]
```

Reject embedding length mismatch, non-2D embeddings, and inconsistent ungapped alignment.

- [ ] **Step 3: Implement immutable records and mapping**

```python
@dataclass(frozen=True)
class HASequenceRecord:
    seq_id: str
    role: str
    sequence: str
    aligned_sequence: str
    embedding_key: str
```

Build candidates from `seq_id_a/seq_a/serumHA` as reference and `seq_id_c/seq_c/virusHA` as query. Merge identical records and role labels; never silently choose among conflicts. Human-facing aligned positions are one-based; `token_index` remains zero-based.

- [ ] **Step 4: Write failing extraction tests**

Using a tiny pure-attention encoder in `.eval()` mode, assert each unique ID is encoded once; output columns are exactly `seq_id,role,token_index,aligned_position,amino_acid,attention_weight`; weights are finite and sum to one per sequence; padding is absent; and repeated extraction is deterministic.

- [ ] **Step 5: Implement extraction through the opt-in encoder API**

Load each `matrix_<seq_id>` tensor, validate it as 2D, create an all-valid mask, call `ha_encoder.encode_with_attention`, and discard the latent output after validation. Convert only detached CPU weights to rows. Do not alter normal model forward or retain graphs.

- [ ] **Step 6: Verify and commit Task 4**

```bash
python -m pytest tests/test_serum_gate_attention.py -q
python -m pytest tests/test_serum_gate.py -q
git add src/fluprofiler/sites/serum_gate_attention.py tests/test_serum_gate_attention.py
git commit -m "feat: extract SerumGate HA attention by position"
```

---

### Task 5: Global summaries, top sites, and plot

**Files:**
- Modify: `src/fluprofiler/sites/serum_gate_attention.py`
- Test: `tests/test_serum_gate_attention.py`

**Interfaces:**
- Consumes: one row per sequence/token from Task 4.
- Produces: `summarize_attention(per_sequence) -> pd.DataFrame`, `select_top_sites(summary, per_sequence, top_k=20) -> pd.DataFrame`, and `plot_attention_profile(summary, top_sites, output_path) -> None`.

- [ ] **Step 1: Write failing equal-weight aggregation tests**

```python
def test_summary_averages_unique_sequences_and_ignores_gaps():
    rows = pd.DataFrame([
        {"seq_id": "a", "role": "reference", "aligned_position": 1, "amino_acid": "A", "attention_weight": .8},
        {"seq_id": "b", "role": "query", "aligned_position": 1, "amino_acid": "T", "attention_weight": .2},
        {"seq_id": "b", "role": "query", "aligned_position": 2, "amino_acid": "C", "attention_weight": .8},
    ])
    summary = summarize_attention(rows)
    pos1 = summary.loc[summary.aligned_position == 1].iloc[0]
    assert pos1.mean_attention == pytest.approx(.5)
    assert pos1.sequence_count == 2
```

Test deterministic descending mean rank with aligned-position tie breaking and separate reference/query counts, treating role `both` in both counts.

- [ ] **Step 2: Implement dataframe-only summary functions**

Group only observed residues; do not insert zeros for alignment gaps. Compute mean, median, population standard deviation, maximum, sequence count, reference count, query count, and deterministic rank. Serialize amino-acid distributions in `attention_top_sites.csv` as sorted JSON.

- [ ] **Step 3: Write failing noninteractive plot test**

Call the plot function on a small table, assert the PNG exists and is nonempty, inspect axis labels returned by a test-only optional axes injection or helper, and ensure the figure is closed. Do not use pixel snapshots.

- [ ] **Step 4: Implement the academic profile plot**

Use a wide white figure, aligned position on x, mean attention on y, a restrained blue line, variability band, and labels for at most top 20 sites. Use a noninteractive backend in CLI/tests. Accept an explicit output path and create only its parent directory.

- [ ] **Step 5: Verify and commit Task 5**

```bash
python -m pytest tests/test_serum_gate_attention.py -q
git add src/fluprofiler/sites/serum_gate_attention.py tests/test_serum_gate_attention.py
git commit -m "feat: summarize and plot HA attention sites"
```

---

### Task 6: Checkpoint-to-artifacts export CLI

**Files:**
- Create: `experiments/serum_gate/export_whole_attention.py`
- Test: `tests/test_export_whole_attention.py`

**Interfaces:**
- Consumes: Task 3 final checkpoint, whole CSV, embedding directory, output directory.
- Produces: `export_attention(args: argparse.Namespace) -> dict[str, Path]` and five finalized interpretation artifacts.

- [ ] **Step 1: Write failing checkpoint/provenance tests**

Reject a checkpoint missing required keys, a pooling mode other than `lowrank_attention_only`, a subtype other than H3N2, a dataset checksum mismatch, and a nonempty output without overwrite.

```python
with pytest.raises(ValueError, match="lowrank_attention_only"):
    validate_checkpoint({"model_config": {"ha_pooling": "lowrank_attention"}})
```

- [ ] **Step 2: Implement strict checkpoint reconstruction**

Build `SerumGateMinusConfig` from `model_config`, instantiate `SerumGateMinusModel`, call `load_state_dict(..., strict=True)`, move to the requested device, and call `eval()`. Verify the current CSV SHA-256 equals the checkpoint's `data_checksum` before loading embeddings.

- [ ] **Step 3: Write a failing end-to-end export test**

Create a tiny valid checkpoint and embeddings, run the export on CPU, and assert:

```text
interpretability/attention_per_sequence.csv
interpretability/attention_position_summary.csv
interpretability/attention_top_sites.csv
interpretability/attention_profile.png
interpretability/interpretation_config.json
```

Assert config fields include checkpoint checksum, dataset checksum, H3N2 filter, one-based aligned positions, unique-sequence aggregation, top K, pooling mode, and timestamp.

- [ ] **Step 4: Implement atomic orchestration**

Load and filter `all.csv`; build the registry; validate required embedding files; extract; summarize; select top 20; plot; and write config. Build all artifacts in a temporary sibling directory and rename it to `interpretability` only after validation. Overwrite only the known interpretation directory when explicitly authorized; never delete the run root.

- [ ] **Step 5: Add CLI and verify**

CLI arguments are `--checkpoint`, `--data-file`, `--embedding-dir`, `--output-dir`, `--type` defaulting to H3N2, `--device` defaulting to CPU, `--top-k` defaulting to 20, and `--overwrite`.

Run:

```bash
python -m pytest tests/test_export_whole_attention.py tests/test_serum_gate_attention.py -q
python experiments/serum_gate/export_whole_attention.py --help
```

Expected: all PASS; help exits 0.

- [ ] **Step 6: Commit Task 6**

```bash
git add experiments/serum_gate/export_whole_attention.py tests/test_export_whole_attention.py
git commit -m "feat: export whole-data SerumGate attention artifacts"
```

---

### Task 7: Integrated regression and real whole-dataset build

**Files:**
- Modify only in-scope files if a focused test exposes an implementation defect.
- Generate, but normally do not commit: `data/dataset/H1H3_HA1_v1.0/whole/all.csv`
- Generate, but normally do not commit: `data/dataset/H1H3_HA1_v1.0/whole/manifest.json`

**Interfaces:**
- Consumes: all completed tasks and the fixed seed-0 source files.
- Produces: verified software and the agreed real whole dataset; no production checkpoint yet.

- [ ] **Step 1: Run the focused suite**

```bash
python -m pytest \
  tests/test_build_whole_dataset.py \
  tests/test_serum_gate.py \
  tests/test_train_whole_minus_attention.py \
  tests/test_serum_gate_attention.py \
  tests/test_export_whole_attention.py \
  tests/test_infer_minus_checkpoint.py -q
```

Expected: all PASS with zero failures.

- [ ] **Step 2: Run a CPU one-epoch synthetic smoke workflow**

Use temporary fixtures to execute `build → train one epoch → reload → export`. Assert the final checkpoint loads strictly, all attention weights normalize, and all five interpretation artifacts exist. Keep this smoke data under pytest temporary paths only.

- [ ] **Step 3: Check diff scope before real data generation**

```bash
git status --short
git diff --check
git diff --stat
```

Expected: only planned code/test files plus pre-existing user changes; no existing source data or result file modified.

- [ ] **Step 4: Preflight and build the real whole dataset**

Record source SHA-256 values and confirm the destination is absent or empty. Then run:

```bash
python scripts/build_whole_dataset.py \
  --source-dir data/dataset/H1H3_HA1_v1.0/splited/20260717_164256/titer/seed_0 \
  --output-dir data/dataset/H1H3_HA1_v1.0/whole
```

Do not pass `--overwrite` unless the user separately authorizes replacing an existing generated dataset.

- [ ] **Step 5: Validate real output**

Reload all four CSVs and assert exact ordered schema equality, absence of `source_split`, output row count equal to the three-source sum, both subtypes present, manifest checksums reproducible, and source checksums unchanged. Report output path, size, total rows, subtype counts, and SHA-256.

- [ ] **Step 6: Run final regression checks**

Discover and run all in-scope tests:

```bash
rg --files tests | rg 'serum_gate|minus|attention|whole'
python -m pytest tests/test_serum_gate.py tests/test_infer_minus_checkpoint.py -q
git diff --check
```

Expected: all PASS and no whitespace errors.

- [ ] **Step 7: Commit only an actual integration fix if one was needed**

If no fix was needed, do not create an empty commit. If a planned file required a correction, stage that exact path and use:

```bash
git commit -m "test: verify pure-attention whole-data workflow"
```

## Production Run Handoff

The 200-epoch run is deliberately outside implementation. Before launching it, verify the selected project Python imports NumPy/pandas/torch/matplotlib, query current GPU memory, choose a device explicitly, validate every required H3N2 embedding, and confirm this output is absent or empty:

```text
results/H1H3_HA1_v1.0/20260717_164256/SerumGate-Minus-PureAttn-latent8/whole/subtype/H3N2
```

After training, require `checkpoints/final_model.pth`, finite in-sample loss, and `evaluation_status=train_all_no_holdout`. Then run the exporter and report the dataset/checkpoint checksums and all interpretation paths. Describe results as **attention-based site importance**, not held-out performance or causal proof.


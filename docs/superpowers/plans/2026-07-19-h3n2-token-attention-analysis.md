# H3N2 Token Attention Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a standalone analysis that extracts the trained all-attention model's 331-token pooling weights for every unique H3N2 virus HA sequence.

**Architecture:** A single CLI script streams the dataset to collect unique `seq_id_c` records, strictly loads the all-attention checkpoint, batches embedding inference through `encode_with_attention`, validates the resulting probability matrix, and atomically writes tables and plots. Unit and integration tests exercise deduplication, statistics, model extraction, and output generation without changing training/model code.

**Tech Stack:** Python, PyTorch, pandas, NumPy, matplotlib, pytest

## Global Constraints

- Do not commit to Git.
- Do not modify existing model or training files.
- Analyze only unique H3N2 virus-side `seq_id_c` sequences with equal weight.
- Preserve all 331 tokens, including token 0 and token 330 special tokens.
- Interpret outputs as pooling weights, not causal prediction attribution.

---

### Task 1: Core data and summary functions

**Files:**
- Create: `experiments/serum_gate/analyze_all_attention_tokens.py`
- Create: `tests/test_analyze_all_attention_tokens.py`

**Interfaces:**
- Produces: `collect_unique_virus_sequences(csv_path, subtype) -> pd.DataFrame`
- Produces: `summarize_attention(attention, token_count) -> tuple[pd.DataFrame, pd.DataFrame]`
- Produces: `validate_attention(attention, expected_rows, token_count) -> dict[str, float]`

- [ ] **Step 1: Write failing tests**

Test stable deduplication, occurrence counts, inconsistent sequence detection,
331-token mapping, summary statistics, top-1/top-5 counts, and probability-row
validation using small synthetic tables and arrays.

- [ ] **Step 2: Verify tests fail because the module does not exist**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/test_analyze_all_attention_tokens.py
```

Expected: collection error for missing `analyze_all_attention_tokens.py`.

- [ ] **Step 3: Implement minimal tested functions**

Stream CSV chunks using `Type` or `virusType`, keep one deterministic metadata
record per `seq_id_c`, validate one sequence per ID, and count occurrences.
Generate token labels `BOS`, `AA_1` through `AA_329`, and `EOS`. Compute count,
mean, median, population standard deviation, min, max, q05, q25, q75, q95,
top-1 count, and top-5 count.

- [ ] **Step 4: Verify Task 1 tests pass**

Run the same pytest command. Expected: all Task 1 tests pass.

### Task 2: Checkpoint extraction and atomic outputs

**Files:**
- Modify: `experiments/serum_gate/analyze_all_attention_tokens.py`
- Modify: `tests/test_analyze_all_attention_tokens.py`

**Interfaces:**
- Produces: `load_model(checkpoint_path, device) -> tuple[SerumGateMinusAllAttentionModel, dict]`
- Produces: `extract_attention(model, records, embedding_dir, device, batch_size, token_count) -> np.ndarray`
- Produces: `run_analysis(args: argparse.Namespace) -> dict[str, object]`

- [ ] **Step 1: Write failing integration test**

Build a tiny all-attention checkpoint and two `[331, hidden_size]` embedding
files in a temporary directory. Assert strict checkpoint loading, a finite
`[2, 331]` matrix whose rows sum to one, and creation of every designed CSV,
PNG, and JSON output.

- [ ] **Step 2: Verify the integration test fails on missing interfaces**

Run the targeted pytest file. Expected: failure naming `load_model`,
`extract_attention`, or `run_analysis`.

- [ ] **Step 3: Implement inference and output pipeline**

Use `model.eval()` and `torch.inference_mode()`. Load embeddings batchwise,
require shape `[331, model.config.hidden_size]`, use an all-true mask, and call
`model.score_model.ha_encoder.encode_with_attention`. Write outputs into a
temporary sibling directory, then rename it to the requested absent output
directory only after all validations and plots succeed.

- [ ] **Step 4: Implement CLI**

Required flags: `--checkpoint`, `--data-csv`, `--embedding-dir`, and
`--output-dir`. Optional flags: `--type H3N2`, `--device cuda:5`,
`--batch-size 16`, `--token-count 331`, `--expected-count 2611`, and
`--top-k 20`.

- [ ] **Step 5: Verify all analysis tests pass**

Run the targeted pytest file and expect zero failures.

### Task 3: Regression verification and full analysis

**Files:**
- Generate: `results/H1H3_HA1_v1.0/20260717_164256/SerumGate-Minus-AllAttention-latent8/whole/subtype/H3N2/attention_token_analysis/`

- [ ] **Step 1: Run existing SerumGate regression tests**

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/test_serum_gate.py
```

Expected: all existing SerumGate tests pass.

- [ ] **Step 2: Run the full analysis**

```bash
python experiments/serum_gate/analyze_all_attention_tokens.py \
  --checkpoint results/H1H3_HA1_v1.0/20260717_164256/SerumGate-Minus-AllAttention-latent8/whole/subtype/H3N2/checkpoints/best_model.pth \
  --data-csv data/dataset/H1H3_HA1_v1.0/splited/20260717_164256/whole/whole.csv \
  --embedding-dir data/embedding/files \
  --output-dir results/H1H3_HA1_v1.0/20260717_164256/SerumGate-Minus-AllAttention-latent8/whole/subtype/H3N2/attention_token_analysis \
  --type H3N2 \
  --device cuda:5 \
  --batch-size 16 \
  --token-count 331 \
  --expected-count 2611 \
  --top-k 20
```

Expected: reports 2,611 sequences, 331 tokens, and successful row-sum checks.

- [ ] **Step 3: Verify generated artifacts**

Confirm all seven designed files exist and are nonempty, CSV dimensions are
`2611 x metadata+331` and `331 x summary-columns`, all attention values are
finite/nonnegative, and every row sums to one within tolerance.

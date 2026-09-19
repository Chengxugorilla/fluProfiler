# Metric HA Antigenic Trainer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an independent trainer for the Metric HA antigenic-space observation model on the H1H3 serum split.

**Architecture:** Keep the trainer separate from HA-only baseline scripts under `experiments/metric_antigenic/`. Align `MetricHAAntigenicModel` with the technical formula, then train it through a fixed-split CSV loader that consumes existing HA embeddings and computes binary NA glycan mismatch from raw NA sequences.

**Tech Stack:** Python, PyTorch, pandas, pytest/unittest, existing fluProfiler embedding and metric utilities.

---

## File Structure

- Modify `src/fluprofiler/models/metric_antigenic_model.py`: add `passage_pair` and `subtype` fields, replace additive assay bias module with passage-pair fixed effect, and make NA glycan residual subtype-specific.
- Modify `src/fluprofiler/features/na_glycan_features.py`: add binary motif-set mismatch helper.
- Create `experiments/metric_antigenic/train_metric_ha.py`: fixed-split trainer, dataset, feature encoders, metrics, checkpointing, and CLI.
- Create `scripts/smoke_test_metric_ha.py`: CPU smoke run that builds synthetic data/embeddings and invokes the trainer for one epoch.
- Modify `tests/test_metric_antigenic_features.py`: add tests for binary NA glycan mismatch.
- Modify `tests/test_metric_antigenic_model.py`: add tests for passage-pair bias and subtype-specific NA residual.
- Create `tests/test_metric_antigenic_trainer.py`: trainer input validation and smoke-oriented helper tests.

## Task 1: NA Glycan Binary Feature

**Files:**
- Modify: `src/fluprofiler/features/na_glycan_features.py`
- Test: `tests/test_metric_antigenic_features.py`

- [ ] **Step 1: Write the failing tests**

Add tests showing identical NA head motif sets produce `0.0`, gain/loss produces `1.0`, and the score is symmetric.

- [ ] **Step 2: Run the feature tests**

Run: `pytest tests/test_metric_antigenic_features.py -q`
Expected: FAIL because `na_head_glycan_mismatch` does not exist.

- [ ] **Step 3: Implement the helper**

Add `na_head_glycan_mismatch(seq_a, seq_b, head_positions=None) -> float`. When `head_positions` is omitted, compare all motif starts in the input sequences.

- [ ] **Step 4: Verify the feature tests pass**

Run: `pytest tests/test_metric_antigenic_features.py -q`
Expected: PASS.

## Task 2: Model Formula Alignment

**Files:**
- Modify: `src/fluprofiler/models/metric_antigenic_model.py`
- Test: `tests/test_metric_antigenic_model.py`

- [ ] **Step 1: Write failing model tests**

Add tests proving:

- `b_assay` is selected by `passage_pair`.
- `lambda_nagly` differs by `subtype`.
- The previous `subtype`-ignored behavior no longer holds.

- [ ] **Step 2: Run model tests**

Run: `pytest tests/test_metric_antigenic_model.py -q`
Expected: FAIL on missing `passage_pair`/`subtype` behavior.

- [ ] **Step 3: Implement model changes**

Add config fields `passage_pair_vocab_size` and `subtype_vocab_size`; add batch fields `passage_pair` and `subtype`; replace `AssayBiasModule` usage with `nn.Embedding(passage_pair_vocab_size, 1)`; replace scalar `na_glycan_logit` with subtype embedding.

- [ ] **Step 4: Verify model tests pass**

Run: `pytest tests/test_metric_antigenic_model.py -q`
Expected: PASS.

## Task 3: Independent Trainer

**Files:**
- Create: `experiments/metric_antigenic/train_metric_ha.py`
- Test: `tests/test_metric_antigenic_trainer.py`

- [ ] **Step 1: Write failing trainer tests**

Add tests for split-file validation, required-column validation, embedding-file validation, passage/subtype vocabulary behavior, and batch construction metadata.

- [ ] **Step 2: Run trainer tests**

Run: `pytest tests/test_metric_antigenic_trainer.py -q`
Expected: FAIL because the trainer module does not exist.

- [ ] **Step 3: Implement trainer module**

Implement CLI arguments, fixed-split frame loading, feature encoders, dataset, dataloaders, embedding file validation, train/evaluate loops, metrics logging, TensorBoard logging, and checkpoint output.

- [ ] **Step 4: Verify trainer tests pass**

Run: `pytest tests/test_metric_antigenic_trainer.py -q`
Expected: PASS.

## Task 4: CPU Smoke Script

**Files:**
- Create: `scripts/smoke_test_metric_ha.py`

- [ ] **Step 1: Implement smoke script**

Build a temporary split with `train.csv`, `valid.csv`, `test.csv`, synthetic HA embeddings, and NA sequences, then invoke `train_metric_ha.py` for one CPU epoch.

- [ ] **Step 2: Run smoke script**

Run: `python scripts/smoke_test_metric_ha.py`
Expected: exit code 0 and a completed one-epoch run.

## Task 5: Full Verification

- [ ] **Step 1: Run focused tests**

Run: `pytest tests/test_metric_antigenic_features.py tests/test_metric_antigenic_model.py tests/test_metric_antigenic_trainer.py -q`
Expected: PASS.

- [ ] **Step 2: Provide full H1H3 command**

Use:

```bash
python experiments/metric_antigenic/train_metric_ha.py \
  --data-dir data/splited/v1/H1H3/H1H3__seed42__tr0.80_va0.10_te0.10__20260601_120452/serum \
  --embedding-dir data/reverse_test/embedding \
  --output-dir results/H1H3/metric_ha_serum \
  --device cuda:1 \
  --batch-size 64 \
  --learning-rate 8e-5 \
  --epochs 250 \
  --patience 10
```

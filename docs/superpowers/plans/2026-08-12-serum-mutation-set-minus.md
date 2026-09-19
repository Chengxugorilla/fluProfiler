# Serum Mutation Set Minus Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an isolated HA-only zero-shot serum route that explicitly encodes directed mutation sets with residue-distance-biased attention.

**Architecture:** A new model shares one low-rank HA site projection between reference-background and mutation-token paths. Directed mutation tokens receive absolute position, residue identity, presence, global reference background, and learned RBF distance attention before set pooling and a Minus-style conditional prediction head. A standalone trainer owns its batch/data logic and reuses stable HA split/metric helpers without modifying or monkey-patching existing code.

**Tech Stack:** Python 3.10, PyTorch, pandas, NumPy, unittest, fluProfiler Conda environment.

## Global Constraints

- Do not modify `src/fluprofiler/models/serum_gate_model.py`, `src/fluprofiler/models/serum_gate_minus_model.py`, or `experiments/serum_gate/train_zero_shot_minus.py`.
- Add only new model, entry-point, test, and documentation files.
- Run all Python and tests with `conda run -n fluProfiler`.
- The first version is HA-only: never require, validate, or load `seq_b`, `seq_d`, or NA embeddings.
- Derive mutation identity from aligned `serumHA` and `virusHA`, not embedding equality.
- Keep aligned-position masks separate from embedding-presence masks so insertions/deletions remain mutation tokens.
- Missing structural distances receive exactly zero structural bias.
- Preserve ALL-run per-subtype metrics and existing NLL/Huber/ranking semantics.

---

### Task 1: Distance-Biased Mutation Model

**Files:**
- Create: `src/fluprofiler/models/serum_mutation_set_model.py`
- Test: `tests/test_serum_mutation_set_model.py`

**Interfaces:**
- Consumes: `torch.Tensor` HA distance matrix and an independent `SerumMutationSetBatch`.
- Produces: `SerumMutationSetConfig`, `SerumMutationSetBatch`, `DistanceBiasedSelfAttention`, and `SerumMutationSetMinusModel.forward(batch) -> dict[str, torch.Tensor]`.

- [ ] **Step 1: Write failing model tests**

Test missing distance bias equals zero, near bias exceeds far bias, empty mutation forward remains finite, substitution direction and absolute position change representations, padded tokens do not affect pooling, and forward/backward produces finite `[B,Q]` outputs and gradients.

- [ ] **Step 2: Verify RED**

Run: `conda run -n fluProfiler python -m unittest tests.test_serum_mutation_set_model -v`

Expected: import failure because the model module does not exist.

- [ ] **Step 3: Implement minimal model**

Implement config/batch dataclasses, shared site projection, reference background, packed directed mutation tokens, null token, custom per-head RBF-biased self-attention, pre-norm block, dual set pooling, serum FiLM conditioning, score/log-variance head, Minus output, and compatible losses.

- [ ] **Step 4: Verify GREEN**

Run: `conda run -n fluProfiler python -m unittest tests.test_serum_mutation_set_model -v`

Expected: all model tests pass.

### Task 2: HA-Only Data Contract and Distance Loader

**Files:**
- Create: `experiments/serum_gate/train_serum_mutation_set.py`
- Test: `tests/test_train_serum_mutation_set.py`

**Interfaces:**
- Consumes: fixed-split CSV frames, HA embedding store, and CSV/TSV/NPY/NPZ distance paths.
- Produces: `HA_ONLY_REQUIRED_COLUMNS`, residue encoding/alignment helpers, `load_ha_distance_matrix`, `SerumMutationSetTaskDataset`, `collate_mutation_set_tasks`, and `move_mutation_batch`.

- [ ] **Step 1: Write failing trainer data tests**

Use literal synthetic fixtures to verify residue IDs, gap-aware coordinate alignment, directed mutation masks through the real collator, supported distance formats, invalid matrices, and frames without NA columns.

- [ ] **Step 2: Verify RED**

Run: `conda run -n fluProfiler python -m unittest tests.test_train_serum_mutation_set -v`

Expected: import failure because the trainer module does not exist.

- [ ] **Step 3: Implement minimal HA-only data path**

Implement a dataset that accesses only HA IDs/sequences and necessary metadata. Align embedding rows exactly to non-gap characters, retain aligned-position and embedding-presence masks, pad task tensors, construct the independent batch, and validate matrices without interpreting NaN as measured zero.

- [ ] **Step 4: Verify GREEN**

Run: `conda run -n fluProfiler python -m unittest tests.test_train_serum_mutation_set -v`

Expected: all data tests pass.

### Task 3: Independent Training CLI

**Files:**
- Extend only: `experiments/serum_gate/train_serum_mutation_set.py`
- Extend: `tests/test_train_serum_mutation_set.py`

**Interfaces:**
- Consumes: the HA-relevant existing CLI subset plus mutation dimensions and required `--ha-distance-matrix`.
- Produces: `parse_args`, `train_one_epoch`, `evaluate_model`, `run_training`, and `main`.

- [ ] **Step 1: Write failing CLI tests**

Verify the distance matrix is required, subtype filtering disables subtype features, nonpositive query chunk size disables chunking, and no NA-branch argument is exposed.

- [ ] **Step 2: Verify RED**

Run: `conda run -n fluProfiler python -m unittest tests.test_train_serum_mutation_set.TrainerCliTests -v`

Expected: failure because CLI functions are incomplete.

- [ ] **Step 3: Implement training and outputs**

Implement HA-only embedding requirements/loaders, config/model creation, AdamW/cosine scheduling, NLL/Huber training, evaluation frames, ALL subtype metrics, output-directory protection, run config, checkpoints, predictions, and refit behavior. Record distance metadata and all mutation-model settings.

- [ ] **Step 4: Verify GREEN**

Run: `conda run -n fluProfiler python -m unittest tests.test_train_serum_mutation_set.TrainerCliTests -v`

Expected: all CLI tests pass.

### Task 4: Regression and Smoke Verification

**Files:**
- Verify all four new files and the three protected existing files.

- [ ] **Step 1: Compile new files**

Run: `conda run -n fluProfiler python -m py_compile src/fluprofiler/models/serum_mutation_set_model.py experiments/serum_gate/train_serum_mutation_set.py tests/test_serum_mutation_set_model.py tests/test_train_serum_mutation_set.py`

- [ ] **Step 2: Run all new tests**

Run: `conda run -n fluProfiler python -m unittest tests.test_serum_mutation_set_model tests.test_train_serum_mutation_set -v`

- [ ] **Step 3: Run existing regression tests**

Run: `conda run -n fluProfiler python -m unittest tests.test_train_zero_shot_minus tests.test_run_standard_gradient_x_input -v`

- [ ] **Step 4: Run CPU smoke forward/backward**

Instantiate a small model with NaN distances, a substitution, a gap mutation, and an empty set; assert finite outputs/losses and nonzero gradients.

- [ ] **Step 5: Confirm protected-file isolation**

Report that this implementation introduced no edits to the three protected files. Do not claim they are clean because they were already untracked in the dirty worktree.

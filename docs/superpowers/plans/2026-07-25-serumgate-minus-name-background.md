# SerumGate-Minus Name Background Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional, additive `serumName` and `virusName` background biases to SerumGate-Minus training without adding a serum-virus interaction path.

**Architecture:** Build distinct train-only vocabularies and reserve ID 0 for empty/OOV names. The trainer passes IDs through `SerumGateBatch`; `SerumGateMinusModel` owns zero-initialized scalar embeddings and adds their sum only to its final distance mean.

**Tech Stack:** Python 3, PyTorch, pandas, unittest, Conda environment `fluProfiler`.

## Global Constraints

- Run Python commands with `conda run -n fluProfiler`.
- Keep the score difference `f(reference, reference) - f(reference, query)` unchanged.
- Keep the feature disabled by default, with old batches and checkpoints supported.
- Build vocabulary from `train` only and map missing/OOV values to 0.
- Never apply name backgrounds to `log_var` or an intermediate interaction feature.

---

### Task 1: Propagate train-only name IDs through the trainer

**Files:**

- Modify: `experiments/serum_gate/train_zero_shot_minus.py`
- Modify: `src/fluprofiler/models/serum_gate_model.py`
- Test: `tests/test_serum_gate.py`

**Interfaces:**

- Produces `NameVocabs(serum_to_id: dict[str, int], virus_to_id: dict[str, int])`.
- Adds optional `SerumGateBatch.serum_name: Tensor[B]` and `virus_name: Tensor[B, Q]`.

- [ ] **Step 1: Write failing dataset and CPU-collator tests**

```python
def test_name_vocabs_use_train_only_and_map_unseen_names_to_zero(self):
    train = pd.DataFrame([_row(serum_name="seen_serum", virus_name="seen_virus")])
    test = pd.DataFrame([_row(serum_name="new_serum", virus_name="new_virus")])
    names = self.module.build_name_vocabs(train)
    vocabs = self.module.build_serum_gate_vocabs({"train": train})
    item = self.module.SerumGateTaskDataset(test, vocabs, name_vocabs=names)[0]
    self.assertEqual(item["serum_name"].item(), 0)
    self.assertTrue(torch.equal(item["virus_name"], torch.tensor([0])))
```

```python
def test_collate_preserves_serum_and_query_name_ids(self):
    frame = pd.DataFrame([_row(serum_name="serum_a", virus_name="virus_a")])
    names = self.module.build_name_vocabs(frame)
    vocabs = self.module.build_serum_gate_vocabs({"train": frame})
    item = self.module.SerumGateTaskDataset(frame, vocabs, name_vocabs=names)[0]
    embeddings = {"matrix_ref_ha": torch.ones(2, 3), "matrix_test_ha": torch.ones(2, 3)}
    batch, _ = self.module.collate_serum_gate_tasks([item], embeddings)
    self.assertTrue(torch.equal(batch.serum_name, torch.tensor([1])))
    self.assertTrue(torch.equal(batch.virus_name, torch.tensor([[1]])))
```

- [ ] **Step 2: Run the focused test and observe the required failure**

Run: `conda run -n fluProfiler pytest tests/test_serum_gate.py -k 'name_vocabs or collate_preserves_serum' -v`

Expected: FAIL because `build_name_vocabs`, the dataset argument, or batch fields do not exist.

- [ ] **Step 3: Implement minimal name-ID transport**

Add `NameVocabs` and `build_name_vocabs(train_frame)`. Require both name columns only when enabled; map nonempty train values to IDs beginning at 1. Add optional name fields to `SerumGateBatch`. Add `name_vocabs=None` to `SerumGateTaskDataset` and encode its task-level serum name plus each query virus name. Populate both fields in CPU/GPU collators and `move_batch`. Add `--use-name-backgrounds` as a `BooleanOptionalAction`, default false. When true, build vocabularies from `frames["train"]`, pass them to every loader, and record mappings and flag in `run_config.json`.

- [ ] **Step 4: Run the focused test and observe success**

Run: `conda run -n fluProfiler pytest tests/test_serum_gate.py -k 'name_vocabs or collate_preserves_serum' -v`

Expected: PASS; unseen values are ID 0 and collated shapes are `(1,)` and `(1, 1)`.

- [ ] **Step 5: Commit the isolated transport change**

Run: `git add experiments/serum_gate/train_zero_shot_minus.py src/fluprofiler/models/serum_gate_model.py tests/test_serum_gate.py && git commit -m "feat: pass SerumGate name background ids"`

### Task 2: Add independent additive biases to SerumGate-Minus

**Files:**

- Modify: `src/fluprofiler/models/serum_gate_minus_model.py`
- Test: `tests/test_serum_gate.py`

**Interfaces:**

- Consumes `use_name_backgrounds: bool`, `serum_name_vocab_size: int`, and `virus_name_vocab_size: int` in `SerumGateMinusConfig`.
- Produces `mean = score_difference + serum_bias + virus_bias`; it does not change `log_var`.

- [ ] **Step 1: Write the failing direct-behavior test**

```python
def test_name_backgrounds_add_only_to_mean_not_log_variance(self):
    config = SerumGateMinusConfig(hidden_size=3, latent_dim=2, theta_dim=2, use_name_backgrounds=True, serum_name_vocab_size=3, virus_name_vocab_size=4)
    model = SerumGateMinusModel(config).eval()
    batch = _make_minus_batch(serum_name=torch.tensor([1]), virus_name=torch.tensor([[2, 3]]))
    baseline = model(batch)
    with torch.no_grad():
        model.serum_name_bias.weight[1, 0] = 0.75
        model.virus_name_bias.weight[2, 0] = -0.25
        model.virus_name_bias.weight[3, 0] = 0.50
    shifted = model(batch)
    self.assertTrue(torch.allclose(shifted["mean"] - baseline["mean"], torch.tensor([[0.50, 1.25]])))
    self.assertTrue(torch.allclose(shifted["log_var"], baseline["log_var"]))
```

- [ ] **Step 2: Run it to prove it fails before production support exists**

Run: `conda run -n fluProfiler pytest tests/test_serum_gate.py -k name_backgrounds -v`

Expected: FAIL with an unsupported config keyword or missing embedding table.

- [ ] **Step 3: Implement the smallest additive model path**

Add the three config fields. When enabled, validate both vocabulary sizes are positive; create `nn.Embedding(vocab_size, 1, padding_idx=0)` tables and zero their weights. In `SerumGateMinusModel.forward`, require both batch name tensors, then add a broadcast serum scalar and per-query virus scalar to `mean` after score subtraction. Do not change self/query scoring, losses other than their use of the adjusted mean, or `log_var`. Disabled mode neither requires IDs nor constructs tables.

- [ ] **Step 4: Run the focused test and observe success**

Run: `conda run -n fluProfiler pytest tests/test_serum_gate.py -k name_backgrounds -v`

Expected: PASS; hand-set biases add `[0.50, 1.25]` and variance is identical.

- [ ] **Step 5: Commit model behavior**

Run: `git add src/fluprofiler/models/serum_gate_minus_model.py tests/test_serum_gate.py && git commit -m "feat: add additive SerumGate name backgrounds"`

### Task 3: Cover CLI defaults, validation, and regression compatibility

**Files:**

- Modify: `tests/test_serum_gate.py`
- Modify: `experiments/serum_gate/train_zero_shot_minus.py` only if final test exposes incomplete wiring.

**Interfaces:**

- Verifies disabled mode accepts batches without name IDs.
- Verifies enabled mode records the exact train-only mappings.

- [ ] **Step 1: Write failing validation and disabled-mode tests**

```python
def test_name_background_flag_is_disabled_by_default(self):
    args = self.module.parse_args(["--data-dir", "/tmp", "--embedding-dir", "/tmp", "--output-dir", "/tmp/out"])
    self.assertFalse(args.use_name_backgrounds)

def test_name_vocabs_reject_missing_columns_when_enabled(self):
    frame = pd.DataFrame([_row()]).drop(columns=["virusName"])
    with self.assertRaisesRegex(ValueError, "virusName"):
        self.module.build_name_vocabs(frame)
```

- [ ] **Step 2: Run the focused tests and observe a failure if wiring is incomplete**

Run: `conda run -n fluProfiler pytest tests/test_serum_gate.py -k 'name_background_flag or name_vocabs_rejects' -v`

Expected: FAIL until parser default and missing-column validation are implemented.

- [ ] **Step 3: Complete metadata and compatibility wiring**

Pass the flag and `len(mapping) + 1` values to `SerumGateMinusConfig`. Persist `use_name_backgrounds`, `serum_name_to_id`, and `virus_name_to_id` in `run_config.json`; use `None` mappings when disabled. Add a disabled-mode test that runs `SerumGateMinusModel` without name batch fields and verifies it returns mean and log variance.

- [ ] **Step 4: Run full relevant verification**

Run: `conda run -n fluProfiler pytest tests/test_serum_gate.py -v`

Expected: PASS with zero failures.

Run: `conda run -n fluProfiler pytest tests/test_train_fixed_split.py -v`

Expected: PASS with zero failures.

- [ ] **Step 5: Inspect and commit integration coverage**

Run: `git diff --check && git status --short && git add experiments/serum_gate/train_zero_shot_minus.py tests/test_serum_gate.py && git commit -m "test: cover SerumGate name background training"`

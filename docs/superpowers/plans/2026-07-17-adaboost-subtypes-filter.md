# AdaBoost Subtype Filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Require a Nextflu-style `--subtypes` argument and filter fixed split frames before AdaBoost training and evaluation.

**Architecture:** Keep the existing fixed-split CLI and model-per-subtype flow. Add small parsing, filtering, and validation helpers so CLI behavior is directly testable without running the full dataset.

**Tech Stack:** Python 3.10, argparse, pandas, unittest, scikit-learn

## Global Constraints

- Accept `--subtypes H3N2` and `--subtypes H1N1 H3N2`.
- Default to an empty list and fail if the option is omitted.
- Never create a new external train/valid/test split.
- Filter train, valid, and test before row counts and train-valid merging.
- Validate requested subtypes against the effective training data.
- Use a tiny fixture and `fast=True` for verification.

---

### Task 1: Add explicit subtype selection

**Files:**
- Modify: `experiments/benchmark_pairwise/train_adaboost_fixed_split.py:283-359`
- Test: `tests/test_pairwise_adaboost_baseline.py`

**Interfaces:**
- Consumes: fixed split frames as `dict[str, pd.DataFrame]` containing a `Type` column.
- Produces: `parse_args(argv: list[str] | None = None) -> argparse.Namespace`, `filter_frames_by_subtypes(frames: dict[str, pd.DataFrame], subtypes: list[str]) -> dict[str, pd.DataFrame]`, and `validate_training_subtypes(frames: dict[str, pd.DataFrame], subtypes: list[str], merge_valid_into_train: bool) -> None`.

- [ ] **Step 1: Write failing parser and filtering tests**

```python
def test_parse_args_requires_subtypes(self):
    with self.assertRaises(SystemExit):
        self.module.parse_args(["--data-dir", "split", "--output-dir", "out"])

def test_parse_args_accepts_nextflu_style_subtypes(self):
    args = self.module.parse_args([
        "--data-dir", "split", "--output-dir", "out",
        "--subtypes", "H1N1", "H3N2",
    ])
    self.assertEqual(args.subtypes, ["H1N1", "H3N2"])

def test_filter_frames_by_subtypes_filters_every_split(self):
    frames = {
        name: pd.DataFrame([_row(subtype="H1N1"), _row(subtype="H3N2")])
        for name in ("train", "valid", "test")
    }
    filtered = self.module.filter_frames_by_subtypes(frames, ["H3N2"])
    self.assertTrue(all(frame["Type"].tolist() == ["H3N2"] for frame in filtered.values()))

def test_validate_training_subtypes_rejects_missing_effective_train_subtype(self):
    frames = {
        "train": pd.DataFrame([_row(subtype="H1N1")]),
        "valid": pd.DataFrame([_row(subtype="H1N1")]),
        "test": pd.DataFrame([_row(subtype="H3N2")]),
    }
    with self.assertRaisesRegex(ValueError, "H3N2"):
        self.module.validate_training_subtypes(frames, ["H3N2"], merge_valid_into_train=True)
```

- [ ] **Step 2: Run tests and verify they fail**

Run: `/home/chenyh/miniconda3/envs/fluProfiler/bin/python -m unittest tests.test_pairwise_adaboost_baseline -v`

Expected: new tests fail because `parse_args` does not accept argv and filtering helpers do not exist.

- [ ] **Step 3: Implement minimal CLI and early filtering**

```python
def filter_frames_by_subtypes(frames, subtypes):
    selected = set(subtypes)
    return {
        name: frame.loc[frame["Type"].astype(str).isin(selected)].reset_index(drop=True)
        for name, frame in frames.items()
    }

def validate_training_subtypes(frames, subtypes, merge_valid_into_train):
    training_frames = [frames["train"]]
    if merge_valid_into_train:
        training_frames.append(frames["valid"])
    available = set(pd.concat(training_frames, ignore_index=True)["Type"].dropna().astype(str))
    missing = [subtype for subtype in subtypes if subtype not in available]
    if missing:
        raise ValueError(
            f"Requested subtype(s) missing from effective training data: {', '.join(missing)}; "
            f"available: {', '.join(sorted(available)) or 'none'}"
        )
```

Add `parser.add_argument("--subtypes", nargs="+", default=[])`, reject an empty list with `parser.error`, validate before filtering, filter all frames before counts, then retain the requested order for model training and `run_config.json`.

- [ ] **Step 4: Run focused tests and verify they pass**

Run: `/home/chenyh/miniconda3/envs/fluProfiler/bin/python -m unittest tests.test_pairwise_adaboost_baseline -v`

Expected: all AdaBoost baseline tests pass.

- [ ] **Step 5: Run a tiny end-to-end CLI smoke test**

Create temporary train/valid/test CSVs containing a few H3N2 rows, invoke the CLI with `--subtypes H3N2 --merge-valid-into-train --sequence-start 0 --sequence-end 5 --fast`, and verify `run_config.json` records only H3N2 and the merged training row count.

- [ ] **Step 6: Commit when repository identity is configured**

```bash
git add experiments/benchmark_pairwise/train_adaboost_fixed_split.py tests/test_pairwise_adaboost_baseline.py docs/superpowers/plans/2026-07-17-adaboost-subtypes-filter.md
git commit -m "feat: filter adaboost runs by subtype"
```

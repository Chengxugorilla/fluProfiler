# Zero-Shot Serum Meta-Learning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build SerumGate for de novo antiserum extrapolation under a serum-level zero-shot setting.

**Architecture:** Add an independent `experiments/serum_gate/` trainer/evaluator that groups pair rows into serum tasks and trains a new `SerumGateModel` in `src/fluprofiler/models/serum_gate_model.py`. The first version has empty support sets and predicts all query rows from `theta0` only.

**Tech Stack:** Python, PyTorch, pandas, numpy, scipy/sklearn metrics already used by fluProfiler, pytest/unittest.

---

## File Structure

- Create `src/fluprofiler/models/serum_gate_model.py`: HA pooling, serum prior encoder, FiLM predictor, zero-shot model, Huber/NLL losses.
- Create `experiments/serum_gate/train_zero_shot.py`: task dataset, strict group split, embedding validation/loading, training loop, evaluation metrics, CLI.
- Create `experiments/serum_gate/evaluate_zero_shot.py`: checkpoint evaluation wrapper.
- Create `scripts/smoke_test_serum_gate.py`: synthetic CPU smoke run.
- Create `tests/test_serum_gate.py`: TDD coverage for grouping, split leakage guard, model forward, metrics, and smoke helpers.

## Task 1: Tests for Serum Task Data and Metrics

- [ ] Write tests in `tests/test_serum_gate.py` for `make_task_key`, `strict_group_resplit`, `SerumGateTaskDataset`, and `serum_regression_metrics`.
- [ ] Run `pytest tests/test_serum_gate.py -q` and confirm it fails because `experiments/serum_gate/train_zero_shot.py` does not exist.
- [ ] Implement the data and metric helpers in `experiments/serum_gate/train_zero_shot.py`.
- [ ] Re-run `pytest tests/test_serum_gate.py -q` and confirm the data/metric tests pass.

## Task 2: Tests for Zero-Shot Model

- [ ] Add tests that instantiate `SerumGateModel` with tiny tensors and assert prediction/log variance shapes plus finite Huber and NLL losses.
- [ ] Run the model test and confirm it fails because `src/fluprofiler/models/serum_gate_model.py` does not exist.
- [ ] Implement `SerumGateConfig`, `SerumGateBatch`, `SerumGatePriorEncoder`, `SerumGateConditionedPredictor`, and `SerumGateModel`.
- [ ] Re-run the model test and confirm it passes.

## Task 3: Trainer and Evaluator

- [ ] Add tests for `run_training` using synthetic CSV rows and synthetic HA embeddings.
- [ ] Run the smoke-oriented test and confirm it fails because training output is missing.
- [ ] Implement `run_training`, checkpoint saving, prediction CSV writing, and the CLI in `experiments/serum_gate/train_zero_shot.py`.
- [ ] Implement `experiments/serum_gate/evaluate_zero_shot.py` as a thin checkpoint loader around the shared helpers.
- [ ] Re-run focused tests and confirm they pass.

## Task 4: Smoke Script and Verification

- [ ] Create `scripts/smoke_test_serum_gate.py` to build a temporary synthetic task dataset and call the trainer for one CPU epoch.
- [ ] Run `python scripts/smoke_test_serum_gate.py`.
- [ ] Run `pytest tests/test_serum_gate.py -q`.
- [ ] Review `git diff` to confirm only intended files changed.

## Self-Review

The plan covers the approved zero-shot scope and leaves few-shot calibrators out of implementation. There are no `TBD` or deferred implementation placeholders in the zero-shot path. Later few-shot work should extend `SerumGateTaskDataset` support selection and add a `SupportCalibrator` without changing the zero-shot checkpoint format.

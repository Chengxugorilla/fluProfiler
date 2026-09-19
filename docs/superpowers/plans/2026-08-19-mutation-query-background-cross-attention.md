# Mutation Query Background Cross-Attention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an isolated MutationSet model where mutation tokens cross-attend to valid query HA site tokens before pooling.

**Architecture:** Subclass the existing model in a new file, preserving its mutation encoder and self-attention. Project per-site query representations to mutation dimension, apply masked cross-attention after the existing mutation blocks, and add it through a zero-initialized scalar residual gate.

**Tech Stack:** Python, PyTorch, unittest.

**Spec:** `docs/superpowers/specs/2026-08-19-mutation-query-background-cross-attention-design.md`

## Global Constraints

- Create independent model and training-entry files only; do not edit existing MutationSet, FusedFiLM, or NameResidual source files.
- Preserve mutation self-attention and its distance bias exactly.
- Query site mask is `query_embedding_mask`; invalid sites have zero weight.
- Cross-attention has no structure-distance bias and gate initializes at exactly zero.
- Return cross-attention weights but do not write them to prediction CSVs.

---

### Task 1: Cross-attention model and tests

**Files:**
- Create: `src/fluprofiler/models/serum_mutation_set_query_background_model.py`
- Create: `tests/test_serum_mutation_set_query_background_model.py`

- [ ] Write failing tests for zero-gate baseline equality, masked query-site weights, enabled-context sensitivity, and gradient flow.
- [ ] Implement a masked multi-head mutation-to-query attention module and a `SerumMutationSetMinusQueryBackgroundModel` subclass.
- [ ] Run `conda run --no-capture-output -n fluProfiler python -m unittest tests/test_serum_mutation_set_query_background_model.py`.

### Task 2: Isolated training entry

**Files:**
- Create: `experiments/serum_gate/train_serum_mutation_set_query_background.py`
- Create: `tests/test_train_serum_mutation_set_query_background.py`

- [ ] Write a failing CLI test for cross-attention options.
- [ ] Add an entry that substitutes only the independent model class and inherits existing data, metrics, and checkpoint behavior.
- [ ] Run focused trainer tests and existing MutationSet tests to verify no source regression.

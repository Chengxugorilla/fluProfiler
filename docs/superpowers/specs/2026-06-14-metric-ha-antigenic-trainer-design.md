# Metric HA Antigenic Trainer Design

Date: 2026-06-14

## Goal

Add a non-destructive training branch for the Metric HA antigenic-space observation model on the H1H3 serum split. The branch must train the new model from the existing split CSVs and HA embedding cache without changing the default HA-only baseline trainer.

## Context

The repository already contains a partial metric model in `src/fluprofiler/models/metric_antigenic_model.py`, NA glycan utilities in `src/fluprofiler/features/na_glycan_features.py`, and fixed-split HA-only training in `experiments/HA_only/train_fixed_split.py`. The H1H3 serum split has the fields needed by the technical document:

- `seq_id_a` and `seq_id_c` identify serum/reference HA and test-virus HA embeddings.
- `seq_b` and `seq_d` provide serum/reference NA and test-virus NA sequences.
- `serumPassCat` and `virusPassCat` provide passage/cultivar labels.
- `Type` provides subtype labels such as `H1N1` and `H3N2`.
- `label` is the HI-derived antigenic distance target.

## Chosen Approach

Create an independent metric trainer under `experiments/metric_antigenic/`. This is the safest path because it leaves `experiments/HA_only/train_fixed_split.py` and existing `model_impl` behavior untouched, while still reusing shared project utilities for embedding loading, GPU cache, metrics, logging, and optimizer setup where appropriate.

Two alternatives were rejected:

- Modify `train_fixed_split.py`: less code, but it would make the HA-only baseline entrypoint carry metric-specific data and loss logic.
- Build the complete v2 recipe framework first: cleaner long-term, but too broad for the current goal of training the documented model on H1H3 serum split.

## Architecture

The new trainer will expose a CLI similar to `train_fixed_split.py`:

```text
python experiments/metric_antigenic/train_metric_ha.py \
  --data-dir data/splited/v1/H1H3/H1H3__seed42__tr0.80_va0.10_te0.10__20260601_120452/serum \
  --embedding-dir data/reverse_test/embedding \
  --output-dir results/H1H3/metric_ha_serum
```

The trainer will:

1. Load `train.csv`, `valid.csv`, and `test.csv` from the fixed serum split.
2. Validate required columns before training starts.
3. Load only HA embeddings required by `seq_id_a` and `seq_id_c`.
4. Compute NA glycan mismatch from `seq_b` and `seq_d` as a binary feature.
5. Encode passage labels into stable categorical ids.
6. Encode subtype labels into stable ids for subtype-specific NA residual weights.
7. Train `MetricHAAntigenicModel` with its decomposed output components.
8. Write `run_config.json`, `metrics.jsonl`, `log.txt`, TensorBoard events, and `checkpoints/best_model.pth`.

## Model Adjustments

`MetricHAAntigenicModel` will be updated to align with the technical document:

- HA latent distance remains `s_HA * ||zA_HA - zB_HA||`.
- Serum response scale remains `softplus(r0 + linear_z(zA_HA) + linear_p(pA))`.
- Assay bias becomes a passage-pair fixed effect: one scalar per `(serumPassCat, virusPassCat)` pair.
- NA glycan residual becomes subtype-specific: `lambda_NAgly_subtype[subtype] * S_NAgly_AB`.
- `S_NAgly_AB` is binary: `1.0` when the serum/reference NA head glycan motif set differs from the test-virus NA head glycan motif set, else `0.0`.

The existing model output keys such as `pred`, `d_ha`, `rho_ha`, `b_assay`, `s_nagly`, `r_na`, and `lambda_nagly` will remain available so current diagnostics and tests can inspect the decomposition.

## Data Flow

Each dataset item will return lightweight identifiers and scalar features:

```text
matrix_seq_id_a, matrix_seq_id_c,
serum_passage_id, test_passage_id, passage_pair_id,
subtype_id, s_nagly, label
```

At batch time, the trainer will use the existing `GpuEmbeddingCache` and `generate_matrix_on_device` utility to materialize padded HA matrices and masks. It then builds `MetricAntigenicBatch`:

```text
serum_ha, virus_ha, serum_ha_mask, virus_ha_mask,
serum_passage, test_passage, passage_pair,
subtype, s_nagly, labels
```

## Error Handling

The trainer must fail before training when:

- `train.csv`, `valid.csv`, or `test.csv` is missing.
- Any required column is absent.
- Any required HA embedding file is missing.
- A passage or subtype label appears in validation/test that was not included in the training vocabulary and cannot be mapped to `unknown`.

Passage labels are normalized so variants like `EGG`, `<EGG>`, `CELL`, `<CELL>`, blank, and missing values map consistently.

## Testing

Tests will be added before implementation:

- NA glycan binary mismatch returns `0.0` for identical motif sets and `1.0` for gain/loss.
- The metric model uses a passage-pair scalar bias rather than separate passage effects.
- The metric model applies different NA residual coefficients by subtype.
- The trainer validates required split files and columns.
- The trainer reports missing HA embeddings before training.
- A CPU smoke test can create synthetic split CSVs and embeddings, run one epoch, and produce metrics/checkpoint artifacts.

## Full Training Target

The default full training target is the existing H1H3 serum split:

```text
data/splited/v1/H1H3/H1H3__seed42__tr0.80_va0.10_te0.10__20260601_120452/serum
```

with HA embeddings from:

```text
data/reverse_test/embedding
```

The implementation will first run a small CPU smoke test. Full H1H3 training can then be launched with the same independent trainer and a non-empty output directory guard to prevent overwriting previous runs.

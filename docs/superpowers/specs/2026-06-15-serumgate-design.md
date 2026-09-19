# Zero-Shot Serum Meta-Learning Design

Date: 2026-06-15

## Goal

Add SerumGate for de novo antiserum extrapolation under a serum-level zero-shot setting. The first implementation treats each antiserum as a task and predicts the full cross-reactivity profile from the serum/reference virus prior only, without few-shot support measurements.

## Scope

This MVP implements zero-shot only. Few-shot calibration, Set Transformer support encoders, and active anchor selection are intentionally deferred, but the task data structure keeps an empty `support` slot so those features can be added without changing the model boundary.

## Serum Task Definition

The default serum task key is:

```text
seq_id_a, seq_id_b, serumPassCat
```

This captures the reference HA, reference NA, and serum passage condition. The training script exposes `--serum-task-cols` so later experiments can include columns such as `serumName`, `serumDate`, `serumIslID`, or `sheet`.

Each task contains one serum/reference virus and many query test viruses. Query rows keep the existing split columns, including `seq_id_c`, `seq_id_d`, `virusPassCat`, `Type`, `virusDate`, `seq_b`, `seq_d`, `serumHA`, `virusHA`, and `label`.

## Data Flow

The new branch lives beside, not inside, the existing pairwise metric trainer:

```text
fixed split CSVs
-> optional strict serum-task split by task key
-> SerumGateTaskDataset
-> episodic DataLoader with zero support rows
-> SerumGateModel
-> query loss and serum-level metrics
```

The loader validates required columns and HA embedding files before training. It can either consume the existing train/valid/test CSVs or rebuild strict group splits from their union using `--strict-group-resplit`. The resplit mode is recommended because the current H1H3 serum valid split is row-random within train-pool serum groups.

## Model

`SerumGateModel` has three components:

1. `HAPoolEncoder`: pools foundation-model HA token embeddings into task-aligned vectors for reference and test viruses.
2. `SerumGatePriorEncoder`: combines reference HA vector with serum metadata embeddings and scalar features to produce `theta0`.
3. `SerumGateConditionedPredictor`: builds pair features from `z_ref`, `z_test`, `z_test - z_ref`, `abs(z_test - z_ref)`, passage pair, subtype, NA glycan mismatch, and optional HA mismatch summary. A FiLM-conditioned MLP uses `theta0` to output `mean` and `log_var`.

The model returns prediction mean, log variance, Gaussian NLL, Huber loss, and intermediate task latent values. The trainer can optimize either Huber or Gaussian NLL.

## Evaluation

Evaluation is serum-task aware and reports:

- pooled MAE and MSE across all query rows
- pooled Pearson and Spearman
- per-serum MAE mean and median
- within-serum Pearson and Spearman averaged over tasks with enough variation
- serum-level bias mean and absolute mean
- Gaussian NLL
- 80% and 95% prediction interval coverage

The test split must have no task-key overlap with train. If overlap is detected, evaluation fails unless the user explicitly disables the leakage guard.

## Outputs

The trainer writes:

- `run_config.json`
- `metrics.jsonl`
- `log.txt`
- `checkpoints/best_model.pth`
- `predictions_<split>.csv`

The evaluation script can load a saved checkpoint and write the same metric/prediction format for a chosen split.

## Tests

Tests cover:

- task-key construction from `seq_id_a,seq_id_b,serumPassCat`
- strict group resplitting with no train/valid/test task overlap
- dataset grouping and zero-support task item shape
- model forward shape and finite Huber/NLL loss
- prediction interval coverage metric
- a CPU synthetic smoke run that produces metrics and a checkpoint

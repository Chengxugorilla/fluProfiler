#!/usr/bin/env bash
set -euo pipefail

# Edit this parameter block for each Metric HA artificial-warmup experiment.
data_dir="data/H1H3/splited/v1/H1H3__seed42__tr0.80_va0.10_te0.10__20260601_120452/serum"
embedding_dir="data/embedding/files"
artificial_csv="$data_dir/artificial_data.csv"
output_dir="results/H1H3/metric_ha_serum_artificial_warmup5"

device="cuda:4"
batch_size=64
learning_rate=8e-5
epochs=200
patience=10
gpu_cache_gb=20
seed=42
latent_dim=128
na_lambda_max=0.25
artificial_warmup_epochs=5

# No need to edit below this line.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python scripts/build_metric_artificial_data.py \
  --data-dir "$data_dir" \
  --output-csv "$artificial_csv" \
  --overwrite

python experiments/metric_antigenic/train_metric_ha.py \
  --data-dir "$data_dir" \
  --embedding-dir "$embedding_dir" \
  --output-dir "$output_dir" \
  --device "$device" \
  --batch-size "$batch_size" \
  --learning-rate "$learning_rate" \
  --epochs "$epochs" \
  --patience "$patience" \
  --gpu-cache-gb "$gpu_cache_gb" \
  --seed "$seed" \
  --latent-dim "$latent_dim" \
  --na-lambda-max "$na_lambda_max" \
  --artificial-train-csv "$artificial_csv" \
  --artificial-warmup-epochs "$artificial_warmup_epochs"

#!/usr/bin/env bash
set -euo pipefail

# Edit this parameter block for each HA-only fixed-split experiment.
conda_env=fluProfiler
split_type=serum

data_dir=/home/chenyh/workspace/fluProfiler/data/H1H3/splited/v1/H1H3__seed42__tr0.80_va0.10_te0.10__20260601_120452/${split_type}
embedding_dir=/home/chenyh/workspace/fluProfiler/data/embedding/files
output_dir=/home/chenyh/workspace/fluProfiler/results/H1H3/${split_type}
model_config=/home/chenyh/workspace/fluProfiler/configs/config_dict.json
model_impl=distance

# Smoke-test defaults. For the formal run, set epochs=250, sample_limit=-1,
# and choose a new output_dir such as .../titer_full.
batch_size=16
learning_rate=1e-4
epochs=200
patience=30
device=cuda:4
gpu_cache_gb=24
sample_limit=-1       # -1 means use every row.
seed=42
add_special_token=true
use_lr_schedule=true

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

extra_args=()
if [[ "$add_special_token" == true ]]; then
  extra_args+=(--add-special-token)
else
  extra_args+=(--no-add-special-token)
fi
if [[ "$use_lr_schedule" == true ]]; then
  extra_args+=(--use-lr-schedule)
else
  extra_args+=(--no-use-lr-schedule)
fi

cd "$repo_root"
conda run -n "$conda_env" python experiments/HA_only/train_fixed_split.py \
  --data-dir "$data_dir" \
  --embedding-dir "$embedding_dir" \
  --output-dir "$output_dir" \
  --model-config "$model_config" \
  --model-impl "$model_impl" \
  --batch-size "$batch_size" \
  --learning-rate "$learning_rate" \
  --epochs "$epochs" \
  --patience "$patience" \
  --device "$device" \
  --gpu-cache-gb "$gpu_cache_gb" \
  --sample-limit "$sample_limit" \
  --seed "$seed" \
  "${extra_args[@]}"

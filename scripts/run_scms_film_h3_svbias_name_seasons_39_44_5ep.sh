#!/usr/bin/env bash

# Train SCMS-FiLM-H3 with serum/virus name offsets (SVBias) for seasons 39--44.
# GPU assignment: 39,40 -> GPU 4; 41,42 -> GPU 5; 43,44 -> GPU 6.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

DATASET="H1H3_HA1_v1.0"
SPLIT_VERSION="20260717_164256"
RUN_NAME="SCMS-FiLM-H3+SVBias-name-5ep"
LOG_DIR="results/${DATASET}/${SPLIT_VERSION}/${RUN_NAME}/launch_logs"

mkdir -p "$LOG_DIR"

launch() {
  local season="$1"
  local gpu="$2"
  local output_dir="results/${DATASET}/${SPLIT_VERSION}/${RUN_NAME}/season/${season}/subtype/H3N2_seed42"
  local log_file="${LOG_DIR}/season_${season}.log"

  if [[ -e "$output_dir" ]]; then
    echo "Refusing to overwrite existing output: $output_dir" >&2
    return 1
  fi

  CUDA_VISIBLE_DEVICES="$gpu" nohup conda run --no-capture-output -n fluProfiler python \
    experiments/serum_gate/train_serum_mutation_set.py \
    --data-dir "data/dataset/${DATASET}/splited/${SPLIT_VERSION}/season/${season}" \
    --embedding-dir data/embedding/files \
    --ha-distance-matrix ha1_distance_no_bias_329.npy \
    --output-dir "$output_dir" \
    --type H3N2 \
    --refit-train-valid \
    --batch-size 1 \
    --max-queries-per-task 32 \
    --epochs 5 \
    --save-epoch 5 \
    --learning-rate 1e-4 \
    --weight-decay 0.01 \
    --lr-scheduler cosine \
    --lr-min 1e-6 \
    --site-dim 64 \
    --background-dim 64 \
    --mutation-dim 128 \
    --position-dim 32 \
    --amino-acid-dim 16 \
    --presence-dim 4 \
    --theta-dim 128 \
    --mutation-attention-heads 4 \
    --mutation-attention-layers 1 \
    --mutation-ffn-dim 256 \
    --attention-dropout 0.1 \
    --attention-alpha-init 0.05 \
    --attention-tau-init 8.0 \
    --predictor-hidden-dim 256 \
    --predictor-dropout 0.1 \
    --zero-init-film \
    --direct-background \
    --no-use-background-to-mutation \
    --task-bias-loss-weight 0.1 \
    --use-output-identity-bias \
    --device cuda:0 \
    --gpu-cache-gb 24 \
    --seed 42 \
    --no-progress \
    > "$log_file" 2>&1 &

  echo "season ${season} -> GPU ${gpu}; PID $!; log: ${log_file}"
}

launch 39 4
launch 40 4
launch 41 5
launch 42 5
launch 43 6
launch 44 6

wait

#!/usr/bin/env bash

set -euo pipefail

for season in 39 40 41 42 43; do
  mkdir -p logs

  nohup conda run --no-capture-output -n fluProfiler python \
    experiments/serum_gate/train_serum_mutation_set.py \
    --data-dir "data/dataset/H1H3_HA1_v1.0/splited/20260717_164256/season/${season}" \
    --embedding-dir data/embedding/files \
    --ha-distance-matrix ha1_distance_matrix.npy \
    --output-dir "results/H1H3_HA1_v1.0/20260717_164256/SerumMutationSet-Minus/season/${season}/subtype/H3N2" \
    --type H3N2 \
    --refit-train-valid \
    --batch-size 1 \
    --max-queries-per-task 32 \
    --epochs 100 \
    --learning-rate 1e-4 \
    --weight-decay 0.01 \
    --lr-scheduler cosine \
    --lr-min 1e-6 \
    --site-dim 64 \
    --background-dim 128 \
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
    --loss nll \
    --score-log-var-mode sum \
    --device cuda:6 \
    --gpu-cache-gb 24 \
    --seed 42 \
    --progress \
    > "logs/SerumMutationSet-Minus_H3N2_${season}.log" 2>&1 &

  echo "season ${season}: PID $!"
done

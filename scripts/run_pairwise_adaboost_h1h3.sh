#!/usr/bin/env bash
set -euo pipefail

# Edit this parameter block for each fixed-split AdaBoost benchmark.
data_dir="data/H1H3/splited/v1/H1H3__seed42__tr0.80_va0.10_te0.10__20260601_120452/titer"
output_dir="results/H1H3/adaboost_pairwise_titer"

sample_limit=-1
sequence_start=16
sequence_end=345
random_state=100
fast=false

# No need to edit below this line.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

extra_args=()
if [[ "$fast" == true ]]; then
  extra_args+=(--fast)
fi

python experiments/benchmark_pairwise/train_adaboost_fixed_split.py \
  --data-dir "$data_dir" \
  --output-dir "$output_dir" \
  --sample-limit "$sample_limit" \
  --sequence-start "$sequence_start" \
  --sequence-end "$sequence_end" \
  --random-state "$random_state" \
  "${extra_args[@]}"

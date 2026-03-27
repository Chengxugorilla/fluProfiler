#!/usr/bin/env bash
set -euo pipefail

# Raw dataset version directory, expected to contain source.csv
raw_version_dir="data/raw/r2026_03_27_ha_only"
dataset_name="ha_only"   # 用户自定义数据集名称（逻辑分组）

# Optional metadata description written into dataset_meta.json
dataset_description="HA-only training table as a protocol raw dataset"

# Split protocol settings
protocol_version="v1"
seed=42
test_ratio=0.2
valid_ratio=0.1
strain_col="seq_id_c"
serum_col="seq_id_a"
split_modes="titer,strain,serum"   # 可选：titer / strain / serum，逗号分隔

python "experiments/tools/build_splits.py" \
  --raw-version-dir "${raw_version_dir}" \
  --dataset-name "${dataset_name}" \
  --dataset-description "${dataset_description}" \
  --protocol-version "${protocol_version}" \
  --seed "${seed}" \
  --test-ratio "${test_ratio}" \
  --valid-ratio "${valid_ratio}" \
  --strain-col "${strain_col}" \
  --serum-col "${serum_col}" \
  --split-modes "${split_modes}" \
  "$@"

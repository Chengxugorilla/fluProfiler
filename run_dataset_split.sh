#!/usr/bin/env bash
set -euo pipefail

# Raw dataset version directory, expected to contain source.csv
raw_version_dir="data/H1H3/raw"
dataset_name="H1H3"   # 用户自定义数据集名称（逻辑分组）

# Optional metadata description written into dataset_meta.json
dataset_description="test for H1 and H3 subtype"

# Split protocol settings
splits_root="data/H1H3/splited"
protocol_version="v1"
seed=42
test_ratio=0.1
valid_ratio=0.1
group_valid=false   # true: valid按组隔离；false: valid从非test数据中随机抽取
strain_col="seq_id_c"
serum_col="seq_id_a"
pre_split_agg_cols="seq_id_a,seq_id_b,seq_id_c,seq_id_d"
train_agg_cols="seq_id_a,seq_id_c"
split_modes="titer,strain,serum"   # 可选：titer / strain / serum，逗号分隔

python "experiments/tools/build_splits.py" \
  --raw-version-dir "${raw_version_dir}" \
  --dataset-name "${dataset_name}" \
  --dataset-description "${dataset_description}" \
  --splits-root "${splits_root}" \
  --protocol-version "${protocol_version}" \
  --dataset-version-id "H1H3" \
  --dataset-scoped-output \
  --seed "${seed}" \
  --test-ratio "${test_ratio}" \
  --valid-ratio "${valid_ratio}" \
  --group-valid "${group_valid}" \
  --strain-col "${strain_col}" \
  --serum-col "${serum_col}" \
  --pre-split-agg-cols "${pre_split_agg_cols}" \
  --train-agg-cols "${train_agg_cols}" \
  --split-modes "${split_modes}" \
  "$@"

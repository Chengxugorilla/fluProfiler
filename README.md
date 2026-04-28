<p align="center">
  <img src="assets/fluprofiler-icon.png" width="180" alt="fluProfiler icon">
</p>

<h1 align="center">fluProfiler</h1>

<p align="center">
  Influenza antigenicity modeling toolkit for training and evaluation with a simple user-facing workflow.
</p>

## 1) Environment Setup

Recommended:

- Python `3.10`
- Linux + CUDA GPU (for training speed)

Create environment and install dependencies:

```bash
conda create -n fluProfiler python=3.10
conda activate fluProfiler
pip install -r requirements.txt
```

Verify PyTorch GPU availability (optional):

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

## 2) Required Data Layout

The training flow expects:

- Split CSV files: `train.csv`, `test.csv`
- Embedding tensors: `matrix_<seq_id>.pt`
- Reverse-test style directories under `data/reverse_test/`

Typical paths used by configs:

- CSV root: `data/reverse_test/processed/<season>/`
- Embedding root: `data/reverse_test/embedding/`

Check and edit the config file before running:

- `experiments/HA_only/config_v2_ha_only.json`
- `experiments/HANA/config_v2_hana.json`

## 3) Recommended Run Method (Single Entry)

Use the top-level script:

```bash
bash run_fluprofiler.sh
```

Edit parameters at the top of `run_fluprofiler.sh`:

- `task`: `ha_only` or `hana`
- `impl`: `v2` or `legacy`
- `config`: config file path
- `batch_size`
- `learning_rate`
- `epochs`
- `device` (example: `cuda:0`)
- `gpu_cache_gb`
- `sample_limit` (`-1` = full data, small value = quick test)

Example quick smoke test:

```bash
bash run_fluprofiler.sh --sample-limit 128 --epochs 1 --batch-size 8
```

## 4) Direct Commands (Optional)

If you want to bypass the shell entry:

```bash
# HA-only v2
python experiments/HA_only/train_v2_ha_only.py experiments/HA_only/config_v2_ha_only.json

# HANA v2
python experiments/HANA/train_v2_hana.py experiments/HANA/config_v2_hana.json
```

Dispatcher usage from repo root:

```bash
PYTHONPATH=src python src/fluprofiler/cli/dispatch.py --task ha_only --impl v2
PYTHONPATH=src python src/fluprofiler/cli/dispatch.py --task hana --impl v2
```

## 5) Outputs

Run artifacts are saved under `runs/`, including:

- TensorBoard logs
- Checkpoints
- Run logs / metadata

## 6) Dataset Split Tool (Independent)

This repository includes a standalone split builder for the three paper split modes:

- `titer`  (row-level random split)
- `strain` (group split by strain key, default `seq_id_c`)
- `serum`  (group split by serum key, default `seq_id_a`)

Script:

- `experiments/tools/build_splits.py`

### Recommended Raw Data Protocol

Use one directory per raw dataset version:

```text
data/raw/
└── r2026_03_27_mix_h1h3/
    ├── source.csv
    └── dataset_meta.json   # auto-created/updated by script
```

Run split generation from that raw version directory:

```bash
python experiments/tools/build_splits.py \
  --raw-version-dir data/raw/r2026_03_27_mix_h1h3 \
  --dataset-name hi_mix_h1h3 \
  --dataset-description "Merged H1N1 and H3N2 dataset" \
  --protocol-version v1 \
  --seed 42 \
  --test-ratio 0.2 \
  --valid-ratio 0.1 \
  --strain-col seq_id_c \
  --serum-col seq_id_a \
  --split-modes titer,strain,serum
```

Generated files:

```text
data/splits/v1/hi_mix_h1h3/r2026_03_27_mix_h1h3/
├── titer/<split_id>/{train.csv,valid.csv,test.csv,manifest.json}
├── strain/<split_id>/{train.csv,valid.csv,test.csv,manifest.json}
└── serum/<split_id>/{train.csv,valid.csv,test.csv,manifest.json}
```

`manifest.json` records split parameters, source checksum, dataset metadata, and overlap/leakage checks.

Notes:

- This utility is standalone and is **not** wired into training entrypoints.
- You can still use `--input-csv` + `--dataset-version-id` directly if needed.
- Use `--id-col` if your input has a stable unique row identifier.
- `--split-modes` controls which splits are generated (for example: `titer,serum`).

## 7) Troubleshooting

### CUDA Out of Memory

Reduce memory pressure by:

- Lowering `batch_size` (e.g. `64 -> 16 -> 8`)
- Lowering `gpu_cache_gb`
- Switching to a less busy GPU (`device`)
- Using a small `sample_limit` for validation first

Optional allocator setting:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### `CalledProcessError`

This is usually a wrapper error from the dispatcher.  
Check the first traceback above it to find the real cause.

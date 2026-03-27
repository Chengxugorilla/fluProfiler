# fluProfiler

Influenza antigenicity modeling toolkit for training and evaluation with a simple user-facing workflow.

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

## 6) Troubleshooting

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

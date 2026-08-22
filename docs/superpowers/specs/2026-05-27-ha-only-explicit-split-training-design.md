# HA-Only Explicit Split Training Design

## Goal

Add a reusable HA-only v2 training/testing entrypoint for any pre-built fixed
split with the expected CSV columns. The entrypoint must use the provided
train, validation, and test sets exactly as stored, while exposing all data
asset and result locations at invocation time.

## Interface

Create `experiments/HA_only/train_test_explicit.py` with required path
arguments:

```bash
python experiments/HA_only/train_test_explicit.py \
  --data-dir /path/to/a/split/directory \
  --embedding-dir /path/to/embedding/files \
  --output-dir /path/to/results \
  --device cuda:5
```

`--data-dir` represents one completed split from any dataset, for example an
H5 `titer`, `strain`, or `serum` directory. It must contain:

- `train.csv`
- `valid.csv`
- `test.csv`

If any required CSV is missing, the script exits before building a model and
reports the missing filename or filenames.

The script also accepts training overrides needed for smoke and production
runs: `--epochs`, `--batch-size`, `--learning-rate`, `--patience`, `--seed`,
`--device`, `--gpu-cache-gb`, `--sample-limit`, and learning-rate schedule
enable/disable flags. Defaults follow the existing HA-only v2 entrypoint.

Each CSV must provide the columns required by the HA-only input pipeline:
`seq_id_a`, `seq_id_c`, `serumPassCat`, `virusPassCat`, and `label`. This
schema validation lets the entrypoint be reused for H5 or later datasets
without encoding dataset names in the training code.

## Data Flow

The script reads all three CSV files directly from `--data-dir`. It does not
call `train_test_split` and does not alter the fixed partition identity beyond
an optional `--sample-limit` requested for a short smoke run.

The model remains HA-only: required embedding keys are generated from
`seq_id_a` and `seq_id_c` across all three loaded partitions. Before loading
embedding tensors, the script checks for every required
`matrix_<seq_id>.pt` within `--embedding-dir`. Missing embedding files produce
a clear failure with a count and representative missing paths.

After validation, data loaders, the GPU cache, model construction, optimizer,
scheduler, training step, and metric evaluation reuse the same concepts as
`experiments/HA_only/train_v2_ha_only.py`, using `fluProfiler_HA_only_v2`.

## Results

`--output-dir` is the exact destination for one run. To protect an existing
run, the script rejects an output directory that already contains generated
result files.

For a successful run it creates:

```text
output-dir/
  run_config.json
  metrics.jsonl
  log.txt
  tensorboard/
  checkpoints/
```

`run_config.json` records the resolved input paths and training settings.
`metrics.jsonl` contains one JSON record per completed epoch with training
loss plus validation and test metrics. `log.txt` retains readable per-epoch
logging. `checkpoints/` stores the best model according to validation MSE;
`tensorboard/` stores visualization event data.

## Error Handling

Validation happens before a run directory is populated:

- missing or non-directory `--data-dir` fails immediately;
- missing `train.csv`, `valid.csv`, or `test.csv` fails immediately;
- missing or non-directory `--embedding-dir` fails immediately;
- absent required HA embedding files fail with a concise missing-file report;
- an output location containing prior result artifacts fails rather than
  silently mixing runs.

## Testing

Add focused automated tests around behavior that differs from the older
entrypoint:

- a split directory missing `valid.csv` is rejected;
- loading a fixed split uses rows from `valid.csv` and does not create a new
  random validation partition;
- embedding validation identifies missing HA embedding files;
- a valid minimal fixed split resolves the expected embedding key set and
  output locations without launching a long training run.

Run a CPU smoke invocation with tiny synthetic data and one epoch after unit
tests pass, so the command-line path, result writing, and model loop are
exercised together.

# Data Layout

The repository now uses a dataset-scoped data layout plus one global embedding store.

```text
data/
├── embedding/
│   ├── registry/
│   │   ├── sequences.csv
│   │   └── pending/
│   └── files/
│       └── matrix_<seq_id>.pt
├── dataset/
│   ├── H1H3/
│   │   ├── raw/
│   │   ├── processed/
│   │   └── splited/
│   ├── H1H3_new/
│   │   ├── raw/
│   │   ├── processed/
│   │   └── splited/
│   └── H5/
│       ├── raw/
│       ├── processed/
│       └── splited/
└── deprecated/
```

`data/embedding/registry/sequences.csv` is the global sequence registry. Each row
maps one unique protein sequence to a stable `seq_id`, its segment (`HA` or `NA`),
its sequence hash, and its embedding path.

Embedding files use the stable ID:

```text
HA_4219 -> data/embedding/files/matrix_HA_4219.pt
NA_2284 -> data/embedding/files/matrix_NA_2284.pt
```

Dataset CSVs should store `seq_id_a`, `seq_id_b`, `seq_id_c`, and `seq_id_d`.
Data preparation should first look up each sequence in the global registry. If a
sequence is missing, append a new ID to the registry and mark its embedding as
missing until the embedding file is generated.

The recommended dataset preparation commands are:

```bash
python scripts/prepare_dataset_processed.py --dataset-dir data/dataset/H1H3_new
python scripts/prepare_dataset_splits.py --dataset-dir data/dataset/H1H3_new
```

Legacy layouts such as `data/raw`, `data/splited`, and `data/reverse_test` have
been moved under `data/deprecated/`.

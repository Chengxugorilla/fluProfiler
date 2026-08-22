#!/usr/bin/env python3
"""Build processed/source.csv from raw/data4model(...).csv files.

The script only performs one operation: merge all data4model tables, then
deduplicate by user-specified columns. For each duplicate group, the `label` column 
is averaged and allother columns keep the first observed value.

Example:

python src/fluprofiler/dataset/raw2processed.py \
  --dataset-dir /home/chenyh/workspace/fluProfiler/data/dataset/H1H3_HA1 \
  --registry-csv /home/chenyh/workspace/fluProfiler/data/embedding/registry/sequences.csv \
  --embedding-files-dir /home/chenyh/workspace/fluProfiler/data/embedding/files \
  --group-cols seq_a,seq_c,serumPassCat,virusPassCat

"""

import argparse
import json
from pathlib import Path
import hashlib
import re
from datetime import datetime

import pandas as pd

REGISTRY_COLUMNS = [
    "seq_id",
    "segment",
    "sequence_hash",
    "sequence",
    "embedding_path",
    "embedding_exists",
]
DATA4MODEL_CSV_PATTERN = re.compile(r"^data4model\([^)]+\)\.csv$")


def sequence_hash(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def read_registry(path: Path) -> pd.DataFrame:
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        return pd.DataFrame(columns=REGISTRY_COLUMNS)
    registry = pd.read_csv(path, dtype=str, keep_default_na=False).fillna("")
    missing = [col for col in REGISTRY_COLUMNS if col not in registry.columns]
    if missing:
        raise ValueError(f"{path} is missing registry column(s): {', '.join(missing)}")
    return registry.loc[:, REGISTRY_COLUMNS].copy()


def next_id(segment: str, used_ids: set[str]) -> str:
    pattern = re.compile(rf"^{re.escape(segment)}_(\d+)$")
    max_value = 0
    for seq_id in used_ids:
        match = pattern.match(str(seq_id))
        if match:
            max_value = max(max_value, int(match.group(1)))
    return f"{segment}_{max_value + 1}"


def registry_lookup(registry: pd.DataFrame) -> dict[tuple[str, str], str]:
    lookup = {}
    for _, row in registry.iterrows():
        key = (str(row["segment"]), str(row["sequence"]).strip())
        if key in lookup and lookup[key] != str(row["seq_id"]):
            raise ValueError(f"Duplicate sequence registry entry for {key[0]}")
        lookup[key] = str(row["seq_id"])
    return lookup

def parse_cols(text: str) -> list[str]:
    """Parse the --group-cols argument, e.g. "seq_a,seq_c", into column names."""
    cols = [col.strip() for col in text.split(",") if col.strip()]
    if not cols:
        raise ValueError("--group-cols cannot be empty")
    return list(dict.fromkeys(cols))


def is_data4model_csv(path: Path) -> bool:
    return path.is_file() and DATA4MODEL_CSV_PATTERN.fullmatch(path.name) is not None


def register_sequences(
    df: pd.DataFrame,
    registry_csv: Path,
    embedding_files_dir: Path,
) -> dict[str, int]:
    registry = read_registry(registry_csv)
    registry_rows = registry.to_dict("records")
    lookup = registry_lookup(registry)
    used_ids = {str(seq_id) for seq_id in registry["seq_id"].tolist()}

    assignments = {
        "seq_id_a": ("seq_a", "HA"),
        "seq_id_b": ("seq_b", "NA"),
        "seq_id_c": ("seq_c", "HA"),
        "seq_id_d": ("seq_d", "NA"),
    }

    new_counts = {"HA": 0, "NA": 0}

    for id_col, (seq_col, segment) in assignments.items():
        if seq_col not in df.columns:
            raise ValueError(f"Missing required sequence column: {seq_col}")

        ids = []
        for value in df[seq_col].tolist():
            sequence = str(value).strip()
            if not sequence:
                ids.append("")
                continue

            key = (segment, sequence)
            if key in lookup:
                ids.append(lookup[key])
                continue

            seq_id = next_id(segment, used_ids)
            used_ids.add(seq_id)
            lookup[key] = seq_id

            embedding_path = embedding_files_dir / f"matrix_{seq_id}.pt"
            registry_rows.append(
                {
                    "seq_id": seq_id,
                    "segment": segment,
                    "sequence_hash": sequence_hash(sequence),
                    "sequence": sequence,
                    "embedding_path": str(embedding_path),
                    "embedding_exists": str(embedding_path.is_file()).lower(),
                }
            )
            new_counts[segment] += 1
            ids.append(seq_id)

        df[id_col] = ids

    updated_registry = pd.DataFrame(registry_rows, columns=REGISTRY_COLUMNS)
    updated_registry.to_csv(registry_csv, index=False)
    return new_counts


def write_pending_fasta(
    df: pd.DataFrame,
    embedding_files_dir: Path,
    pending_dir: Path,
    timestamp: str,
) -> tuple[Path | None, int]:
    used_sequences: dict[str, str] = {}
    for id_col, seq_col in [
        ("seq_id_a", "seq_a"),
        ("seq_id_b", "seq_b"),
        ("seq_id_c", "seq_c"),
        ("seq_id_d", "seq_d"),
    ]:
        for seq_id, sequence in zip(df[id_col].tolist(), df[seq_col].tolist()):
            seq_id = str(seq_id).strip()
            sequence = str(sequence).strip()
            if seq_id and sequence:
                used_sequences[seq_id] = sequence

    missing = [
        (seq_id, sequence)
        for seq_id, sequence in sorted(used_sequences.items())
        if not (embedding_files_dir / f"matrix_{seq_id}.pt").is_file()
    ]
    if not missing:
        return None, 0

    pending_dir.mkdir(parents=True, exist_ok=True)
    pending_fasta = pending_dir / f"{timestamp}.fasta"
    with pending_fasta.open("w", encoding="utf-8") as handle:
        for seq_id, sequence in missing:
            handle.write(f">{seq_id}\n{sequence}\n")
    return pending_fasta, len(missing)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]

    parser = argparse.ArgumentParser(
        description="Merge raw data4model CSVs into processed/source.csv."
    )
    parser.add_argument(
        "--registry-csv",
        type=Path,
        default=repo_root / "data" / "embedding" / "registry" / "sequences.csv",
    )
    parser.add_argument(
        "--embedding-files-dir",
        type=Path,
        default=repo_root / "data" / "embedding" / "files",
    )
    parser.add_argument(
        "--pending-dir",
        type=Path,
        default=repo_root / "data" / "embedding" / "registry" / "pending",
    )
    parser.add_argument(
        "--timestamp",
        default=None,
        help="Timestamp used for pending FASTA filename; defaults to current time.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Dataset directory containing raw/ and receiving processed/source.csv.",
    )
    parser.add_argument("--group-cols", default="seq_a,seq_c,serumPassCat,virusPassCat")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.expanduser().resolve()
    raw_dir = dataset_dir / "raw"
    processed_dir = dataset_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(path for path in raw_dir.iterdir() if is_data4model_csv(path))
    if not csv_files:
        raise FileNotFoundError(f"No data4model(...).csv files found in {raw_dir}")

    # Keep all columns from all input files; pandas fills missing columns with NaN.
    df = pd.concat(
        [pd.read_csv(path, keep_default_na=False) for path in csv_files],
        ignore_index=True,
        sort=False,
    )
    new_sequences = register_sequences(
        df,
        args.registry_csv.expanduser().resolve(),
        args.embedding_files_dir.expanduser().resolve(),
    )
    embedding_files_dir = args.embedding_files_dir.expanduser().resolve()
    pending_fasta, missing_embedding_count = write_pending_fasta(
        df,
        embedding_files_dir,
        args.pending_dir.expanduser().resolve(),
        args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S"),
    )

    group_cols = parse_cols(args.group_cols)

    missing = [col for col in [*group_cols, "label", "serumType", "virusType"] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    subtype_mismatch = df["serumType"].astype(str) != df["virusType"].astype(str)
    if subtype_mismatch.any():
        raise ValueError("serumType and virusType differ for at least one row")
    df["Type"] = df["serumType"]

    rows_input = len(df)
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"]).copy()

    # Deduplicate only by group_cols: label is the mean, every other column is first.
    agg = {col: "first" for col in df.columns if col not in group_cols}
    agg["label"] = "mean"

    out = df.groupby(group_cols, as_index=False, dropna=False, sort=False).agg(agg)
    out = out[df.columns.intersection(out.columns)]

    output_csv = processed_dir / "source.csv"
    out.to_csv(output_csv, index=False)

    source_config = {
        "script": "src/fluprofiler/dataset/raw2processed.py",
        "dataset_dir": str(dataset_dir),
        "raw_dir": str(raw_dir),
        "input_files": [path.name for path in csv_files],
        "group_cols": group_cols,
        "label_aggregation": "mean",
        "other_columns": "first",
        "rows_input": rows_input,
        "rows_output": len(out),
        "registry_csv": str(args.registry_csv.expanduser().resolve()),
        "new_sequences": new_sequences,
        "missing_embedding_count": missing_embedding_count,
        "pending_fasta": str(pending_fasta) if pending_fasta is not None else None,
    }
    (processed_dir / "source_config.json").write_text(
        json.dumps(source_config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"input files: {len(csv_files)}")
    for path in csv_files:
        print(f"  {path}")
    print(f"rows input: {rows_input}")
    print(f"rows output: {len(out)}")
    print(f"group cols: {','.join(group_cols)}")
    print(f"wrote: {output_csv}")
    if pending_fasta is not None:
        print(f"pending fasta: {pending_fasta}")


if __name__ == "__main__":
    main()

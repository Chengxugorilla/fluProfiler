#!/usr/bin/env python3
"""Prepare a no-test full-data split for post-hoc model interpretation.

The output preserves the fixed-split interface expected by
``train_serum_mutation_set.py`` while deliberately placing every eligible row
in train.csv.  valid.csv and test.csv contain headers only; the associated
training command must use --refit-train-valid --skip-test-eval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an all-data, no-test split for interpretation only."
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--type", default="H3N2", dest="type_filter")
    parser.add_argument(
        "--task-cols",
        default="seq_id_a,serumPassCat,serumName",
        help="Columns defining antiserum conditions for manifest reporting.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.input_csv).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {source}")
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")

    frame = pd.read_csv(source)
    if "Type" not in frame.columns:
        raise KeyError("Input CSV must contain a Type column.")
    selected = frame.loc[frame["Type"].astype(str) == args.type_filter].copy()
    if selected.empty:
        raise ValueError(f"No rows found for Type={args.type_filter!r}.")

    task_cols = [item.strip() for item in args.task_cols.split(",") if item.strip()]
    missing = [column for column in task_cols if column not in selected.columns]
    if missing:
        raise KeyError(f"Missing task columns: {', '.join(missing)}")
    task_count = int(selected.loc[:, task_cols].fillna("unknown").drop_duplicates().shape[0])

    output.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output / "train.csv", index=False)
    selected.iloc[0:0].to_csv(output / "valid.csv", index=False)
    selected.iloc[0:0].to_csv(output / "test.csv", index=False)

    manifest = {
        "purpose": "full_data_interpretation_only",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_csv": str(source),
        "input_sha256": sha256(source),
        "type_filter": args.type_filter,
        "task_columns": task_cols,
        "counts": {
            "all_eligible_rows": int(len(selected)),
            "train": int(len(selected)),
            "valid": 0,
            "test": 0,
            "antiserum_conditions": task_count,
        },
        "training_contract": {
            "required_flags": ["--refit-train-valid", "--skip-test-eval"],
            "benchmark_use_permitted": False,
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()

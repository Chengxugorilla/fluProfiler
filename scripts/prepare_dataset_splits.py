#!/usr/bin/env python3
"""Prepare a dataset processed/source.csv into splited train/valid/test CSVs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare dataset processed/source.csv into splited train/valid/test CSVs."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--source-csv-name", default="source.csv")
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--dataset-version-id", default="")
    parser.add_argument("--dataset-description", default="")
    parser.add_argument("--protocol-version", default="v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--group-valid", choices=("true", "false"), default="false")
    parser.add_argument("--strain-col", default="seq_id_c")
    parser.add_argument("--serum-col", default="seq_id_a")
    parser.add_argument("--pre-split-agg-cols", default="seq_id_a,seq_id_b,seq_id_c,seq_id_d")
    parser.add_argument("--train-agg-cols", default="seq_id_a,seq_id_c")
    parser.add_argument("--split-modes", default="titer,strain,serum")
    parser.add_argument("--season-col", default="sheet")
    parser.add_argument("--test-seasons", default="")
    parser.add_argument("--id-col", default="")
    parser.add_argument("--split-id", default="")
    return parser.parse_args()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    from experiments.tools import build_splits

    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    dataset_name = args.dataset_name.strip() or dataset_dir.name
    dataset_version_id = args.dataset_version_id.strip() or dataset_dir.name

    build_argv = [
        "build_splits.py",
        "--raw-version-dir",
        str(dataset_dir / "processed"),
        "--raw-csv-name",
        args.source_csv_name,
        "--dataset-name",
        dataset_name,
        "--dataset-version-id",
        dataset_version_id,
        "--dataset-description",
        args.dataset_description,
        "--splits-root",
        str(dataset_dir / "splited"),
        "--protocol-version",
        args.protocol_version,
        "--dataset-scoped-output",
        "--seed",
        str(args.seed),
        "--test-ratio",
        str(args.test_ratio),
        "--valid-ratio",
        str(args.valid_ratio),
        "--group-valid",
        args.group_valid,
        "--strain-col",
        args.strain_col,
        "--serum-col",
        args.serum_col,
        "--pre-split-agg-cols",
        args.pre_split_agg_cols,
        "--train-agg-cols",
        args.train_agg_cols,
        "--split-modes",
        args.split_modes,
        "--season-col",
        args.season_col,
    ]
    if args.test_seasons:
        build_argv.extend(["--test-seasons", args.test_seasons])
    if args.id_col:
        build_argv.extend(["--id-col", args.id_col])
    if args.split_id:
        build_argv.extend(["--split-id", args.split_id])

    old_argv = sys.argv
    try:
        sys.argv = build_argv
        build_splits.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()

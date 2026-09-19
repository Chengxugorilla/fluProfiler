#!/usr/bin/env python3
"""
Build artificial train rows for the Metric HA antigenic trainer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[1]
sys.path.append(str(_REPO_ROOT / "experiments" / "metric_antigenic"))

from train_metric_ha import build_reverse_pair_frame, validate_training_frame_columns  # noqa: E402


def build_artificial_frame(train_frame: pd.DataFrame, method: str = "reverse_pair") -> pd.DataFrame:
    if method != "reverse_pair":
        raise ValueError(f"Unsupported artificial data method: {method}")

    artificial_frame = build_reverse_pair_frame(train_frame)
    artificial_frame["is_artificial"] = True
    artificial_frame["artificial_method"] = method
    return artificial_frame


def resolve_paths(data_dir: Path | None, train_csv: Path | None, output_csv: Path | None) -> tuple[Path, Path]:
    if (data_dir is None) == (train_csv is None):
        raise ValueError("Provide exactly one of --data-dir or --train-csv")

    resolved_train_csv = (
        Path(train_csv).expanduser().resolve()
        if train_csv is not None
        else Path(data_dir).expanduser().resolve() / "train.csv"
    )
    resolved_output_csv = (
        Path(output_csv).expanduser().resolve()
        if output_csv is not None
        else resolved_train_csv.parent / "artificial_data.csv"
    )
    return resolved_train_csv, resolved_output_csv


def write_artificial_csv(
    train_csv: Path,
    output_csv: Path,
    method: str = "reverse_pair",
    overwrite: bool = False,
) -> pd.DataFrame:
    if not train_csv.is_file():
        raise FileNotFoundError(f"Train CSV does not exist: {train_csv}")
    if output_csv.exists() and not overwrite:
        raise FileExistsError(f"Output CSV already exists; pass --overwrite to replace it: {output_csv}")

    train_frame = pd.read_csv(train_csv)
    validate_training_frame_columns(train_frame, train_csv)
    artificial_frame = build_artificial_frame(train_frame, method=method)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    artificial_frame.to_csv(output_csv, index=False)
    return artificial_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Metric HA artificial train data.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data-dir", type=Path, help="Split directory containing train.csv.")
    source.add_argument("--train-csv", type=Path, help="Explicit train.csv path.")
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--method", choices=("reverse_pair",), default="reverse_pair")
    parser.add_argument("--overwrite", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_csv, output_csv = resolve_paths(args.data_dir, args.train_csv, args.output_csv)
    artificial_frame = write_artificial_csv(
        train_csv,
        output_csv,
        method=args.method,
        overwrite=args.overwrite,
    )
    print(f"Wrote {len(artificial_frame)} artificial row(s) to {output_csv}")


if __name__ == "__main__":
    main()

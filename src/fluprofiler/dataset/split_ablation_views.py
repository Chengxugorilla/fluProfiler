#!/usr/bin/env python3
"""为 HA / HA1 view 生成共享切分。"""

import argparse
from pathlib import Path

import pandas as pd

from processed2splited import split_by_group, split_by_season


def write_split(view: pd.DataFrame, manifest: pd.DataFrame, output_dir: Path, name: str) -> None:
    indexed = view.set_index("row_id")
    output_dir = output_dir / name
    output_dir.mkdir(parents=True, exist_ok=True)
    for partition in ("train", "valid", "test"):
        row_ids = manifest.loc[manifest["partition"] == partition, "row_id"]
        indexed.loc[row_ids].reset_index().to_csv(output_dir / f"{partition}.csv", index=False)


def save_split(ha: pd.DataFrame, ha1: pd.DataFrame, frames, output_dir: Path) -> None:
    train, valid, test = frames
    manifest = pd.concat(
        [
            train[["row_id"]].assign(partition="train"),
            valid[["row_id"]].assign(partition="valid"),
            test[["row_id"]].assign(partition="test"),
        ],
        ignore_index=True,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_dir / "manifest.csv", index=False)
    write_split(ha, manifest, output_dir, "HA")
    write_split(ha1, manifest, output_dir, "HA1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--subtype", choices=("H1N1", "H3N2"), required=True)
    parser.add_argument("--split-modes", default="serum,season")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="88,99")
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--serum-cols", default="task_serum_id,serumPassCat,serumName")
    parser.add_argument("--season-col", default="sheet")
    parser.add_argument("--test-seasons", default="33,34")
    args = parser.parse_args()

    view_dir = args.dataset_dir / "main" / "views" / args.subtype
    ha = pd.read_csv(view_dir / "HA_source.csv", keep_default_na=False)
    ha1 = pd.read_csv(view_dir / "HA1_source.csv", keep_default_na=False)

    modes = [mode.strip() for mode in args.split_modes.split(",")]

    if "serum" in modes:
        serum_cols = [column.strip() for column in args.serum_cols.split(",")]
        seeds = [int(seed) for seed in args.seeds.split(",")]
        serum_splits = split_by_group(
            ha,
            serum_cols,
            seeds,
            args.test_ratio,
            args.valid_ratio,
            "group",
        )
        for seed, frames in serum_splits.items():
            save_split(ha, ha1, frames, args.output_dir / "serum" / f"seed_{seed}")

    if "season" in modes:
        for test_season in args.test_seasons.split(","):
            frames = split_by_season(ha, test_season, args.season_col, args.valid_ratio)[0]
            save_split(ha, ha1, frames, args.output_dir / "season" / test_season)


if __name__ == "__main__":
    main()

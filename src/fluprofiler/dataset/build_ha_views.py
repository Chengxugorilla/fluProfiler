#!/usr/bin/env python3
"""从 processed/source.csv 构建 HA 与 HA1 训练视图。"""

import argparse
from pathlib import Path

import pandas as pd


PREFIX = {"H1N1": "H1", "H3N2": "H3"}


def deduplicate(source: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    columns = source.columns.tolist()
    aggregation = {column: "first" for column in columns if column not in group_cols}
    aggregation["label"] = "mean"
    source["label"] = pd.to_numeric(source["label"])
    source = source.groupby(group_cols, as_index=False, sort=False, dropna=False).agg(aggregation)
    return source[columns]


def make_view(source: pd.DataFrame, lookup: pd.DataFrame, sequence: str, sequence_id: str) -> pd.DataFrame:
    view = source.copy()
    view["serumHA"] = view["seq_a"].map(lookup[sequence])
    view["virusHA"] = view["seq_c"].map(lookup[sequence])
    view["seq_id_a"] = view["seq_a"].map(lookup[sequence_id])
    view["seq_id_c"] = view["seq_c"].map(lookup[sequence_id])
    return view


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--subtype", choices=PREFIX, required=True)
    parser.add_argument("--group-cols", required=True)
    args = parser.parse_args()

    prefix = PREFIX[args.subtype]
    mapping_csv = args.dataset_dir / "main" / "HA1_sequences" / f"{prefix}.csv"
    mapping = pd.read_csv(mapping_csv, keep_default_na=False)
    mapping["ha1_lucavirus_sequence"] = mapping["ha1_aligned"].str.replace("-", "", regex=False)
    ha1_ids = {
        sequence: f"{prefix}_HA1_{index:06d}"
        for index, sequence in enumerate(mapping["ha1_lucavirus_sequence"].drop_duplicates(), 1)
    }
    mapping["ha1_id"] = mapping["ha1_lucavirus_sequence"].map(ha1_ids)
    mapping.to_csv(mapping_csv, index=False)

    source = pd.read_csv(args.dataset_dir / "processed" / "source.csv", keep_default_na=False)
    source = source[source["Type"] == args.subtype].reset_index(drop=True)
    group_cols = [column.strip() for column in args.group_cols.split(",")]
    source = deduplicate(source, group_cols).reset_index(drop=True)
    source.insert(0, "row_id", [f"{args.subtype}_{index:06d}" for index in range(1, len(source) + 1)])

    lookup = mapping.set_index("full_ha_sequence")
    source["task_serum_id"] = source["seq_a"].map(lookup["full_ha_id"])
    ha_view = make_view(source, lookup, "full_ha_aligned", "full_ha_id")
    ha1_view = make_view(source, lookup, "ha1_aligned", "ha1_id")

    output_dir = args.dataset_dir / "main" / "views" / args.subtype
    output_dir.mkdir(parents=True, exist_ok=True)
    ha_view.to_csv(output_dir / "HA_source.csv", index=False)
    ha1_view.to_csv(output_dir / "HA1_source.csv", index=False)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""为 HA 与 HA1 view 生成 LucaVirus embedding FASTA。"""

import argparse
from pathlib import Path

import pandas as pd


def write_fasta(view_csv: Path, output_fasta: Path) -> None:
    view = pd.read_csv(view_csv, keep_default_na=False)
    sequences = pd.concat(
        [
            view[["seq_id_a", "serumHA"]].set_axis(["seq_id", "sequence"], axis=1),
            view[["seq_id_c", "virusHA"]].set_axis(["seq_id", "sequence"], axis=1),
        ],
        ignore_index=True,
    ).drop_duplicates("seq_id")

    with output_fasta.open("w") as handle:
        for row in sequences.itertuples(index=False):
            handle.write(f">{row.seq_id}\n{row.sequence.replace('-', '')}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--subtype", choices=("H1N1", "H3N2"), required=True)
    args = parser.parse_args()

    view_dir = args.dataset_dir / "main" / "views" / args.subtype
    embedding_dir = args.dataset_dir / "main" / "embedding" / args.subtype
    (embedding_dir / "HA").mkdir(parents=True, exist_ok=True)
    (embedding_dir / "HA1").mkdir(parents=True, exist_ok=True)

    write_fasta(view_dir / "HA_source.csv", view_dir / "HA_lucavirus.fasta")
    write_fasta(view_dir / "HA1_source.csv", view_dir / "HA1_lucavirus.fasta")


if __name__ == "__main__":
    main()

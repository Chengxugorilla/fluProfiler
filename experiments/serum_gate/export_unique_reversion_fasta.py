#!/usr/bin/env python3
"""Export unique observed-mutation reversion sequences as a FASTA file."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reverted-counterfactuals-csv", type=Path, required=True)
    parser.add_argument("--output-fasta", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequences = pd.read_csv(args.reverted_counterfactuals_csv, usecols=["reverted_query_sequence"])["reverted_query_sequence"]
    unique_sequences = sorted(set(sequences.astype(str)))
    with args.output_fasta.open("w", encoding="utf-8") as handle:
        for sequence in unique_sequences:
            sequence_id = "REV_HA_" + hashlib.sha256(sequence.replace("-", "").encode("utf-8")).hexdigest()
            handle.write(f">{sequence_id}\n{sequence.replace('-', '')}\n")
    print({"unique_reverted_sequences": len(unique_sequences), "output_fasta": str(args.output_fasta)})


if __name__ == "__main__":
    main()

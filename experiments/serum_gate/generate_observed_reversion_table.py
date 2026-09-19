#!/usr/bin/env python3
"""Create raw original-pair and single-mutation-reversion CSVs from observed data."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


CANONICAL_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-csv", type=Path, required=True)
    parser.add_argument("--original-pairs-csv", type=Path, required=True)
    parser.add_argument("--reverted-counterfactuals-csv", type=Path, required=True)
    return parser.parse_args()


def revert_at_position(sequence: str, position: int, reference_aa: str) -> str:
    index = position - 1
    return sequence[:index] + reference_aa + sequence[index + 1 :]


def main() -> None:
    args = parse_args()
    records = pd.read_csv(args.records_csv).reset_index(names="sample_index")
    pair_columns = [
        "sample_index", "seq_id_a", "seq_id_c", "serumName", "virusName", "serumPassCat",
        "virusPassCat", "serumDate", "virusDate", "label", "serumHA", "virusHA",
    ]
    original_pairs = records[pair_columns].rename(columns={
        "serumHA": "reference_sequence",
        "virusHA": "query_sequence",
    })
    args.original_pairs_csv.parent.mkdir(parents=True, exist_ok=True)
    original_pairs.to_csv(args.original_pairs_csv, index=False)

    rows: list[dict[str, object]] = []
    counterfactual_id = 0
    for pair in original_pairs.itertuples(index=False):
        reference = str(pair.reference_sequence)
        query = str(pair.query_sequence)
        positions = [
            position
            for position, (reference_aa, query_aa) in enumerate(zip(reference, query), start=1)
            if reference_aa != query_aa
            and reference_aa in CANONICAL_AMINO_ACIDS
            and query_aa in CANONICAL_AMINO_ACIDS
        ]
        for position in positions:
            reference_aa = reference[position - 1]
            query_aa = query[position - 1]
            rows.append({
                "counterfactual_id": counterfactual_id,
                "sample_index": pair.sample_index,
                "aa_position": position,
                "reference_aa": reference_aa,
                "query_aa": query_aa,
                "substitution": reference_aa + "→" + query_aa,
                "mutation_count": len(positions),
                "reverted_query_sequence": revert_at_position(query, position, reference_aa),
            })
            counterfactual_id += 1
    counterfactuals = pd.DataFrame(rows)
    counterfactuals.to_csv(args.reverted_counterfactuals_csv, index=False)
    print({
        "original_pairs": len(original_pairs),
        "counterfactual_instances": len(counterfactuals),
        "unique_reverted_query_sequences": int(counterfactuals["reverted_query_sequence"].nunique()),
        "original_pairs_csv": str(args.original_pairs_csv),
        "reverted_counterfactuals_csv": str(args.reverted_counterfactuals_csv),
    })


if __name__ == "__main__":
    main()

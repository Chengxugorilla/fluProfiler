#!/usr/bin/env python3
"""Generate the serum-conditioned, single-site HA1 DMS input tables.

Each unique (reference sequence, serum name, serum passage) background gets
one baseline ref-ref row and one row for every canonical HA1 position paired
with each of the 19 non-reference amino acids.  The mutant table is written as
an uncompressed CSV in streaming background-sized chunks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


CANONICAL_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def mutate(sequence: str, position: int, amino_acid: str) -> str:
    return sequence[:position] + amino_acid + sequence[position + 1 :]


def main() -> None:
    args = parse_args()
    records = pd.read_csv(args.records_csv)
    backgrounds = (
        records[["seq_id_a", "serumName", "serumPassCat", "serumHA"]]
        .drop_duplicates()
        .reset_index(drop=True)
        .rename(columns={"serumHA": "reference_sequence"})
    )
    backgrounds.insert(0, "background_id", range(len(backgrounds)))
    backgrounds["canonical_position_count"] = backgrounds["reference_sequence"].map(
        lambda sequence: sum(amino_acid in CANONICAL_AMINO_ACIDS for amino_acid in sequence)
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    baselines = backgrounds.assign(
        query_sequence=backgrounds["reference_sequence"],
        pair_definition="reference_sequence equals query_sequence",
    )
    baselines.to_csv(args.output_dir / "dms_baselines.csv", index=False)

    mutant_path = args.output_dir / "dms_single_mutants.csv"
    dms_id = 0
    first_chunk = True
    for background in backgrounds.itertuples(index=False):
        sequence = background.reference_sequence
        rows = []
        for position, reference_amino_acid in enumerate(sequence):
            if reference_amino_acid not in CANONICAL_AMINO_ACIDS:
                continue
            for query_amino_acid in CANONICAL_AMINO_ACIDS:
                if query_amino_acid == reference_amino_acid:
                    continue
                rows.append({
                    "dms_id": dms_id,
                    "background_id": background.background_id,
                    "aa_position": position + 1,
                    "reference_aa": reference_amino_acid,
                    "query_aa": query_amino_acid,
                    "substitution": f"{reference_amino_acid}→{query_amino_acid}",
                    "mutant_sequence": mutate(sequence, position, query_amino_acid),
                })
                dms_id += 1
        pd.DataFrame(rows).to_csv(
            mutant_path,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
        )
        first_chunk = False
        print(f"{background.background_id + 1:,} / {len(backgrounds):,}", flush=True)

    with (args.output_dir / "generation_config.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "records_csv": str(args.records_csv),
            "background_definition": "unique (seq_id_a, serumName, serumPassCat, serumHA)",
            "baseline_definition": "reference_sequence equals query_sequence",
            "mutation_definition": "all 19 canonical non-reference amino acids at each canonical reference position",
            "canonical_amino_acids": CANONICAL_AMINO_ACIDS,
            "background_count": len(backgrounds),
            "mutant_count": dms_id,
            "baseline_count": len(backgrounds),
        }, handle, indent=2)


if __name__ == "__main__":
    main()

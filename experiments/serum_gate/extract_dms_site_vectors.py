#!/usr/bin/env python3
"""Extract exact DMS mutant-site vectors from one temporary LucAvirus batch.

For this model, only the query embedding at the mutated HA1 position enters
the mutation-token path.  This script therefore retains that vector for every
DMS row and lets the full temporary LucAvirus matrices be removed afterwards.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--dms-index", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    return parser.parse_args()


def aligned_site_vector(matrix: torch.Tensor, aligned_sequence: str, position: int) -> torch.Tensor:
    matrix = torch.as_tensor(matrix).float()
    sequence = str(aligned_sequence).strip().upper()
    non_gap_count = sum(residue != "-" for residue in sequence)
    if matrix.shape[0] == non_gap_count + 2:
        matrix = matrix[1:-1]
    source_position = sum(residue != "-" for residue in sequence[: position - 1])
    return matrix[source_position]


def main() -> None:
    args = parse_args()
    connection = sqlite3.connect(args.dms_index)
    dms_ids: list[int] = []
    vectors: list[torch.Tensor] = []
    for matrix_path in sorted(args.matrix_dir.glob("matrix_DMS_HA_*.pt")):
        embedding_id = matrix_path.stem.removeprefix("matrix_")
        matrix = torch.load(matrix_path, map_location="cpu", weights_only=False)
        rows = connection.execute(
            "SELECT dms_id, aa_position, mutant_sequence FROM dms_rows WHERE embedding_id = ? ORDER BY dms_id",
            (embedding_id,),
        ).fetchall()
        for dms_id, aa_position, mutant_sequence in rows:
            dms_ids.append(dms_id)
            vectors.append(aligned_site_vector(matrix, mutant_sequence, aa_position))
    connection.close()
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "dms_id": torch.tensor(dms_ids, dtype=torch.long),
        "mutant_site_embedding": torch.stack(vectors).float(),
    }, args.output_file)
    print({"dms_rows": len(dms_ids), "shape": tuple(vectors[0].shape), "output": str(args.output_file)})


if __name__ == "__main__":
    main()

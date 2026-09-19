#!/usr/bin/env python3
"""Split pending DMS sequences into LucAvirus batches and index DMS rows.

The index connects each generated LucAvirus embedding ID back to the DMS rows
and aligned HA1 coordinates that need its mutant-site vector.  It is used by
the extraction step after each batch finishes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
from pathlib import Path

import pandas as pd


def sequence_hash(sequence: str) -> str:
    return hashlib.sha256(str(sequence).replace("-", "").strip().encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pending-fasta", type=Path, required=True)
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--mutants-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequences-per-batch", type=int, default=1000)
    parser.add_argument("--chunksize", type=int, default=100_000)
    return parser.parse_args()


def write_batches(pending_fasta: Path, batch_dir: Path, sequences_per_batch: int) -> int:
    batch_dir.mkdir(parents=True, exist_ok=True)
    batch_index = -1
    sequence_count = 0
    handle = None
    with pending_fasta.open(encoding="utf-8") as source:
        while True:
            header = source.readline()
            if not header:
                break
            sequence = source.readline()
            if sequence_count % sequences_per_batch == 0:
                if handle is not None:
                    handle.close()
                batch_index += 1
                handle = (batch_dir / f"dms_{batch_index:04d}.fasta").open("w", encoding="utf-8")
            handle.write(header)
            handle.write(sequence)
            sequence_count += 1
    if handle is not None:
        handle.close()
    return sequence_count


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    batch_dir = args.output_dir / "fasta_batches"
    batch_count = write_batches(args.pending_fasta, batch_dir, args.sequences_per_batch)

    index_path = args.output_dir / "dms_embedding_index.sqlite"
    manifest = pd.read_csv(args.manifest_csv, usecols=["sequence_hash", "embedding_id"])
    embedding_ids = dict(zip(manifest["sequence_hash"], manifest["embedding_id"]))
    connection = sqlite3.connect(index_path)
    connection.execute("DROP TABLE IF EXISTS dms_rows")
    connection.execute(
        "CREATE TABLE dms_rows (embedding_id TEXT, dms_id INTEGER, background_id INTEGER, "
        "aa_position INTEGER, mutant_sequence TEXT)"
    )
    connection.execute("CREATE INDEX dms_rows_embedding_id ON dms_rows(embedding_id)")
    total_rows = 0
    for chunk in pd.read_csv(args.mutants_csv, chunksize=args.chunksize):
        rows = [
            (
                embedding_ids[sequence_hash(row.mutant_sequence)],
                int(row.dms_id),
                int(row.background_id),
                int(row.aa_position),
                str(row.mutant_sequence),
            )
            for row in chunk.itertuples(index=False)
        ]
        connection.executemany("INSERT INTO dms_rows VALUES (?, ?, ?, ?, ?)", rows)
        total_rows += len(rows)
    connection.commit()
    connection.close()

    with (args.output_dir / "batch_config.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "pending_fasta": str(args.pending_fasta),
            "manifest_csv": str(args.manifest_csv),
            "mutants_csv": str(args.mutants_csv),
            "sequences_per_batch": args.sequences_per_batch,
            "missing_sequences": batch_count,
            "dms_rows": total_rows,
            "fasta_batches": str(batch_dir),
            "index": str(index_path),
        }, handle, indent=2)
    print(json.dumps({"missing_sequences": batch_count, "dms_rows": total_rows, "batches": (batch_count + args.sequences_per_batch - 1) // args.sequences_per_batch, "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()

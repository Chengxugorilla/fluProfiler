#!/usr/bin/env python3
"""Prepare missing LucAvirus embeddings for serum-conditioned HA1 DMS tables.

Existing embeddings are reused by matching the unaligned protein sequence to
the project embedding registry. Missing sequences receive deterministic DMS
embedding IDs and are written to a FASTA suitable for LucAvirus token-level
embedding generation. The script does not modify the global registry or the
existing embedding directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import tempfile
from pathlib import Path

import pandas as pd


def sequence_hash(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def unaligned_sequence(sequence: str) -> str:
    return str(sequence).replace("-", "").strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--mutants-csv", type=Path, required=True)
    parser.add_argument("--registry-csv", type=Path, required=True)
    parser.add_argument("--existing-embedding-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunksize", type=int, default=100_000)
    return parser.parse_args()


def existing_sequence_ids(registry_csv: Path, embedding_dir: Path) -> dict[str, str]:
    registry = pd.read_csv(registry_csv, dtype=str, keep_default_na=False)
    existing: dict[str, str] = {}
    for row in registry.itertuples(index=False):
        if str(row.segment) != "HA":
            continue
        seq_id = str(row.seq_id)
        if (embedding_dir / f"matrix_{seq_id}.pt").is_file():
            existing[unaligned_sequence(str(row.sequence))] = seq_id
    return existing


def add_sequences(connection: sqlite3.Connection, sequences: pd.Series) -> None:
    rows = ((sequence_hash(unaligned_sequence(sequence)), unaligned_sequence(sequence)) for sequence in sequences)
    connection.executemany(
        "INSERT OR IGNORE INTO sequences(sequence_hash, sequence) VALUES (?, ?)",
        rows,
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing_ids = existing_sequence_ids(args.registry_csv, args.existing_embedding_dir)

    with tempfile.TemporaryDirectory(dir=args.output_dir, prefix="sequence_index_") as temporary_dir:
        database_path = Path(temporary_dir) / "sequences.sqlite"
        connection = sqlite3.connect(database_path)
        connection.execute(
            "CREATE TABLE sequences(sequence_hash TEXT PRIMARY KEY, sequence TEXT NOT NULL)"
        )

        baselines = pd.read_csv(args.baseline_csv)
        add_sequences(connection, baselines["reference_sequence"])
        for chunk in pd.read_csv(args.mutants_csv, chunksize=args.chunksize):
            add_sequences(connection, chunk["mutant_sequence"])
        connection.commit()

        manifest_path = args.output_dir / "dms_embedding_manifest.csv"
        fasta_path = args.output_dir / "pending_dms_embeddings.fasta"
        baseline_map_path = args.output_dir / "dms_baseline_embedding_ids.csv"

        missing_count = 0
        existing_count = 0
        manifest_rows = 0
        sequence_to_id: dict[str, str] = {}
        with manifest_path.open("w", newline="", encoding="utf-8") as manifest_handle, fasta_path.open(
            "w", encoding="utf-8"
        ) as fasta_handle:
            writer = csv.DictWriter(
                manifest_handle,
                fieldnames=["sequence_hash", "embedding_id", "embedding_status", "sequence"],
            )
            writer.writeheader()
            for sequence_hash_value, sequence in connection.execute(
                "SELECT sequence_hash, sequence FROM sequences ORDER BY sequence_hash"
            ):
                embedding_id = existing_ids.get(sequence)
                if embedding_id is None:
                    embedding_id = f"DMS_HA_{sequence_hash_value}"
                    status = "missing"
                    fasta_handle.write(f">{embedding_id}\n{sequence}\n")
                    missing_count += 1
                else:
                    status = "existing"
                    existing_count += 1
                writer.writerow({
                    "sequence_hash": sequence_hash_value,
                    "embedding_id": embedding_id,
                    "embedding_status": status,
                    "sequence": sequence,
                })
                sequence_to_id[sequence] = embedding_id
                manifest_rows += 1

        baseline_map = baselines[["background_id", "seq_id_a", "serumName", "serumPassCat"]].copy()
        baseline_map["reference_embedding_id"] = baselines["reference_sequence"].map(
            lambda sequence: sequence_to_id[unaligned_sequence(sequence)]
        )
        baseline_map.to_csv(baseline_map_path, index=False)
        connection.close()

    with (args.output_dir / "prepare_config.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "baseline_csv": str(args.baseline_csv),
            "mutants_csv": str(args.mutants_csv),
            "registry_csv": str(args.registry_csv),
            "existing_embedding_dir": str(args.existing_embedding_dir),
            "sequence_key": "unaligned sequence with alignment gaps removed",
            "missing_embedding_id": "DMS_HA_<sha256 of unaligned sequence>",
            "unique_sequences": manifest_rows,
            "existing_embeddings_reused": existing_count,
            "missing_embeddings": missing_count,
            "chunksize": args.chunksize,
        }, handle, indent=2)
    print(json.dumps({
        "unique_sequences": manifest_rows,
        "existing_embeddings_reused": existing_count,
        "missing_embeddings": missing_count,
        "pending_fasta": str(fasta_path),
        "manifest": str(manifest_path),
    }, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build the global sequence-to-embedding registry."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]


def sequence_hash(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def load_sequence_map(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return {str(sequence): str(seq_id) for sequence, seq_id in data.items()}


def build_rows(
    segment: str,
    sequence_to_id: dict[str, str],
    embedding_dir: Path,
    files_dir: Path,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for sequence, seq_id in sorted(sequence_to_id.items(), key=lambda item: item[1]):
        embedding_name = f"matrix_{seq_id}.pt"
        target_path = files_dir / embedding_name
        source_path = embedding_dir / embedding_name
        rows.append(
            {
                "seq_id": seq_id,
                "segment": segment,
                "sequence_hash": sequence_hash(sequence),
                "sequence": sequence,
                "embedding_path": str(target_path),
                "embedding_exists": str(source_path.exists()).lower(),
            }
        )
    return rows


def write_registry(rows: list[dict[str, str]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "seq_id",
        "segment",
        "sequence_hash",
        "sequence",
        "embedding_path",
        "embedding_exists",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sync_embedding_links(rows: list[dict[str, str]], source_dir: Path, files_dir: Path) -> int:
    files_dir.mkdir(parents=True, exist_ok=True)
    created = 0
    for row in rows:
        if row["embedding_exists"] != "true":
            continue
        embedding_name = f"matrix_{row['seq_id']}.pt"
        source = source_dir / embedding_name
        destination = files_dir / embedding_name
        if destination.exists() or destination.is_symlink():
            continue
        destination.symlink_to(source.resolve())
        created += 1
    return created


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build global embedding registry from legacy HA/NA maps.")
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument(
        "--legacy-root",
        type=Path,
        default=Path("data/deprecated/reverse_test"),
        help="Legacy directory containing HA_map.json, NA_map.json, and embedding/.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/embedding/registry/sequences.csv"),
    )
    parser.add_argument(
        "--files-dir",
        type=Path,
        default=Path("data/embedding/files"),
    )
    parser.add_argument("--sync-links", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    legacy_root = (repo_root / args.legacy_root).resolve()
    output_csv = (repo_root / args.output_csv).resolve()
    files_dir = (repo_root / args.files_dir).resolve()
    embedding_dir = legacy_root / "embedding"

    ha_map = load_sequence_map(legacy_root / "HA_map.json")
    na_map = load_sequence_map(legacy_root / "NA_map.json")
    rows = build_rows("HA", ha_map, embedding_dir, files_dir)
    rows.extend(build_rows("NA", na_map, embedding_dir, files_dir))

    ids = [row["seq_id"] for row in rows]
    hashes = [(row["segment"], row["sequence_hash"]) for row in rows]
    sequences = [(row["segment"], row["sequence"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate seq_id detected while building registry")
    if len(hashes) != len(set(hashes)):
        raise ValueError("Duplicate segment+sequence_hash detected while building registry")
    if len(sequences) != len(set(sequences)):
        raise ValueError("Duplicate segment+sequence detected while building registry")

    write_registry(rows, output_csv)
    linked = sync_embedding_links(rows, embedding_dir, files_dir) if args.sync_links else 0
    existing = sum(row["embedding_exists"] == "true" for row in rows)
    print(f"Wrote {len(rows)} registry rows to {output_csv}")
    print(f"Existing embeddings: {existing}; symlinks created: {linked}")


if __name__ == "__main__":
    main()

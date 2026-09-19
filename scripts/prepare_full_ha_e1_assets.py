#!/usr/bin/env python3
"""Prepare isolated Full-HA assets for the HA1-vs-Full-HA ablation.

The source dataset stores Full-HA strings in ``serumHA`` and ``virusHA`` but
its ``seq_id_a``/``seq_id_c`` keys refer to HA1 embeddings.  This utility
derives stable Full-HA keys (``FHA_<sha256-prefix>``), writes a H3N2-only
source CSV using those keys, and emits the corresponding LucaVirus FASTA.
It never reads from or writes to the global embedding registry or store.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def full_ha_id(sequence: str) -> str:
    digest = hashlib.sha256(sequence.encode("ascii")).hexdigest()[:20]
    return f"FHA_{digest}"


def ungap(sequence: str) -> str:
    return sequence.replace("-", "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--subtype", default="H3N2")
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=None,
        help="Default: <dataset-dir>/full_ha_assets",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    assets_dir = (args.assets_dir or dataset_dir / "full_ha_assets").expanduser().resolve()
    source_path = dataset_dir / "processed" / "source.csv"
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing source CSV: {source_path}")

    source = pd.read_csv(source_path, keep_default_na=False)
    required = {"Type", "serumHA", "virusHA", "seq_id_a", "seq_id_c"}
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"Source CSV is missing columns: {', '.join(missing)}")

    frame = source.loc[source["Type"].astype(str) == args.subtype].copy()
    if frame.empty:
        raise ValueError(f"No rows found for subtype {args.subtype!r}")
    for column in ("serumHA", "virusHA"):
        frame[column] = frame[column].astype(str).str.strip().str.upper()
        invalid = frame[column].eq("") | ~frame[column].str.fullmatch("[A-Z-]+")
        if invalid.any():
            raise ValueError(f"{column} contains empty or non-amino-acid sequences")

    lengths = set(frame["serumHA"].str.len()) | set(frame["virusHA"].str.len())
    if len(lengths) != 1:
        raise ValueError(f"Expected one aligned Full-HA length, found: {sorted(lengths)}")

    embedding_sequences = pd.concat([frame["serumHA"], frame["virusHA"]]).map(ungap)
    if embedding_sequences.eq("").any():
        raise ValueError("A Full-HA alignment contains no amino-acid residues")
    sequence_to_id = {
        sequence: full_ha_id(sequence) for sequence in embedding_sequences.unique()
    }
    if len(set(sequence_to_id.values())) != len(sequence_to_id):
        raise RuntimeError("Full-HA ID hash collision")

    frame["ha1_seq_id_a"] = frame["seq_id_a"]
    frame["ha1_seq_id_c"] = frame["seq_id_c"]
    frame["seq_id_a"] = frame["serumHA"].map(ungap).map(sequence_to_id)
    frame["seq_id_c"] = frame["virusHA"].map(ungap).map(sequence_to_id)

    registry = pd.DataFrame(
        {
            "seq_id": list(sequence_to_id.values()),
            "sequence": list(sequence_to_id.keys()),
        }
    ).sort_values("seq_id", ignore_index=True)
    registry["sequence_length"] = registry["sequence"].str.len()
    registry["sequence_sha256"] = registry["sequence"].map(
        lambda sequence: hashlib.sha256(sequence.encode("ascii")).hexdigest()
    )

    processed_dir = assets_dir / "processed"
    embeddings_dir = assets_dir / "embeddings"
    fasta_path = assets_dir / f"full_ha_{args.subtype.lower()}.fasta"
    outputs = [
        processed_dir / "source.csv",
        processed_dir / "source_config.json",
        assets_dir / "sequence_registry.csv",
        fasta_path,
    ]
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing Full-HA assets: " + ", ".join(map(str, existing))
        )
    processed_dir.mkdir(parents=True, exist_ok=False)
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    frame.to_csv(processed_dir / "source.csv", index=False)
    source_config = {
        "purpose": "isolated_full_ha_e1_ablation",
        "parent_source_csv": str(source_path),
        "subtype": args.subtype,
        "sequence_columns": {"serum": "serumHA", "virus": "virusHA"},
        "embedding_key_columns": {"serum": "seq_id_a", "virus": "seq_id_c"},
        "embedding_directory": str(embeddings_dir),
        "group_cols": ["ha1_seq_id_a", "ha1_seq_id_c", "serumPassCat", "virusPassCat", "serumName", "virusName"],
        "rows": int(len(frame)),
        "unique_full_ha_sequences": int(len(registry)),
        "aligned_sequence_length": int(next(iter(lengths))),
    }
    (processed_dir / "source_config.json").write_text(
        json.dumps(source_config, indent=2) + "\n", encoding="utf-8"
    )
    registry.to_csv(assets_dir / "sequence_registry.csv", index=False)
    with fasta_path.open("x", encoding="ascii") as handle:
        for row in registry.itertuples(index=False):
            handle.write(f">{row.seq_id}\n{row.sequence}\n")
    print(json.dumps({"assets_dir": str(assets_dir), **source_config, "fasta": str(fasta_path)}, indent=2))


if __name__ == "__main__":
    main()

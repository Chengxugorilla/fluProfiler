#!/usr/bin/env python3
"""Prepare a dataset raw directory into processed/source.csv."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


REGISTRY_COLUMNS = [
    "seq_id",
    "segment",
    "sequence_hash",
    "sequence",
    "embedding_path",
    "embedding_exists",
]
OUTPUT_COLUMNS = [
    "seq_id_a",
    "seq_id_b",
    "seq_id_c",
    "seq_id_d",
    "seq_a",
    "seq_b",
    "seq_c",
    "seq_d",
    "serumPassCat",
    "virusPassCat",
    "serumName",
    "virusName",
    "label",
    "serumDate",
    "Type",
    "virusDate",
    "serumIslID",
    "virusIslID",
    "sheet",
    "serumHA",
    "virusHA",
]
REQUIRED_RAW_COLUMNS = [
    "label",
    "seq_a",
    "seq_b",
    "seq_c",
    "seq_d",
    "serumPassCat",
    "virusPassCat",
    "serumName",
    "virusName",
    "serumDate",
    "virusDate",
    "serumIslID",
    "virusIslID",
    "sheet",
]


def sequence_hash(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def parse_fasta(path: Path | None) -> dict[str, str]:
    if path is None or not path.is_file():
        return {}
    records: dict[str, str] = {}
    current_id: str | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            if text.startswith(">"):
                current_id = text[1:].split()[0]
                records[current_id] = ""
            elif current_id is not None:
                records[current_id] += text
    return records


def read_registry(path: Path) -> pd.DataFrame:
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        return pd.DataFrame(columns=REGISTRY_COLUMNS)
    registry = pd.read_csv(path, dtype=str, keep_default_na=False).fillna("")
    missing = [col for col in REGISTRY_COLUMNS if col not in registry.columns]
    if missing:
        raise ValueError(f"{path} is missing registry column(s): {', '.join(missing)}")
    return registry.loc[:, REGISTRY_COLUMNS].copy()


def write_registry(path: Path, registry: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    registry.loc[:, REGISTRY_COLUMNS].to_csv(path, index=False)


def next_id(segment: str, used_ids: set[str]) -> str:
    pattern = re.compile(rf"^{re.escape(segment)}_(\d+)$")
    max_value = 0
    for seq_id in used_ids:
        match = pattern.match(str(seq_id))
        if match:
            max_value = max(max_value, int(match.group(1)))
    return f"{segment}_{max_value + 1}"


def normalize_sequence(value: Any, column: str) -> str:
    if pd.isna(value):
        raise ValueError(f"Missing sequence in {column}")
    text = str(value).strip()
    if not text:
        raise ValueError(f"Missing sequence in {column}")
    return text


def sequence_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_passage(value: Any) -> str:
    if pd.isna(value):
        return "<NONE>"
    text = str(value).strip()
    if not text or text.upper() in {"NA", "NAN", "NONE", "<NA>"}:
        return "<NONE>"
    return text


def load_raw_frames(raw_dir: Path) -> pd.DataFrame:
    csv_files = sorted(path for path in raw_dir.glob("*.csv") if path.is_file())
    if not csv_files:
        raise FileNotFoundError(f"No raw CSV files found under {raw_dir}")
    frames = [pd.read_csv(path) for path in csv_files]
    frame = pd.concat(frames, axis=0, ignore_index=True)
    missing = [col for col in REQUIRED_RAW_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"Raw CSV is missing required column(s): {', '.join(missing)}")
    if "Type" not in frame.columns:
        if "serumType" not in frame.columns or "virusType" not in frame.columns:
            raise ValueError("Raw CSV must contain Type or both serumType and virusType")
        serum_type = frame["serumType"].fillna("").astype(str)
        virus_type = frame["virusType"].fillna("").astype(str)
        mismatched = serum_type != virus_type
        if mismatched.any():
            raise ValueError("serumType and virusType differ for at least one row")
        frame["Type"] = serum_type
    return frame


def registry_lookup(registry: pd.DataFrame) -> dict[tuple[str, str], str]:
    lookup: dict[tuple[str, str], str] = {}
    for _, row in registry.iterrows():
        key = (str(row["segment"]), str(row["sequence"]))
        if key in lookup and lookup[key] != str(row["seq_id"]):
            raise ValueError(f"Duplicate sequence registry entry for {key[0]}")
        lookup[key] = str(row["seq_id"])
    return lookup


def add_sequence_to_registry(
    registry_rows: list[dict[str, str]],
    lookup: dict[tuple[str, str], str],
    used_ids: set[str],
    segment: str,
    sequence: str,
    embedding_files_dir: Path,
) -> tuple[str, bool]:
    key = (segment, sequence)
    if key in lookup:
        return lookup[key], False
    seq_id = next_id(segment, used_ids)
    used_ids.add(seq_id)
    lookup[key] = seq_id
    embedding_path = embedding_files_dir / f"matrix_{seq_id}.pt"
    registry_rows.append(
        {
            "seq_id": seq_id,
            "segment": segment,
            "sequence_hash": sequence_hash(sequence),
            "sequence": sequence,
            "embedding_path": str(embedding_path),
            "embedding_exists": str(embedding_path.is_file()).lower(),
        }
    )
    return seq_id, True


def add_qc_step(
    steps: list[dict[str, Any]],
    name: str,
    action: str,
    rows_before: int,
    rows_after: int,
    details: dict[str, Any] | None = None,
) -> None:
    steps.append(
        {
            "name": name,
            "action": action,
            "rows_before": int(rows_before),
            "rows_after": int(rows_after),
            "rows_removed": int(max(rows_before - rows_after, 0)),
            "details": details or {},
        }
    )


def prepare_dataset(
    dataset_dir: Path,
    registry_csv: Path,
    embedding_files_dir: Path,
    pending_dir: Path,
    aligned_ha_fasta: Path | None,
    timestamp: str,
) -> dict[str, Any]:
    raw_dir = dataset_dir / "raw"
    processed_dir = dataset_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw_frames(raw_dir)
    steps: list[dict[str, Any]] = []
    rows_input = int(len(raw))
    add_qc_step(
        steps,
        name="load_raw_csv",
        action="loaded and concatenated raw CSV files",
        rows_before=0,
        rows_after=rows_input,
        details={"raw_dir": str(raw_dir), "csv_files": len(list(raw_dir.glob("*.csv")))},
    )

    before = len(raw)
    original_label_missing = int(raw["label"].isna().sum())
    raw["label"] = pd.to_numeric(raw["label"], errors="coerce")
    coerced_label_missing = int(raw["label"].isna().sum())
    add_qc_step(
        steps,
        name="coerce_label",
        action="converted label to numeric",
        rows_before=before,
        rows_after=len(raw),
        details={
            "missing_before": original_label_missing,
            "missing_after": coerced_label_missing,
            "new_missing_from_conversion": int(coerced_label_missing - original_label_missing),
        },
    )

    before = len(raw)
    raw = raw.dropna(subset=["label"]).copy()
    add_qc_step(
        steps,
        name="drop_missing_label",
        action="removed rows with missing or non-numeric label",
        rows_before=before,
        rows_after=len(raw),
        details={"removed_rows": int(before - len(raw))},
    )

    sequence_cols = ["seq_a", "seq_b", "seq_c", "seq_d"]
    before = len(raw)
    for col in sequence_cols:
        raw[col] = raw[col].map(sequence_text)
    empty_sequence_counts = {col: int((raw[col] == "").sum()) for col in sequence_cols}
    add_qc_step(
        steps,
        name="normalize_sequence_text",
        action="trimmed sequence columns and converted missing values to empty strings",
        rows_before=before,
        rows_after=len(raw),
        details={"empty_sequence_counts_after_normalization": empty_sequence_counts},
    )

    before = len(raw)
    raw = raw[(raw[sequence_cols] != "").all(axis=1)].copy()
    add_qc_step(
        steps,
        name="drop_missing_sequences",
        action="removed rows missing any of seq_a, seq_b, seq_c, seq_d",
        rows_before=before,
        rows_after=len(raw),
        details={"empty_sequence_counts": empty_sequence_counts},
    )

    before = len(raw)
    serum_passage_before = raw["serumPassCat"].copy()
    virus_passage_before = raw["virusPassCat"].copy()
    raw["serumPassCat"] = raw["serumPassCat"].map(normalize_passage)
    raw["virusPassCat"] = raw["virusPassCat"].map(normalize_passage)
    add_qc_step(
        steps,
        name="normalize_passage",
        action="normalized empty passage categories to <NONE>",
        rows_before=before,
        rows_after=len(raw),
        details={
            "serumPassCat_changed": int((serum_passage_before.fillna("") != raw["serumPassCat"].fillna("")).sum()),
            "virusPassCat_changed": int((virus_passage_before.fillna("") != raw["virusPassCat"].fillna("")).sum()),
        },
    )

    before = len(raw)
    for col in sequence_cols:
        raw[col] = raw[col].map(lambda value, column=col: normalize_sequence(value, column))
    raw = raw.reset_index(drop=True)
    add_qc_step(
        steps,
        name="finalize_clean_rows",
        action="reset row index after row-level filtering",
        rows_before=before,
        rows_after=len(raw),
    )
    rows_dropped = int(rows_input - len(raw))

    registry = read_registry(registry_csv)
    registry_rows = registry.to_dict("records")
    lookup = registry_lookup(registry)
    used_ids = {str(seq_id) for seq_id in registry["seq_id"].tolist()}
    new_counts: Counter[str] = Counter()

    assignments = {
        "seq_id_a": ("HA", "seq_a"),
        "seq_id_b": ("NA", "seq_b"),
        "seq_id_c": ("HA", "seq_c"),
        "seq_id_d": ("NA", "seq_d"),
    }
    for id_col, (segment, seq_col) in assignments.items():
        ids: list[str] = []
        for sequence in raw[seq_col].tolist():
            seq_id, created = add_sequence_to_registry(
                registry_rows,
                lookup,
                used_ids,
                segment,
                str(sequence),
                embedding_files_dir,
            )
            ids.append(seq_id)
            if created:
                new_counts[segment] += 1
        raw[id_col] = ids
    add_qc_step(
        steps,
        name="assign_sequence_ids",
        action="looked up existing sequence IDs and registered new sequences",
        rows_before=len(raw),
        rows_after=len(raw),
        details={"new_sequences": {"HA": int(new_counts["HA"]), "NA": int(new_counts["NA"])}},
    )

    updated_registry = pd.DataFrame(registry_rows, columns=REGISTRY_COLUMNS)
    updated_registry["embedding_path"] = updated_registry["seq_id"].map(
        lambda seq_id: str(embedding_files_dir / f"matrix_{seq_id}.pt")
    )
    updated_registry["embedding_exists"] = updated_registry["seq_id"].map(
        lambda seq_id: str((embedding_files_dir / f"matrix_{seq_id}.pt").is_file()).lower()
    )
    write_registry(registry_csv, updated_registry)
    add_qc_step(
        steps,
        name="update_embedding_registry",
        action="refreshed embedding paths and embedding_exists flags",
        rows_before=len(raw),
        rows_after=len(raw),
        details={
            "registry_csv": str(registry_csv),
            "registry_rows": int(len(updated_registry)),
            "embedding_exists_false": int((updated_registry["embedding_exists"] == "false").sum()),
        },
    )

    aligned_ha = parse_fasta(aligned_ha_fasta)
    raw["serumHA"] = [
        aligned_ha.get(seq_id, sequence) for seq_id, sequence in zip(raw["seq_id_a"], raw["seq_a"])
    ]
    raw["virusHA"] = [
        aligned_ha.get(seq_id, sequence) for seq_id, sequence in zip(raw["seq_id_c"], raw["seq_c"])
    ]
    add_qc_step(
        steps,
        name="attach_aligned_ha",
        action="filled serumHA and virusHA from aligned HA FASTA when available",
        rows_before=len(raw),
        rows_after=len(raw),
        details={"aligned_ha_records": int(len(aligned_ha))},
    )

    output = raw.loc[:, OUTPUT_COLUMNS].copy()
    source_csv = processed_dir / "source.csv"
    output.to_csv(source_csv, index=False)
    add_qc_step(
        steps,
        name="write_processed_source",
        action="wrote processed source.csv",
        rows_before=len(raw),
        rows_after=len(output),
        details={"processed_source_csv": str(source_csv), "output_columns": OUTPUT_COLUMNS},
    )

    used_sequences: dict[str, str] = {}
    for id_col, seq_col in [("seq_id_a", "seq_a"), ("seq_id_b", "seq_b"), ("seq_id_c", "seq_c"), ("seq_id_d", "seq_d")]:
        for seq_id, sequence in zip(raw[id_col].tolist(), raw[seq_col].tolist()):
            used_sequences[str(seq_id)] = str(sequence)
    missing = [
        (seq_id, sequence)
        for seq_id, sequence in sorted(used_sequences.items())
        if not (embedding_files_dir / f"matrix_{seq_id}.pt").is_file()
    ]
    pending_fasta: Path | None = None
    if missing:
        pending_dir.mkdir(parents=True, exist_ok=True)
        pending_fasta = pending_dir / f"{timestamp}.fasta"
        with pending_fasta.open("w", encoding="utf-8") as handle:
            for seq_id, sequence in missing:
                handle.write(f">{seq_id}\n{sequence}\n")
    add_qc_step(
        steps,
        name="write_pending_fasta",
        action="wrote one pending FASTA for sequences without embedding, when needed",
        rows_before=len(output),
        rows_after=len(output),
        details={
            "missing_embedding_count": int(len(missing)),
            "pending_fasta": str(pending_fasta) if pending_fasta is not None else None,
        },
    )

    qc = {
        "dataset_dir": str(dataset_dir),
        "rows_input": rows_input,
        "rows_output": int(len(output)),
        "rows_dropped": rows_dropped,
        "steps": steps,
        "new_sequences": {"HA": int(new_counts["HA"]), "NA": int(new_counts["NA"])},
        "missing_embedding_count": int(len(missing)),
        "pending_fasta": str(pending_fasta) if pending_fasta is not None else None,
        "processed_source_csv": str(source_csv),
    }
    (processed_dir / "qc_summary.json").write_text(
        json.dumps(qc, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return qc


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Prepare dataset raw CSVs into processed/source.csv.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument(
        "--registry-csv",
        type=Path,
        default=repo_root / "data" / "embedding" / "registry" / "sequences.csv",
    )
    parser.add_argument(
        "--embedding-files-dir",
        type=Path,
        default=repo_root / "data" / "embedding" / "files",
    )
    parser.add_argument(
        "--pending-dir",
        type=Path,
        default=repo_root / "data" / "embedding" / "registry" / "pending",
    )
    parser.add_argument(
        "--aligned-ha-fasta",
        type=Path,
        default=repo_root / "data" / "deprecated" / "reverse_test" / "HA_aligned.fasta",
    )
    parser.add_argument("--timestamp", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    qc = prepare_dataset(
        dataset_dir=args.dataset_dir.expanduser().resolve(),
        registry_csv=args.registry_csv.expanduser().resolve(),
        embedding_files_dir=args.embedding_files_dir.expanduser().resolve(),
        pending_dir=args.pending_dir.expanduser().resolve(),
        aligned_ha_fasta=args.aligned_ha_fasta.expanduser().resolve() if args.aligned_ha_fasta else None,
        timestamp=timestamp,
    )
    print(json.dumps(qc, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

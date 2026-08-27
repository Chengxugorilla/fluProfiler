#!/usr/bin/env python3
"""Audit the paired HA1 and Full-HA inputs for the Crick ablation dataset.

This is deliberately read-only: it records incompatibilities that must be
resolved before a shared master manifest is constructed.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd


SUBTYPES = ("H1N1", "H3N2")
PAIR_KEY_COLUMNS = (
    "serumName",
    "virusName",
    "serumPassCat",
    "virusPassCat",
)
EXCLUDED_SEQUENCE_COLUMNS = {"seq_a", "seq_b", "seq_c", "seq_d", "serumHA", "virusHA"}


def csv_name(subtype: str) -> str:
    return f"data4model(Crick-{subtype}).csv"


def length_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    return {
        str(length): int(count)
        for length, count in sorted(Counter(frame[column].astype(str).str.len()).items())
    }


def audit_subtype(ha1_path: Path, full_path: Path) -> dict:
    ha1 = pd.read_csv(ha1_path, keep_default_na=False)
    full = pd.read_csv(full_path, keep_default_na=False)
    common = [column for column in ha1.columns if column in full.columns]
    comparable = [column for column in common if column not in EXCLUDED_SEQUENCE_COLUMNS]

    report = {
        "ha1_file": str(ha1_path),
        "full_ha_file": str(full_path),
        "ha1_rows": len(ha1),
        "full_ha_rows": len(full),
        "same_row_count": len(ha1) == len(full),
        "row_order_mismatches": {},
        "full_ha_length_counts": {
            column: length_counts(full, column) for column in ("seq_a", "seq_c")
        },
    }
    if len(ha1) != len(full):
        report["status"] = "blocked"
        report["reason"] = "HA1 and Full-HA tables have different row counts."
        return report

    for column in comparable:
        mismatches = int((ha1[column].astype(str) != full[column].astype(str)).sum())
        if mismatches:
            report["row_order_mismatches"][column] = mismatches

    missing_pair_columns = [column for column in PAIR_KEY_COLUMNS if column not in common]
    report["missing_pair_key_columns"] = missing_pair_columns
    report["labels_match_in_row_order"] = (
        "label" in common
        and int((ha1["label"].astype(str) != full["label"].astype(str)).sum()) == 0
    )
    report["status"] = "ready_for_master" if not report["row_order_mismatches"] else "blocked"
    if report["status"] == "blocked":
        report["reason"] = (
            "Shared-master key fields differ between sources; choose an authoritative "
            "metadata source or provide a reconciliation table before building views."
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--ha1-dir", type=Path, required=True)
    parser.add_argument("--full-ha-dir", type=Path, required=True)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    report = {
        "dataset_dir": str(dataset_dir),
        "pair_key_columns": list(PAIR_KEY_COLUMNS),
        "subtypes": {},
    }
    for subtype in SUBTYPES:
        report["subtypes"][subtype] = audit_subtype(
            args.ha1_dir.resolve() / csv_name(subtype),
            args.full_ha_dir.resolve() / csv_name(subtype),
        )
    report["status"] = (
        "ready_for_master"
        if all(item["status"] == "ready_for_master" for item in report["subtypes"].values())
        else "blocked"
    )
    output = dataset_dir / "validation" / "input_audit_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(output)
    if report["status"] != "ready_for_master":
        raise SystemExit("Input audit found blocking inconsistencies; see report.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Trial extraction of H3 HA1 sequences from Crick H3N2 data4model CSV."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


START_RE = re.compile(r"QK[A-Z]PGN[A-Z]{2}")
FUSION_FALLBACKS = ("GIFGA", "GLFGA", "GILGA", "GMFGA", "GIXGA", "SIFGA")


def extract_h3_ha1(sequence: str) -> tuple[str, dict[str, int]]:
    seq = "".join(str(sequence).strip().upper().split())

    start = 0
    start_match = START_RE.search(seq[:80])
    if start_match:
        start = start_match.start()
    elif len(seq) > 330 and seq.startswith("M"):
        start = 16

    end = len(seq)
    qtr = seq.find("QTR", start + 250)
    if qtr != -1:
        end = qtr + 3
    else:
        hits = [seq.find(motif, start + 250) for motif in FUSION_FALLBACKS]
        hits = [hit for hit in hits if hit != -1]
        if hits:
            end = min(hits)

    return seq[start:end], {
        "input_len": len(seq),
        "output_len": end - start,
        "start_trim": start,
        "end_trim": len(seq) - end,
    }


def counter_to_dict(counter: Counter[int]) -> dict[str, int]:
    return {str(k): v for k, v in sorted(counter.items())}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/dataset/H1H3_HA1/raw/data4model(Crick-H3N2).csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/dataset/H1H3_HA1/trial/data4model(Crick-H3N2).HA1_trial.csv"
        ),
    )
    parser.add_argument(
        "--qc",
        type=Path,
        default=Path(
            "data/dataset/H1H3_HA1/trial/data4model(Crick-H3N2).HA1_trial.qc.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.qc.parent.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "input_csv": str(args.input),
        "output_csv": str(args.output),
        "rule": (
            "For seq_a and seq_c, trim N terminus to first QK?PGN?? within "
            "first 80 aa when present; otherwise use 16-aa signal peptide "
            "fallback for full HA. Trim C terminus after QTR found after HA1 "
            "position 250, with HA2 fusion-peptide fallback motifs when QTR "
            "is absent."
        ),
        "columns_trimmed": ["seq_a", "seq_c"],
        "columns_unchanged": ["seq_b", "seq_d"],
        "rows": 0,
        "columns": {},
    }
    for col in ("seq_a", "seq_c"):
        summary["columns"][col] = {
            "changed_rows": 0,
            "unchanged_rows": 0,
            "before_lengths": Counter(),
            "after_lengths": Counter(),
            "start_trim": Counter(),
            "end_trim": Counter(),
            "non_329_examples": [],
        }

    with args.input.open(newline="", encoding="utf-8") as fin, args.output.open(
        "w", newline="", encoding="utf-8"
    ) as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()

        for line_no, row in enumerate(reader, start=2):
            summary["rows"] += 1
            for col in ("seq_a", "seq_c"):
                old = row[col]
                old_norm = "".join(str(old).strip().upper().split())
                new, info = extract_h3_ha1(old)
                row[col] = new

                col_summary = summary["columns"][col]
                col_summary["before_lengths"][info["input_len"]] += 1
                col_summary["after_lengths"][info["output_len"]] += 1
                col_summary["start_trim"][info["start_trim"]] += 1
                col_summary["end_trim"][info["end_trim"]] += 1
                key = "changed_rows" if new != old_norm else "unchanged_rows"
                col_summary[key] += 1

                if (
                    info["output_len"] != 329
                    and len(col_summary["non_329_examples"]) < 25
                ):
                    col_summary["non_329_examples"].append(
                        {
                            "csv_line": line_no,
                            "serumName": row.get("serumName", ""),
                            "virusName": row.get("virusName", ""),
                            "input_len": info["input_len"],
                            "output_len": info["output_len"],
                            "start": new[:16],
                            "end": new[-16:],
                        }
                    )
            writer.writerow(row)

    for col_summary in summary["columns"].values():
        for key in ("before_lengths", "after_lengths", "start_trim", "end_trim"):
            col_summary[key] = counter_to_dict(col_summary[key])

    with args.qc.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(args.output)
    print(args.qc)


if __name__ == "__main__":
    main()

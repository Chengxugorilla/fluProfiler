#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


JOB_DIR = Path("/home/chenyh/workspace/fluProfiler/temp job")
DEFAULT_RECURRENT_FASTA = JOB_DIR / "H1N1_recurrent_clean.fasta"
DEFAULT_VACCINE_FASTA = JOB_DIR / "H1N1_vaccine_origin.fasta"
DEFAULT_EMBEDDING_DIR = JOB_DIR / "embedding"
DEFAULT_OUTPUT_CSV = JOB_DIR / "recurrent_vaccine_pairs.csv"


def parse_fasta_ids(fasta_path: Path, pipe_part: str = "first") -> list[str]:
    fasta_path = Path(fasta_path)
    ids: list[str] = []
    seen: set[str] = set()

    with fasta_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith(">"):
                continue
            seq_id = line[1:].strip().split()[0]
            if "|" in seq_id:
                parts = [part.strip() for part in seq_id.split("|")]
                if pipe_part == "first":
                    seq_id = parts[0]
                elif pipe_part == "second":
                    seq_id = parts[1] if len(parts) > 1 else parts[0]
                elif pipe_part == "full":
                    seq_id = "|".join(parts)
                else:
                    raise ValueError(f"Unsupported pipe_part: {pipe_part}")
            if seq_id and seq_id not in seen:
                ids.append(seq_id)
                seen.add(seq_id)

    return ids


def build_pair_rows(recurrent_ids: list[str], vaccine_ids: list[str], passage: str = "<EGG>") -> list[dict[str, str]]:
    return [
        {
            "seq_id_a": recurrent_id,
            "seq_id_c": vaccine_id,
            "serumPassCat": passage,
            "virusPassCat": passage,
        }
        for recurrent_id in recurrent_ids
        for vaccine_id in vaccine_ids
    ]


def validate_embedding_files(embedding_dir: Path, seq_ids: list[str]) -> None:
    missing = [
        f"matrix_{seq_id}.pt"
        for seq_id in seq_ids
        if not (Path(embedding_dir) / f"matrix_{seq_id}.pt").is_file()
    ]
    if missing:
        preview = ", ".join(missing[:10])
        suffix = f" and {len(missing) - 10} more" if len(missing) > 10 else ""
        raise FileNotFoundError(f"Missing {len(missing)} embedding file(s): {preview}{suffix}")


def write_pair_csv(rows: list[dict[str, str]], output_csv: Path) -> None:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["seq_id_a", "seq_id_c", "serumPassCat", "virusPassCat"]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build recurrent-vaccine pair CSV for fluProfiler inference.")
    parser.add_argument("--recurrent-fasta", type=Path, default=DEFAULT_RECURRENT_FASTA)
    parser.add_argument("--vaccine-fasta", type=Path, default=DEFAULT_VACCINE_FASTA)
    parser.add_argument("--embedding-dir", type=Path, default=DEFAULT_EMBEDDING_DIR)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--passage", default="<EGG>")
    parser.add_argument("--recurrent-pipe-part", choices=("first", "second", "full"), default="first")
    parser.add_argument("--vaccine-pipe-part", choices=("first", "second", "full"), default="second")
    parser.add_argument("--skip-embedding-check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    recurrent_ids = parse_fasta_ids(args.recurrent_fasta, pipe_part=args.recurrent_pipe_part)
    vaccine_ids = parse_fasta_ids(args.vaccine_fasta, pipe_part=args.vaccine_pipe_part)

    if not args.skip_embedding_check:
        validate_embedding_files(args.embedding_dir, recurrent_ids + vaccine_ids)

    rows = build_pair_rows(recurrent_ids, vaccine_ids, passage=args.passage)
    write_pair_csv(rows, args.output_csv)

    print(f"recurrent: {len(recurrent_ids)}")
    print(f"vaccine: {len(vaccine_ids)}")
    print(f"pairs: {len(rows)}")
    print(f"output: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

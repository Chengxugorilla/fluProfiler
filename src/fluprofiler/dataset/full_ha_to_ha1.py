#!/usr/bin/env python3
"""从 Full-HA source.csv 构建 Full-HA 与 HA1 的对齐映射。"""

import argparse
import subprocess
from pathlib import Path

import pandas as pd


PREFIX = {"H1N1": "H1", "H3N2": "H3"}


def write_fasta(records, path: Path) -> None:
    with path.open("w") as handle:
        for sequence_id, sequence in records:
            handle.write(f">{sequence_id}\n{sequence}\n")


def read_fasta(path: Path) -> dict[str, str]:
    records = {}
    sequence_id = None
    sequence = []

    for line in path.open():
        line = line.strip()
        if line.startswith(">"):
            if sequence_id is not None:
                records[sequence_id] = "".join(sequence)
            sequence_id, sequence = line[1:], []
        else:
            sequence.append(line)
    records[sequence_id] = "".join(sequence)
    return records


def export(source_csv: Path, output_dir: Path, subtype: str, prefix: str) -> None:
    source = pd.read_csv(source_csv)
    source = source[source["Type"] == subtype]
    sequences = pd.concat([source["seq_a"], source["seq_c"]]).drop_duplicates().tolist()
    sequence_ids = [f"{prefix}_{index:06d}" for index in range(1, len(sequences) + 1)]

    mapping = pd.DataFrame(
        {
            "full_ha_sequence": sequences,
            "full_ha_id": sequence_ids,
            "full_ha_aligned": "",
            "ha1_aligned": "",
        }
    )
    mapping.to_csv(output_dir / f"{prefix}.csv", index=False)
    write_fasta(zip(sequence_ids, sequences), output_dir / f"{prefix}_full.fasta")


def align(output_dir: Path, prefix: str) -> None:
    full_fasta = output_dir / f"{prefix}_full.fasta"
    map_fasta = output_dir / f"{prefix}_map.fasta"
    with map_fasta.open("w") as handle:
        subprocess.run(["mafft", "--auto", str(full_fasta)], stdout=handle, check=True)

    mapping_csv = output_dir / f"{prefix}.csv"
    mapping = pd.read_csv(mapping_csv, keep_default_na=False)
    mapping["full_ha_aligned"] = mapping["full_ha_id"].map(read_fasta(map_fasta))
    mapping.to_csv(mapping_csv, index=False)


def truncate(output_dir: Path, prefix: str, start: int, end: int) -> None:
    mapping_csv = output_dir / f"{prefix}.csv"
    mapping = pd.read_csv(mapping_csv, keep_default_na=False)
    mapping["ha1_aligned"] = mapping["full_ha_aligned"].str.slice(start - 1, end)
    mapping.to_csv(mapping_csv, index=False)
    write_fasta(
        zip(mapping["full_ha_id"], mapping["ha1_aligned"]),
        output_dir / f"{prefix}_truncated.fasta",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--subtype", choices=PREFIX, required=True)
    parser.add_argument("--step", choices=("export", "align", "truncate"), required=True)
    parser.add_argument("--ha1-start", type=int)
    parser.add_argument("--ha1-end", type=int)
    args = parser.parse_args()

    output_dir = args.dataset_dir / "main" / "HA1_sequences"
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = PREFIX[args.subtype]

    if args.step == "export":
        export(args.dataset_dir / "processed" / "source.csv", output_dir, args.subtype, prefix)
    elif args.step == "align":
        align(output_dir, prefix)
    else:
        truncate(output_dir, prefix, args.ha1_start, args.ha1_end)


if __name__ == "__main__":
    main()

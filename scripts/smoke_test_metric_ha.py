#!/usr/bin/env python3
"""
Run a tiny CPU smoke test for the Metric HA antigenic trainer.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _make_rows(split: str, count: int):
    rows = []
    for idx in range(count):
        rows.append(
            {
                "seq_id_a": f"{split}_serum_{idx % 3}",
                "seq_id_c": f"{split}_virus_{idx % 4}",
                "seq_b": "ANVTAAANST" if idx % 2 == 0 else "ANVTAAAAAA",
                "seq_d": "ANVTAAAAAA" if idx % 2 == 0 else "ANVTAAANST",
                "serumPassCat": "<EGG>" if idx % 2 == 0 else "<CELL>",
                "virusPassCat": "<CELL>" if idx % 2 == 0 else "<EGG>",
                "Type": "H3N2" if idx % 2 == 0 else "H1N1",
                "label": round(0.25 * (idx + 1), 4),
            }
        )
    return rows


def _write_dataset(split_dir: Path):
    import pandas as pd

    split_dir.mkdir(parents=True, exist_ok=True)
    frames = {
        "train": pd.DataFrame(_make_rows("train", 8)),
        "valid": pd.DataFrame(_make_rows("valid", 3)),
        "test": pd.DataFrame(_make_rows("test", 3)),
    }
    for name, frame in frames.items():
        frame.to_csv(split_dir / f"{name}.csv", index=False)

    sequence_ids = set()
    for frame in frames.values():
        sequence_ids.update(frame["seq_id_a"].tolist())
        sequence_ids.update(frame["seq_id_c"].tolist())
    return sorted(sequence_ids)


def _write_embeddings(embedding_dir: Path, sequence_ids, hidden_size: int = 4, seq_len: int = 3):
    import torch

    embedding_dir.mkdir(parents=True, exist_ok=True)
    for idx, seq_id in enumerate(sequence_ids):
        torch.manual_seed(idx)
        tensor = torch.randn(seq_len, hidden_size, dtype=torch.float32)
        torch.save(tensor, embedding_dir / f"matrix_{seq_id}.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Metric HA trainer smoke test on CPU.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--keep-output", action="store_true", default=False)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    with tempfile.TemporaryDirectory(prefix="fluprofiler_metric_smoke_") as tmp:
        tmpdir = Path(tmp)
        split_dir = tmpdir / "split"
        embedding_dir = tmpdir / "embedding"
        output_dir = tmpdir / "output"
        sequence_ids = _write_dataset(split_dir)
        _write_embeddings(embedding_dir, sequence_ids)
        artificial_csv = split_dir / "artificial_data.csv"

        subprocess.run(
            [
                args.python,
                str(repo_root / "scripts" / "build_metric_artificial_data.py"),
                "--data-dir",
                str(split_dir),
                "--output-csv",
                str(artificial_csv),
            ],
            cwd=repo_root,
            check=True,
        )

        cmd = [
            args.python,
            str(repo_root / "experiments" / "metric_antigenic" / "train_metric_ha.py"),
            "--data-dir",
            str(split_dir),
            "--embedding-dir",
            str(embedding_dir),
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--patience",
            "1",
            "--latent-dim",
            "2",
            "--gpu-cache-gb",
            "1",
            "--artificial-train-csv",
            str(artificial_csv),
            "--artificial-warmup-epochs",
            "1",
            "--no-use-lr-schedule",
        ]
        subprocess.run(cmd, cwd=repo_root, check=True)

        required_outputs = [
            output_dir / "run_config.json",
            output_dir / "metrics.jsonl",
            output_dir / "checkpoints" / "best_model.pth",
        ]
        missing = [path for path in required_outputs if not path.exists()]
        if missing:
            raise SystemExit(f"Smoke run did not create required output(s): {missing}")

        run_config = json.loads((output_dir / "run_config.json").read_text(encoding="utf-8"))
        if run_config.get("artificial_warmup_epochs") != 1:
            raise SystemExit("Smoke run did not enable one artificial warmup epoch")
        if run_config.get("warmup_train_rows") != 16:
            raise SystemExit("Smoke run did not mix real and artificial train rows for warmup")

        if args.keep_output:
            final_dir = repo_root / "results" / "smoke_metric_ha"
            if final_dir.exists():
                shutil.rmtree(final_dir)
            shutil.copytree(output_dir, final_dir)
            print(f"Kept smoke output at {final_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Minimal local HA-only smoke test.

This script creates a tiny synthetic dataset and embeddings under the system
temporary directory, launches the HA-only v2 trainer on CPU for one epoch, and
optionally removes the generated run directory afterwards.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

def _make_rows(count: int, prefix: str):
    rows = []
    for idx in range(count):
        rows.append(
            {
                "seq_id_a": f"{prefix}_serum_{idx % 4}",
                "seq_id_c": f"{prefix}_virus_{idx % 5}",
                "serumPassCat": "<CELL>",
                "virusPassCat": "<EGG>",
                "label": round(0.1 * (idx + 1), 4),
            }
        )
    return rows


def _write_embeddings(embedding_dir: Path, seq_ids, hidden_size: int = 2560, seq_len: int = 4):
    import torch

    embedding_dir.mkdir(parents=True, exist_ok=True)
    for seq_id in seq_ids:
        tensor = torch.randn(seq_len, hidden_size, dtype=torch.float32)
        torch.save(tensor, embedding_dir / f"matrix_{seq_id}.pt")


def _write_dataset(split_dir: Path):
    import pandas as pd

    split_dir.mkdir(parents=True, exist_ok=True)
    train_df = pd.DataFrame(_make_rows(12, "train"))
    test_df = pd.DataFrame(_make_rows(4, "test"))
    train_df.to_csv(split_dir / "train.csv", index=False)
    test_df.to_csv(split_dir / "test.csv", index=False)

    seq_ids = sorted(set(train_df["seq_id_a"]) | set(train_df["seq_id_c"]) | set(test_df["seq_id_a"]) | set(test_df["seq_id_c"]))
    return seq_ids


def _load_base_config(repo_root: Path):
    cfg_path = repo_root / "experiments" / "HA_only" / "config_v2_ha_only.json"
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def _snapshot_run_dirs(runs_root: Path):
    if not runs_root.exists():
        return set()
    return {p.resolve() for p in runs_root.rglob("*") if p.is_dir()}


def main():
    try:
        import pandas  # noqa: F401
        import torch  # noqa: F401
    except ModuleNotFoundError as exc:
        missing = exc.name or "required package"
        raise SystemExit(
            f"Missing dependency: {missing}. Run this smoke test inside an environment with torch and pandas installed."
        ) from exc

    parser = argparse.ArgumentParser(description="Run a local HA-only CPU smoke test.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--keep-run-dir", action="store_true", default=False)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    runs_root = repo_root / "runs"
    before_dirs = _snapshot_run_dirs(runs_root)

    with tempfile.TemporaryDirectory(prefix="fluprofiler_smoke_") as tmp:
        tmpdir = Path(tmp)
        split_dir = tmpdir / "split"
        embedding_dir = tmpdir / "embedding"
        seq_ids = _write_dataset(split_dir)
        _write_embeddings(embedding_dir, seq_ids)

        cfg = _load_base_config(repo_root)
        cfg["experiment"]["name"] = "ha_only_v2_smoke"
        cfg["experiment"]["run_tag"] = "smoke_cpu"
        cfg["experiment"]["exp_id"] = "smoke/ha_only"
        cfg["data"]["data_root"] = str(tmpdir)
        cfg["data"]["season_path"] = str(split_dir)
        cfg["data"]["embedding_root"] = str(embedding_dir)
        cfg["train"]["epochs"] = 1
        cfg["train"]["batch_size"] = 1
        cfg["train"]["patience"] = 1
        cfg["train"]["learning_rate"] = 8e-5
        cfg["runtime"]["device"] = "cpu"
        cfg["runtime"]["gpu_cache_gb"] = 1

        cfg_path = tmpdir / "smoke_config.json"
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

        cmd = [
            args.python,
            str(repo_root / "experiments" / "HA_only" / "train_v2_ha_only.py"),
            str(cfg_path),
            "--device",
            "cpu",
            "--epochs",
            "1",
            "--batch-size",
            "1",
        ]
        subprocess.run(cmd, cwd=repo_root, check=True)

    if not args.keep_run_dir:
        after_dirs = _snapshot_run_dirs(runs_root)
        new_dirs = sorted(after_dirs - before_dirs, key=lambda p: len(p.parts), reverse=True)
        for path in new_dirs:
            if path.exists() and path.is_dir():
                shutil.rmtree(path, ignore_errors=True)

        smoke_root = runs_root / "smoke"
        if smoke_root.exists():
            shutil.rmtree(smoke_root, ignore_errors=True)


if __name__ == "__main__":
    main()

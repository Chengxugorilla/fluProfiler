# src/utils/run_logger.py
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # 可选：没装tensorboard也能跑，只是不能写tb
    SummaryWriter = None  # type: ignore


def sha1_json(obj: Any) -> str:
    """Stable sha1 for any JSON-serializable object."""
    s = json.dumps(
        obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha1(s).hexdigest()


def git_state(repo_dir: Optional[Union[str, Path]] = None) -> Dict[str, str]:
    """Return git commit and dirty flag. Works offline; returns placeholders if not a git repo."""
    cwd = str(repo_dir) if repo_dir is not None else None
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        dirty = (
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=cwd, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        return {"commit": commit, "state": "dirty" if dirty else "clean"}
    except Exception:
        return {"commit": "nogit", "state": "unknown"}


def _json_default(o):
    # torch.device / Path / numpy 类型兜底
    try:
        import torch
        if isinstance(o, torch.device):
            return str(o)
    except Exception:
        pass
    try:
        from pathlib import Path as _Path
        if isinstance(o, _Path):
            return str(o)
    except Exception:
        pass
    try:
        import numpy as np
        if isinstance(o, (np.integer, np.floating)):
            return o.item()
    except Exception:
        pass
    return str(o)

def _dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)



@dataclass
class RunLogger:
    run_dir: Path
    run_id: str
    metrics_path: Path
    _tb: Any = field(default=None, init=False, repr=False)

    @classmethod
    def create(
        cls,
        run_root: Union[str, Path],
        name: str,
        config: Dict[str, Any],
        tag: str = "",
        make_subdirs: bool = True,
        repo_dir: Optional[Union[str, Path]] = None,
    ) -> "RunLogger":
        run_root = Path(run_root)
        ts = time.strftime("%Y%m%d_%H%M%S")

        gs = git_state(repo_dir=repo_dir)
        base = f"{ts}__{name}"
        if tag:
            base += f"__{tag}"
        base += f"__{gs['commit'][:8]}"

        run_dir = run_root / base
        i = 1
        while run_dir.exists():
            i += 1
            run_dir = run_root / f"{base}__{i}"

        run_dir.mkdir(parents=True, exist_ok=False)

        if make_subdirs:
            (run_dir / "checkpoints").mkdir(exist_ok=True)
            (run_dir / "artifacts").mkdir(exist_ok=True)
            (run_dir / "tb").mkdir(exist_ok=True)

        _dump_json(run_dir / "config.json", config)
        _dump_json(run_dir / "git.json", gs)

        return cls(
            run_dir=run_dir,
            run_id=run_dir.name,
            metrics_path=run_dir / "metrics.jsonl",
        )

    # --- TensorBoard ---
    def tb(self):
        if self._tb is None:
            if SummaryWriter is None:
                raise RuntimeError("TensorBoard SummaryWriter unavailable (tensorboard not installed).")
            self._tb = SummaryWriter(log_dir=str(self.run_dir / "tb"))
        return self._tb

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self.tb().add_scalar(tag, value, step)

    def flush_tb(self) -> None:
        if self._tb is not None:
            self._tb.flush()

    def close_tb(self) -> None:
        if self._tb is not None:
            self._tb.flush()
            self._tb.close()
            self._tb = None

    # --- Convenience writers ---
    def write_text(self, filename: str, text: str) -> None:
        (self.run_dir / filename).write_text(text, encoding="utf-8")

    def save_json(self, filename: str, obj: Any) -> None:
        _dump_json(self.run_dir / filename, obj)

    # --- Metrics logging (JSONL) ---
    def log(self, record: Dict[str, Any], split: Optional[str] = None, epoch: Optional[int] = None) -> None:
        rec = dict(record)
        rec["time"] = time.time()
        rec["run_id"] = self.run_id
        if split is not None:
            rec["split"] = split
        if epoch is not None:
            rec["epoch"] = epoch

        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # --- Common artifacts helpers ---
    def save_split(self, split: Dict[str, Any], filename: str = "split.json") -> str:
        split = dict(split)
        split["split_hash"] = sha1_json(split)
        self.save_json(filename, split)
        return split["split_hash"]

    def save_summary(self, summary: Dict[str, Any], filename: str = "summary.json") -> None:
        self.save_json(filename, summary)

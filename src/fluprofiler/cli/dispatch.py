from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    # src/fluprofiler/cli/dispatch.py -> src/fluprofiler/cli -> src/fluprofiler -> src -> repo root
    return Path(__file__).resolve().parents[3]


def _run(cmd: list[str], cwd: Path):
    print("[dispatch]", " ".join(map(str, cmd)))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main(argv: list[str] | None = None):
    repo_root = _repo_root()
    cwd = repo_root

    args = sys.argv[1:] if argv is None else list(argv)

    # Manual parsing:
    # - only extract --task/--config (and positional task/config as backward compat)
    # - everything else is forwarded to the selected training script (no argparse rejection)
    task = None
    impl = None
    config = None
    forwarded: list[str] = []

    i = 0
    while i < len(args):
        a = args[i]
        if a == "--task":
            if i + 1 >= len(args):
                raise SystemExit("Missing value for --task")
            task = args[i + 1]
            i += 2
            continue
        if a == "--impl":
            if i + 1 >= len(args):
                raise SystemExit("Missing value for --impl")
            impl = args[i + 1]
            i += 2
            continue
        if a == "--config":
            if i + 1 >= len(args):
                raise SystemExit("Missing value for --config")
            config = args[i + 1]
            i += 2
            continue
        forwarded.append(a)
        i += 1

    # backward compatible positional style:
    #   run_fluprofiler.sh ha_only_v2 [config_json] [other...]
    if task is None and forwarded:
        if not forwarded[0].startswith("-"):
            task = forwarded.pop(0)
        if config is None and forwarded and not forwarded[0].startswith("-"):
            config = forwarded.pop(0)

    if task is None:
        raise SystemExit(
            "Missing task. Use either:\n"
            "  --task ha_only --impl v2 [--config path] [extra args...]\n"
            "  --task ha_only --impl legacy [extra args...]\n"
            "  --task hana --impl v2 [--config path] [extra args...]\n"
            "or positional:\n"
            "  ha_only_v2 [config_json] [extra args...]"
        )

    task = task.strip()
    impl = (impl or "v2").strip().lower()

    # Backward compatibility aliases
    if task == "ha_only_v2":
        task, impl = "ha_only", "v2"
    elif task == "ha_only_legacy":
        task, impl = "ha_only", "legacy"
    elif task == "hana_v2":
        task, impl = "hana", "v2"
    elif task == "hana_legacy":
        task, impl = "hana", "legacy"

    if task == "ha_only" and impl == "legacy":
        if forwarded:
            print("[dispatch] warning: forwarding extra args to legacy trainer:", " ".join(forwarded))
        cmd = ["python", "experiments/HA_only/train_ha_only.py", *forwarded]
        _run(cmd, cwd=cwd)
        return

    if task == "ha_only" and impl == "v2":
        cfg = config or "experiments/HA_only/config_v2_ha_only.json"
        cfg_path = (repo_root / cfg).resolve() if not os.path.isabs(cfg) else Path(cfg)
        if not cfg_path.exists():
            raise SystemExit(f"Config not found: {cfg_path}")
        cmd = ["python", "experiments/HA_only/train_v2_ha_only.py", str(cfg_path), *forwarded]
        _run(cmd, cwd=cwd)
        return

    if task == "hana" and impl == "v2":
        cfg = config or "experiments/HANA/config_v2_hana.json"
        cfg_path = (repo_root / cfg).resolve() if not os.path.isabs(cfg) else Path(cfg)
        if not cfg_path.exists():
            raise SystemExit(f"Config not found: {cfg_path}")
        cmd = ["python", "experiments/HANA/train_v2_hana.py", str(cfg_path), *forwarded]
        _run(cmd, cwd=cwd)
        return

    if task == "hana" and impl == "legacy":
        raise SystemExit(
            "hana legacy entry is intentionally frozen and not exposed in the new dispatcher yet. "
            "Please use: --task hana --impl v2"
        )

    raise SystemExit(
        f"Unknown task/impl: task={task!r}, impl={impl!r}. "
        "Expected task in {ha_only, hana} and impl in {legacy, v2}."
    )


if __name__ == "__main__":
    main()


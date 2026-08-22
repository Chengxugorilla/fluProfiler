"""Evaluate a HA-mean, no-NA SerumGate-Minus checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__:
    from . import infer_minus_checkpoint as base_infer
else:
    script_dir = str(Path(__file__).resolve().parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    import infer_minus_checkpoint as base_infer

from fluprofiler.models.serum_gate_minus_all_mean_model import (
    SerumGateMinusAllMeanModel,
)


def run_inference(args: argparse.Namespace) -> dict[str, Any]:
    original_model_class = base_infer.SerumGateMinusModel
    base_infer.SerumGateMinusModel = SerumGateMinusAllMeanModel
    try:
        return base_infer.run_inference(args)
    finally:
        base_infer.SerumGateMinusModel = original_model_class


def main() -> None:
    result = run_inference(base_infer.parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

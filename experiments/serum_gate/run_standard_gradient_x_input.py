"""Gradient-times-input entry point for a standard SerumGate-Minus checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

import torch

import run_gradient_x_input as base

from fluprofiler.models.serum_gate_minus_model import SerumGateMinusModel


def load_standard_model(
    path: Path,
    device: str | torch.device,
) -> tuple[SerumGateMinusModel, dict]:
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    required = {"model_config", "model_state_dict", "passage_to_id"}
    if not required.issubset(checkpoint):
        raise ValueError(f"Checkpoint must contain: {', '.join(sorted(required))}")
    if checkpoint["model_config"].get("ha_pooling") == "lowrank_attention_only":
        raise ValueError("This entry point expects a standard, non-all-attention checkpoint")
    model = SerumGateMinusModel(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(target).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint


def run(args):
    base.load_model = load_standard_model
    return base.run(args)


if __name__ == "__main__":
    print(json.dumps(run(base.parser().parse_args()), ensure_ascii=False, indent=2))

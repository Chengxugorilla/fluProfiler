#!/usr/bin/env python3
"""Train SerumMutationSet-Minus with its condition and FiLM layers fused."""

from __future__ import annotations

import sys
from pathlib import Path


_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
sys.path.append(str(_REPO_ROOT / "src"))

import train_serum_mutation_set  # noqa: E402
from fluprofiler.models.serum_mutation_set_fused_film_model import (  # noqa: E402
    SerumMutationSetMinusFusedFiLMModel,
)


train_serum_mutation_set.SerumMutationSetMinusModel = SerumMutationSetMinusFusedFiLMModel


if __name__ == "__main__":
    train_serum_mutation_set.main()

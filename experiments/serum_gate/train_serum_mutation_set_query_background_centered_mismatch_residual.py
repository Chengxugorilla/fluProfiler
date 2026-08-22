#!/usr/bin/env python3
"""Train QueryBackground with a centered position-wise mismatch residual."""

from __future__ import annotations

import sys
from pathlib import Path


_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
sys.path.append(str(_REPO_ROOT / "src"))
sys.path.append(str(_THIS_FILE.parent))

import train_serum_mutation_set  # noqa: E402
from fluprofiler.models.serum_mutation_set_query_background_centered_mismatch_residual_model import (  # noqa: E402
    SerumMutationSetMinusQueryBackgroundCenteredMismatchResidualModel,
)


train_serum_mutation_set.SerumMutationSetMinusModel = (
    SerumMutationSetMinusQueryBackgroundCenteredMismatchResidualModel
)


if __name__ == "__main__":
    train_serum_mutation_set.main()

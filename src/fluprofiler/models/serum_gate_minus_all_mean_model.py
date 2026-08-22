"""SerumGate-Minus with pure masked-mean HA pooling and no NA branch.

The pretrained HA token embeddings are pooled directly by a masked arithmetic
mean and then projected to the latent dimension. No downstream self-attention,
attention scoring, attention-weighted pooling, or NA module is instantiated.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

import torch.nn as nn

from .serum_gate_minus_model import SerumGateMinusConfig, SerumGateMinusModel
from .serum_gate_model import SerumGateConfig, SerumGateModel


class SerumGateMinusAllMeanModel(SerumGateMinusModel):
    """SerumGate-Minus using only masked mean pooling for HA tokens."""

    def __init__(self, config: SerumGateMinusConfig | SerumGateConfig | Mapping[str, Any]):
        nn.Module.__init__(self)
        if isinstance(config, Mapping):
            config = SerumGateMinusConfig(**dict(config))
        elif not isinstance(config, SerumGateMinusConfig):
            config = SerumGateMinusConfig(**config.__dict__)
        if config.score_log_var_mode not in {"sum", "query"}:
            raise ValueError("score_log_var_mode must be 'sum' or 'query'")

        self.config = replace(config, ha_pooling="mean", na_branch="none")
        self.score_model = SerumGateModel(self.config)


__all__ = ["SerumGateMinusAllMeanModel"]

"""SerumMutationSet-Minus with the condition/FiLM linear maps fused.

This variant leaves the mutation-set encoder unchanged.  It replaces the
unfused sequence ``Linear -> ReLU -> Linear -> Linear`` used to derive FiLM
parameters with ``Linear -> ReLU -> Linear``.  The final two linear maps are
algebraically composable, so checkpoints from the original model can be
converted without changing predictions.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn

from fluprofiler.models.serum_mutation_set_model import (
    SerumMutationSetConfig,
    SerumMutationSetMinusModel,
)


class SerumMutationSetMinusFusedFiLMModel(SerumMutationSetMinusModel):
    """Independent model variant with a fused serum-condition FiLM head."""

    def __init__(
        self,
        config: SerumMutationSetConfig | Mapping[str, Any],
        ha_distance_matrix: torch.Tensor,
    ):
        super().__init__(config, ha_distance_matrix)
        condition_input_dim = (
            self.config.background_dim
            + self.config.passage_dim
            + self.config.subtype_dim
        )
        self.serum_condition_encoder = nn.Sequential(
            nn.Linear(condition_input_dim, self.config.theta_dim),
            nn.ReLU(),
        )
        self.film = nn.Linear(
            self.config.theta_dim,
            self.config.mutation_dim * 2,
        )

    @classmethod
    def from_unfused_model(
        cls,
        model: SerumMutationSetMinusModel,
    ) -> "SerumMutationSetMinusFusedFiLMModel":
        """Create an exactly equivalent fused model from an original model."""
        fused = cls(model.config, model.ha_distance_matrix)
        source = model.state_dict()
        target = fused.state_dict()

        for name, value in source.items():
            if name in target and target[name].shape == value.shape:
                target[name] = value.detach().clone()

        condition_second_weight = source["serum_condition_encoder.2.weight"]
        condition_second_bias = source["serum_condition_encoder.2.bias"]
        film_weight = source["film.weight"]
        film_bias = source["film.bias"]
        target["film.weight"] = film_weight @ condition_second_weight
        target["film.bias"] = film_weight @ condition_second_bias + film_bias

        fused.load_state_dict(target)
        return fused


__all__ = ["SerumMutationSetMinusFusedFiLMModel"]

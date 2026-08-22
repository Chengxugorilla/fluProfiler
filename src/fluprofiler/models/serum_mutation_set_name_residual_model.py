"""MutationSet-Minus with discrete, regularized serum and virus name offsets.

The sequence architecture is inherited unchanged from ``SerumMutationSetMinusModel``.
Name IDs are only looked up after the sequence prediction is complete; they do
not enter mutation encoding, attention, FiLM, or passage/subtype conditioning.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from fluprofiler.models.serum_mutation_set_model import (
    SerumMutationSetBatch,
    SerumMutationSetConfig,
    SerumMutationSetMinusModel,
    _label_bin_weights,
    _pairwise_ranking_loss,
    _weighted_masked_mean,
)


class SerumMutationSetMinusNameResidualModel(SerumMutationSetMinusModel):
    """Add independent scalar serum-name and virus-name residuals to ``mean``.

    Index zero is UNK and is fixed at zero through ``padding_idx=0``. Every
    nonzero index represents only one discrete normalized name, with no shared
    embedding geometry.
    """

    def __init__(
        self,
        config: SerumMutationSetConfig | Mapping[str, Any],
        ha_distance_matrix: torch.Tensor,
        serum_name_vocab_size: int,
        virus_name_vocab_size: int,
        serum_name_l2: float = 1e-3,
        virus_name_l2: float = 1e-2,
    ):
        super().__init__(config, ha_distance_matrix)
        if serum_name_vocab_size < 1 or virus_name_vocab_size < 1:
            raise ValueError("name vocabulary sizes must include UNK at index zero")
        if serum_name_l2 < 0 or virus_name_l2 < 0:
            raise ValueError("name L2 coefficients must be non-negative")
        self.serum_name_l2 = float(serum_name_l2)
        self.virus_name_l2 = float(virus_name_l2)
        self.serum_name_values = nn.Embedding(int(serum_name_vocab_size), 1, padding_idx=0)
        self.virus_name_values = nn.Embedding(int(virus_name_vocab_size), 1, padding_idx=0)
        nn.init.zeros_(self.serum_name_values.weight)
        nn.init.zeros_(self.virus_name_values.weight)

    @staticmethod
    def _name_ids(batch: SerumMutationSetBatch, field: str, shape: tuple[int, ...]) -> torch.Tensor:
        value = getattr(batch, field, None)
        if value is None:
            return torch.zeros(shape, dtype=torch.long, device=batch.reference_ha.device)
        if tuple(value.shape) != shape:
            raise ValueError(f"{field} must have shape {shape}, got {tuple(value.shape)}")
        return value.long().to(batch.reference_ha.device)

    def name_regularization(self) -> torch.Tensor:
        return (
            self.serum_name_values.weight[1:].square().sum() * self.serum_name_l2
            + self.virus_name_values.weight[1:].square().sum() * self.virus_name_l2
        )

    def forward(self, batch: SerumMutationSetBatch) -> dict[str, torch.Tensor]:
        out = super().forward(batch)
        sequence_mean = out["mean"]
        batch_size, query_count = sequence_mean.shape
        serum_ids = self._name_ids(batch, "serum_name_id", (batch_size,))
        virus_ids = self._name_ids(batch, "virus_name_id", (batch_size, query_count))
        serum_effect = self.serum_name_values(serum_ids).squeeze(-1).unsqueeze(-1)
        serum_effect = serum_effect * serum_ids.ne(0).to(serum_effect.dtype).unsqueeze(-1)
        serum_effect = serum_effect.expand_as(sequence_mean)
        virus_effect = self.virus_name_values(virus_ids).squeeze(-1)
        virus_effect = virus_effect * virus_ids.ne(0).to(virus_effect.dtype)
        mean = sequence_mean + serum_effect + virus_effect
        name_regularization = self.name_regularization()
        out.update({
            "mean": mean,
            "sequence_mean": sequence_mean,
            "serum_name_effect": serum_effect,
            "virus_name_effect": virus_effect,
            "name_regularization": name_regularization,
        })
        if batch.labels is not None:
            labels = batch.labels.float().to(mean.device)
            query_mask = batch.query_mask.float().to(mean.device) if batch.query_mask is not None else None
            huber = F.smooth_l1_loss(mean, labels, beta=0.5, reduction="none")
            label_weights = _label_bin_weights(labels, self.config.label_weight_thresholds, self.config.label_weight_values).to(mean.device)
            rank_loss = _pairwise_ranking_loss(mean, labels, query_mask, margin=self.config.rank_loss_margin, min_label_delta=self.config.rank_loss_min_label_delta)
            prediction_loss = _weighted_masked_mean(huber, label_weights, query_mask)
            prediction_loss = prediction_loss + self.config.rank_loss_weight * rank_loss
            out.update({
                "rank_loss": rank_loss,
                "prediction_loss": prediction_loss,
                "huber_loss": prediction_loss + name_regularization,
            })
        return out


__all__ = ["SerumMutationSetMinusNameResidualModel"]

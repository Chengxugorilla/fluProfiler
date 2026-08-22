"""Query-background MutationSet variant with a position-wise mismatch residual."""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F

from fluprofiler.models.serum_mutation_set_model import (
    UNKNOWN_AMINO_ACID_ID,
    SerumMutationSetBatch,
    SerumMutationSetConfig,
    _label_bin_weights,
    _pairwise_ranking_loss,
    _weighted_masked_mean,
)
from fluprofiler.models.serum_mutation_set_query_background_model import (
    SerumMutationSetMinusQueryBackgroundModel,
)


class SerumMutationSetMinusQueryBackgroundMismatchResidualModel(
    SerumMutationSetMinusQueryBackgroundModel
):
    """Add a zero-initialized linear residual over aligned HA mismatch sites."""

    def __init__(
        self,
        config: SerumMutationSetConfig | Mapping[str, Any],
        ha_distance_matrix: torch.Tensor,
    ):
        super().__init__(config, ha_distance_matrix)
        self.mismatch_position_weight = torch.nn.Parameter(
            torch.zeros(self.config.max_position_embeddings)
        )

    def _mismatch_residual(self, batch: SerumMutationSetBatch) -> torch.Tensor:
        token_count = batch.query_aa.shape[-1]
        if token_count > self.mismatch_position_weight.numel():
            raise ValueError(
                "query alignment length exceeds mismatch residual positions: "
                f"{token_count} > {self.mismatch_position_weight.numel()}"
            )
        reference_aa = batch.reference_aa[:, None, :]
        valid = (
            (batch.reference_aligned_mask[:, None, :] > 0)
            & (batch.query_aligned_mask > 0)
            & (reference_aa != UNKNOWN_AMINO_ACID_ID)
            & (batch.query_aa != UNKNOWN_AMINO_ACID_ID)
        )
        mismatch = valid & (reference_aa != batch.query_aa)
        weights = self.mismatch_position_weight[:token_count]
        return (mismatch.to(weights.dtype) * weights).sum(dim=-1)

    def forward(
        self,
        batch: SerumMutationSetBatch | Mapping[str, Any],
    ) -> dict[str, torch.Tensor]:
        batch = self._as_batch(batch)
        out = super().forward(batch)
        sequence_mean = out["mean"]
        mismatch_residual = self._mismatch_residual(batch)
        mean = sequence_mean + mismatch_residual
        out.update(
            {
                "mean": mean,
                "sequence_mean": sequence_mean,
                "mismatch_residual": mismatch_residual,
            }
        )
        if batch.labels is not None:
            labels = batch.labels.float().to(mean.device)
            query_mask = (
                batch.query_mask.float().to(mean.device)
                if batch.query_mask is not None
                else None
            )
            huber = F.smooth_l1_loss(mean, labels, beta=0.5, reduction="none")
            label_weights = _label_bin_weights(
                labels,
                self.config.label_weight_thresholds,
                self.config.label_weight_values,
            ).to(mean.device)
            rank_loss = _pairwise_ranking_loss(
                mean,
                labels,
                query_mask,
                margin=self.config.rank_loss_margin,
                min_label_delta=self.config.rank_loss_min_label_delta,
            )
            out["rank_loss"] = rank_loss
            out["huber_loss"] = _weighted_masked_mean(
                huber,
                label_weights,
                query_mask,
            ) + self.config.rank_loss_weight * rank_loss
        return out


__all__ = ["SerumMutationSetMinusQueryBackgroundMismatchResidualModel"]

"""Query-background model with a centered position-wise mismatch residual."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from fluprofiler.models.serum_mutation_set_model import (
    UNKNOWN_AMINO_ACID_ID,
    SerumMutationSetBatch,
    SerumMutationSetConfig,
)
from fluprofiler.models.serum_mutation_set_query_background_mismatch_residual_model import (
    SerumMutationSetMinusQueryBackgroundMismatchResidualModel,
)


class SerumMutationSetMinusQueryBackgroundCenteredMismatchResidualModel(
    SerumMutationSetMinusQueryBackgroundMismatchResidualModel
):
    """Learn relative site effects while removing the uniform mismatch-count effect."""

    def __init__(
        self,
        config: SerumMutationSetConfig | Mapping[str, Any],
        ha_distance_matrix: torch.Tensor,
    ):
        super().__init__(config, ha_distance_matrix)

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
        valid_float = valid.to(weights.dtype)
        valid_weight_mean = (
            (valid_float * weights).sum(dim=-1, keepdim=True)
            / valid_float.sum(dim=-1, keepdim=True).clamp_min(1.0)
        )
        centered_weights = weights - valid_weight_mean
        return (mismatch.to(weights.dtype) * centered_weights).sum(dim=-1)


__all__ = ["SerumMutationSetMinusQueryBackgroundCenteredMismatchResidualModel"]

"""
SerumGate-Minus model for ratio-aware antigenic distance prediction.

The wrapped score model predicts a latent log2 titer-like reactivity score.
SerumGate-Minus exposes antigenic distance as:

    f(reference, reference) - f(reference, query)
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .serum_gate_model import (
    SerumGateBatch,
    SerumGateConfig,
    SerumGateModel,
    label_bin_weights,
    pairwise_ranking_loss,
    weighted_masked_mean,
)


@dataclass
class SerumGateMinusConfig(SerumGateConfig):
    score_log_var_mode: str = "sum"


class SerumGateMinusModel(nn.Module):
    def __init__(self, config: SerumGateMinusConfig | SerumGateConfig | Mapping[str, Any]):
        super().__init__()
        if isinstance(config, Mapping):
            config = SerumGateMinusConfig(**dict(config))
        elif not isinstance(config, SerumGateMinusConfig):
            config = SerumGateMinusConfig(**config.__dict__)
        if config.score_log_var_mode not in {"sum", "query"}:
            raise ValueError("score_log_var_mode must be 'sum' or 'query'")
        self.config = config
        self.score_model = SerumGateModel(config)

    @staticmethod
    def _as_batch(batch: SerumGateBatch | Mapping[str, Any]) -> SerumGateBatch:
        return SerumGateModel._as_batch(batch)

    def _without_labels(self, batch: SerumGateBatch) -> SerumGateBatch:
        return replace(batch, labels=None)

    def _self_query_batch(self, batch: SerumGateBatch) -> SerumGateBatch:
        batch_size = int(batch.reference_ha.shape[0])
        self_query_passage = None
        if batch.serum_passage is not None:
            self_query_passage = batch.serum_passage.view(batch_size, 1)
        elif batch.query_passage is not None:
            self_query_passage = torch.zeros(
                batch_size,
                1,
                dtype=batch.query_passage.dtype,
                device=batch.query_passage.device,
            )

        self_passage_pair = None
        if self_query_passage is not None and batch.serum_passage is not None:
            self_passage_pair = batch.serum_passage.view(batch_size, 1) * int(
                self.config.passage_vocab_size
            ) + self_query_passage
        elif batch.passage_pair is not None:
            self_passage_pair = torch.zeros(
                batch_size,
                1,
                dtype=batch.passage_pair.dtype,
                device=batch.passage_pair.device,
            )

        self_query_mask = None
        if batch.query_mask is not None:
            self_query_mask = torch.ones(
                batch_size,
                1,
                dtype=batch.query_mask.dtype,
                device=batch.query_mask.device,
            )

        return replace(
            batch,
            query_ha=batch.reference_ha[:, None, :, :],
            query_ha_mask=batch.reference_ha_mask[:, None, :] if batch.reference_ha_mask is not None else None,
            query_na=batch.reference_na[:, None, :, :] if batch.reference_na is not None else None,
            query_na_mask=batch.reference_na_mask[:, None, :] if batch.reference_na_mask is not None else None,
            query_passage=self_query_passage,
            passage_pair=self_passage_pair,
            s_nagly=torch.zeros(batch_size, 1, dtype=batch.reference_ha.dtype, device=batch.reference_ha.device),
            labels=None,
            query_mask=self_query_mask,
        )

    def _combine_log_var(self, self_log_var: torch.Tensor, query_log_var: torch.Tensor) -> torch.Tensor:
        if self.config.score_log_var_mode == "query":
            return query_log_var
        var = torch.exp(self_log_var) + torch.exp(query_log_var)
        return torch.log(var.clamp_min(1e-12)).clamp(
            self.config.min_log_var,
            self.config.max_log_var,
        )

    def forward(self, batch: SerumGateBatch | Mapping[str, Any]) -> dict[str, torch.Tensor]:
        batch = self._as_batch(batch)
        query_out = self.score_model(self._without_labels(batch))
        self_out = self.score_model(self._self_query_batch(batch))

        query_score = query_out["mean"]
        self_score = self_out["mean"].expand_as(query_score)
        query_log_var = query_out["log_var"]
        self_log_var = self_out["log_var"].expand_as(query_log_var)
        mean = self_score - query_score
        log_var = self._combine_log_var(self_log_var, query_log_var)

        out: dict[str, torch.Tensor] = {
            "mean": mean,
            "log_var": log_var,
            "self_score": self_score,
            "query_score": query_score,
            "self_log_var": self_log_var,
            "query_log_var": query_log_var,
        }

        if batch.labels is not None:
            labels = batch.labels.float().to(mean.device)
            query_mask = batch.query_mask.float().to(mean.device) if batch.query_mask is not None else None
            huber = F.smooth_l1_loss(mean, labels, beta=0.5, reduction="none")
            nll = 0.5 * (log_var + (labels - mean).pow(2) * torch.exp(-log_var))
            loss_weights = label_bin_weights(
                labels,
                self.config.label_weight_thresholds,
                self.config.label_weight_values,
            ).to(mean.device)
            rank_loss = pairwise_ranking_loss(
                mean,
                labels,
                query_mask,
                margin=self.config.rank_loss_margin,
                min_label_delta=self.config.rank_loss_min_label_delta,
            )
            rank_term = float(self.config.rank_loss_weight) * rank_loss
            out["rank_loss"] = rank_loss
            out["huber_loss"] = weighted_masked_mean(huber, loss_weights, query_mask) + rank_term
            out["nll_loss"] = weighted_masked_mean(nll, loss_weights, query_mask) + rank_term

        return out

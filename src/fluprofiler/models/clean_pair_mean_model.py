"""Standalone attention-free model for aligned HA antigenic-distance prediction.

This module intentionally depends only on PyTorch. It does not inherit from or
import any existing fluProfiler model. Pair features are extracted at aligned
HA sites before equal-weight masked mean pooling. NA inputs are not part of the
input contract. The model predicts only the signed response; it does not carry
an uncalibrated uncertainty branch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class CleanPairMeanConfig:
    hidden_size: int
    token_dim: int = 64
    site_hidden_dim: int = 256
    pair_dim: int = 8
    passage_vocab_size: int = 4
    passage_dim: int = 8
    metric_dim: int = 64
    dropout: float = 0.1
    label_weight_thresholds: tuple[float, ...] = (2.0, 4.0, 6.0)
    label_weight_values: tuple[float, ...] = (1.0, 1.3, 1.8, 2.5)
    rank_loss_weight: float = 0.0
    rank_loss_margin: float = 0.1
    rank_loss_min_label_delta: float = 0.0

    @classmethod
    def from_any(cls, config: CleanPairMeanConfig | Mapping[str, Any] | Any) -> CleanPairMeanConfig:
        if isinstance(config, cls):
            return config
        source = config if isinstance(config, Mapping) else vars(config)

        def value(name: str, default: Any) -> Any:
            return source.get(name, default)

        return cls(
            hidden_size=int(source["hidden_size"]),
            token_dim=int(value("token_dim", value("distance_hidden_dim", 64))),
            site_hidden_dim=int(value("site_hidden_dim", value("predictor_hidden_dim", 256))),
            pair_dim=int(value("pair_dim", value("latent_dim", 8))),
            passage_vocab_size=int(value("passage_vocab_size", 4)),
            passage_dim=int(value("passage_dim", 8)),
            metric_dim=int(value("metric_dim", value("distance_hidden_dim", 64))),
            dropout=float(value("dropout", 0.1)),
            label_weight_thresholds=tuple(value("label_weight_thresholds", (2.0, 4.0, 6.0))),
            label_weight_values=tuple(value("label_weight_values", (1.0, 1.3, 1.8, 2.5))),
            rank_loss_weight=float(value("rank_loss_weight", 0.0)),
            rank_loss_margin=float(value("rank_loss_margin", 0.1)),
            rank_loss_min_label_delta=float(value("rank_loss_min_label_delta", 0.0)),
        )

    def __post_init__(self) -> None:
        positive = {
            "hidden_size": self.hidden_size,
            "token_dim": self.token_dim,
            "site_hidden_dim": self.site_hidden_dim,
            "pair_dim": self.pair_dim,
            "passage_vocab_size": self.passage_vocab_size,
            "passage_dim": self.passage_dim,
            "metric_dim": self.metric_dim,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if len(self.label_weight_values) != len(self.label_weight_thresholds) + 1:
            raise ValueError(
                "label_weight_values must contain one more value than label_weight_thresholds"
            )


@dataclass
class CleanPairMeanBatch:
    reference_ha: torch.Tensor
    query_ha: torch.Tensor
    reference_ha_mask: torch.Tensor | None = None
    query_ha_mask: torch.Tensor | None = None
    serum_passage: torch.Tensor | None = None
    query_passage: torch.Tensor | None = None
    labels: torch.Tensor | None = None
    query_mask: torch.Tensor | None = None


def _weighted_mean(
    values: torch.Tensor,
    weights: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    combined = weights.to(values.dtype)
    if mask is not None:
        combined = combined * mask.to(device=values.device, dtype=values.dtype)
    return (values * combined).sum() / combined.sum().clamp_min(1.0)


def _label_weights(labels: torch.Tensor, config: CleanPairMeanConfig) -> torch.Tensor:
    weights = torch.full_like(labels, float(config.label_weight_values[0]))
    for threshold, value in zip(
        config.label_weight_thresholds,
        config.label_weight_values[1:],
    ):
        weights = torch.where(
            labels >= float(threshold),
            torch.full_like(weights, float(value)),
            weights,
        )
    return weights


def _ranking_loss(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor | None,
    margin: float,
    min_label_delta: float,
) -> torch.Tensor:
    label_delta = labels.unsqueeze(-1) - labels.unsqueeze(-2)
    prediction_delta = predictions.unsqueeze(-1) - predictions.unsqueeze(-2)
    valid = torch.triu(
        torch.ones(label_delta.shape[-2:], dtype=torch.bool, device=labels.device),
        diagonal=1,
    )
    while valid.ndim < label_delta.ndim:
        valid = valid.unsqueeze(0)
    valid = valid & (label_delta != 0) & (label_delta.abs() >= float(min_label_delta))
    if mask is not None:
        observed = mask > 0
        valid = valid & observed.unsqueeze(-1) & observed.unsqueeze(-2)
    if not bool(valid.any()):
        return predictions.sum() * 0.0
    signed_prediction = torch.sign(label_delta) * prediction_delta
    return F.relu(float(margin) - signed_prediction)[valid].mean()


class CleanPairMeanModel(nn.Module):
    """Aligned-site pair encoder with equal-weight pooling and signed response."""

    def __init__(self, config: CleanPairMeanConfig | Mapping[str, Any] | Any):
        super().__init__()
        self.config = CleanPairMeanConfig.from_any(config)
        cfg = self.config

        self.token_norm = nn.LayerNorm(cfg.hidden_size)
        self.token_projection = nn.Linear(cfg.hidden_size, cfg.token_dim)
        self.site_encoder = nn.Sequential(
            nn.Linear(cfg.token_dim * 4, cfg.site_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.site_hidden_dim, cfg.pair_dim),
            nn.GELU(),
        )
        self.passage_embedding = nn.Embedding(cfg.passage_vocab_size, cfg.passage_dim)
        self.passage_encoder = nn.Sequential(
            nn.Linear(cfg.passage_dim * 4, cfg.pair_dim),
            nn.GELU(),
        )
        pair_feature_dim = cfg.pair_dim * 2
        self.pair_dropout = nn.Dropout(cfg.dropout)
        self.mean_head = nn.Sequential(
            nn.Linear(pair_feature_dim, cfg.metric_dim, bias=False),
            nn.GELU(),
            nn.Linear(cfg.metric_dim, 1, bias=False),
        )

    @staticmethod
    def _field(batch: CleanPairMeanBatch | Mapping[str, Any] | Any, name: str) -> Any:
        if isinstance(batch, Mapping):
            return batch.get(name)
        return getattr(batch, name, None)

    def _validate_inputs(self, reference: torch.Tensor, query: torch.Tensor) -> None:
        if reference.ndim != 3:
            raise ValueError("reference_ha must have shape [batch, length, hidden_size]")
        if query.ndim != 4:
            raise ValueError("query_ha must have shape [batch, queries, length, hidden_size]")
        if reference.shape[0] != query.shape[0]:
            raise ValueError("reference_ha and query_ha batch dimensions must match")
        if reference.shape[1] != query.shape[2]:
            raise ValueError("reference and query HA embeddings must share an aligned length")
        if reference.shape[2] != self.config.hidden_size or query.shape[3] != self.config.hidden_size:
            raise ValueError("HA embedding hidden size does not match model configuration")

    @staticmethod
    def _default_mask(matrix: torch.Tensor) -> torch.Tensor:
        return torch.ones(matrix.shape[:-1], dtype=matrix.dtype, device=matrix.device)

    @staticmethod
    def _site_features(query: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        delta = query - reference
        return torch.cat([query, reference, delta, delta.abs()], dim=-1)

    def forward(
        self,
        batch: CleanPairMeanBatch | Mapping[str, Any] | Any,
    ) -> dict[str, torch.Tensor]:
        reference = self._field(batch, "reference_ha")
        query = self._field(batch, "query_ha")
        if reference is None or query is None:
            raise ValueError("reference_ha and query_ha are required")
        self._validate_inputs(reference, query)

        batch_size, query_count, length = query.shape[:3]
        reference_mask = self._field(batch, "reference_ha_mask")
        query_mask_sites = self._field(batch, "query_ha_mask")
        if reference_mask is None:
            reference_mask = self._default_mask(reference)
        if query_mask_sites is None:
            query_mask_sites = self._default_mask(query)
        reference_mask = reference_mask.to(device=reference.device, dtype=reference.dtype)
        query_mask_sites = query_mask_sites.to(device=query.device, dtype=query.dtype)
        if tuple(reference_mask.shape) != tuple(reference.shape[:2]):
            raise ValueError("reference_ha_mask must match reference batch and length")
        if tuple(query_mask_sites.shape) != tuple(query.shape[:3]):
            raise ValueError("query_ha_mask must match query batch, query count, and length")

        reference_tokens = self.token_projection(self.token_norm(reference))
        query_tokens = self.token_projection(self.token_norm(query))
        reference_tokens = reference_tokens * reference_mask.unsqueeze(-1)
        query_tokens = query_tokens * query_mask_sites.unsqueeze(-1)
        reference_expanded = reference_tokens[:, None, :, :].expand(-1, query_count, -1, -1)

        observed = torch.maximum(reference_mask[:, None, :], query_mask_sites)
        pair_site_features = self._site_features(query_tokens, reference_expanded)
        self_site_features = self._site_features(reference_expanded, reference_expanded)
        site_effect = self.site_encoder(pair_site_features) - self.site_encoder(self_site_features)
        site_effect = site_effect * observed.unsqueeze(-1)
        pooled_site_effect = site_effect.sum(dim=2) / observed.sum(dim=2).clamp_min(1.0).unsqueeze(-1)

        serum_passage = self._field(batch, "serum_passage")
        query_passage = self._field(batch, "query_passage")
        if serum_passage is None:
            serum_passage = torch.zeros(batch_size, dtype=torch.long, device=reference.device)
        if query_passage is None:
            query_passage = torch.zeros(
                batch_size,
                query_count,
                dtype=torch.long,
                device=reference.device,
            )
        serum_passage = serum_passage.long().to(reference.device).view(batch_size)
        query_passage = query_passage.long().to(reference.device).view(batch_size, query_count)
        serum_passage_emb = self.passage_embedding(serum_passage)[:, None, :].expand(-1, query_count, -1)
        query_passage_emb = self.passage_embedding(query_passage)
        passage_features = self._site_features(query_passage_emb, serum_passage_emb)
        self_passage_features = self._site_features(serum_passage_emb, serum_passage_emb)
        passage_effect = self.passage_encoder(passage_features) - self.passage_encoder(
            self_passage_features
        )

        pair_representation = torch.cat([pooled_site_effect, passage_effect], dim=-1)
        pair_representation = self.pair_dropout(pair_representation)
        mean = self.mean_head(pair_representation).squeeze(-1)

        out = {
            "mean": mean,
            "pair_representation": pair_representation,
            "site_effect": pooled_site_effect,
            "passage_effect": passage_effect,
        }

        labels = self._field(batch, "labels")
        if labels is not None:
            labels = labels.to(device=mean.device, dtype=mean.dtype)
            if tuple(labels.shape) != tuple(mean.shape):
                raise ValueError("labels must match the model output shape")
            observation_mask = self._field(batch, "query_mask")
            if observation_mask is not None:
                observation_mask = observation_mask.to(device=mean.device, dtype=mean.dtype)
            weights = _label_weights(labels, self.config)
            huber = F.smooth_l1_loss(mean, labels, beta=0.5, reduction="none")
            rank_loss = _ranking_loss(
                mean,
                labels,
                observation_mask,
                self.config.rank_loss_margin,
                self.config.rank_loss_min_label_delta,
            )
            rank_term = self.config.rank_loss_weight * rank_loss
            out["rank_loss"] = rank_loss
            out["huber_loss"] = _weighted_mean(huber, weights, observation_mask) + rank_term

        return out


__all__ = [
    "CleanPairMeanBatch",
    "CleanPairMeanConfig",
    "CleanPairMeanModel",
]

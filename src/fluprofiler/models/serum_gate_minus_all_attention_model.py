"""SerumGate-Minus variant with pure low-rank HA attention pooling.

This module keeps the SerumGate-Minus scoring, conditioning, uncertainty,
and loss logic unchanged. Only the shared HA encoder is replaced: HA tokens
are pooled entirely with learned attention weights, with no HA mean-pool path
and no HA attention/mean gate parameter.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

import torch
import torch.nn as nn

from .serum_gate_minus_model import SerumGateMinusConfig, SerumGateMinusModel
from .serum_gate_model import (
    CalibratedMetricPredictor,
    HAPoolEncoder,
    NAPairEffect,
    SerumGateConditionedPredictor,
    SerumGateConfig,
    SerumGateModel,
    SerumGatePriorEncoder,
)


class PureLowRankSelfAttentionPool(nn.Module):
    """Pool token embeddings using learned low-rank attention only."""

    def __init__(
        self,
        hidden_size: int,
        attention_dim: int = 128,
        attention_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if attention_dim <= 0:
            raise ValueError("ha_attention_dim must be positive")
        if attention_heads <= 0:
            raise ValueError("ha_attention_heads must be positive")
        if attention_dim % attention_heads != 0:
            raise ValueError("ha_attention_dim must be divisible by ha_attention_heads")

        self.norm = nn.LayerNorm(hidden_size)
        self.input_projection = nn.Linear(hidden_size, attention_dim)
        self.context_attention = nn.MultiheadAttention(
            embed_dim=attention_dim,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.score = nn.Linear(attention_dim, 1)
        self.dropout = nn.Dropout(p=dropout)

    @staticmethod
    def _valid_tokens(matrix: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if matrix.ndim != 3:
            raise ValueError("matrix must have shape [batch, length, hidden_size]")
        if mask is None:
            valid = torch.ones(matrix.shape[:2], dtype=torch.bool, device=matrix.device)
        else:
            if tuple(mask.shape) != tuple(matrix.shape[:2]):
                raise ValueError("mask must match the matrix batch and length dimensions")
            valid = mask.to(device=matrix.device) > 0
        if not bool(valid.any(dim=1).all()):
            raise ValueError("Every sequence must contain at least one valid token")
        return valid

    def forward(
        self,
        matrix: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        valid = self._valid_tokens(matrix, mask)
        context = self.input_projection(self.norm(matrix))
        context, _ = self.context_attention(
            context,
            context,
            context,
            key_padding_mask=None if mask is None else ~valid,
            need_weights=False,
        )
        logits = self.score(torch.tanh(self.dropout(context))).squeeze(-1)
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=1).masked_fill(~valid, 0.0)
        pooled = (matrix * weights.unsqueeze(-1)).sum(dim=1)
        if return_attention:
            return pooled, weights
        return pooled


class AllAttentionHAPoolEncoder(nn.Module):
    """Encode an HA token matrix using pure attention pooling."""

    def __init__(
        self,
        hidden_size: int,
        latent_dim: int,
        attention_dim: int = 128,
        attention_heads: int = 4,
        attention_dropout: float = 0.1,
    ):
        super().__init__()
        self.pooling = "lowrank_attention_only"
        self.dropout = nn.Dropout(p=0.1)
        self.pure_lowrank_pooler = PureLowRankSelfAttentionPool(
            hidden_size=hidden_size,
            attention_dim=attention_dim,
            attention_heads=attention_heads,
            dropout=attention_dropout,
        )
        self.projection = nn.Linear(hidden_size, latent_dim)

    def forward(self, matrix: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        pooled = self.pure_lowrank_pooler(self.dropout(matrix), mask)
        return self.projection(pooled)

    def encode_with_attention(
        self,
        matrix: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pooled, weights = self.pure_lowrank_pooler(
            self.dropout(matrix),
            mask,
            return_attention=True,
        )
        return self.projection(pooled), weights


class SerumGateAllAttentionScoreModel(SerumGateModel):
    """SerumGate score model whose shared HA encoder is pure attention."""

    def __init__(self, config: SerumGateConfig | Mapping[str, Any]):
        nn.Module.__init__(self)
        if isinstance(config, Mapping):
            config = SerumGateConfig(**dict(config))
        if config.predictor_arch not in {"conditioned_mlp", "calibrated_metric"}:
            raise ValueError(f"Unsupported predictor_arch: {config.predictor_arch}")
        if config.na_branch not in {"none", "pair"}:
            raise ValueError(f"Unsupported na_branch: {config.na_branch}")
        if config.na_pooling not in {"attention", "mean", "lowrank_attention"}:
            raise ValueError(f"Unsupported na_pooling: {config.na_pooling}")
        if config.na_latent_dim <= 0:
            raise ValueError("na_latent_dim must be positive")

        config = replace(config, ha_pooling="lowrank_attention_only")
        self.config = config
        self.ha_encoder = AllAttentionHAPoolEncoder(
            hidden_size=config.hidden_size,
            latent_dim=config.latent_dim,
            attention_dim=config.ha_attention_dim,
            attention_heads=config.ha_attention_heads,
            attention_dropout=config.ha_attention_dropout,
        )
        self.na_encoder = (
            HAPoolEncoder(
                config.hidden_size,
                config.na_latent_dim,
                pooling=config.na_pooling,
                attention_dim=config.ha_attention_dim,
                attention_heads=config.ha_attention_heads,
                attention_dropout=config.ha_attention_dropout,
                mean_gate_init=config.ha_mean_gate_init,
            )
            if config.na_branch != "none"
            else None
        )
        self.na_pair_effect = NAPairEffect(config) if config.na_branch == "pair" else None
        self.passage_embedding = nn.Embedding(config.passage_vocab_size, config.passage_dim)
        self.passage_pair_embedding = nn.Embedding(
            config.passage_pair_vocab_size,
            config.passage_dim,
        )
        self.subtype_embedding = (
            nn.Embedding(config.subtype_vocab_size, config.subtype_dim)
            if config.subtype_dim > 0
            else None
        )
        self.prior_encoder = SerumGatePriorEncoder(config)
        if config.predictor_arch == "conditioned_mlp":
            self.predictor = SerumGateConditionedPredictor(config)
        else:
            self.predictor = CalibratedMetricPredictor(config)


class SerumGateMinusAllAttentionModel(SerumGateMinusModel):
    """SerumGate-Minus with a pure-attention shared HA encoder."""

    def __init__(self, config: SerumGateMinusConfig | SerumGateConfig | Mapping[str, Any]):
        nn.Module.__init__(self)
        if isinstance(config, Mapping):
            config = SerumGateMinusConfig(**dict(config))
        elif not isinstance(config, SerumGateMinusConfig):
            config = SerumGateMinusConfig(**config.__dict__)
        if config.score_log_var_mode not in {"sum", "query"}:
            raise ValueError("score_log_var_mode must be 'sum' or 'query'")

        self.config = replace(config, ha_pooling="lowrank_attention_only")
        self.score_model = SerumGateAllAttentionScoreModel(self.config)


__all__ = [
    "AllAttentionHAPoolEncoder",
    "PureLowRankSelfAttentionPool",
    "SerumGateAllAttentionScoreModel",
    "SerumGateMinusAllAttentionModel",
]

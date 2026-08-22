"""Pair-site attention model with SerumGate-Minus score subtraction."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .serum_gate_model import (
    SerumGateBatch,
    SerumGateModel,
    label_bin_weights,
    pairwise_ranking_loss,
    weighted_masked_mean,
)


@dataclass
class PairSiteAttentionMinusConfig:
    hidden_size: int
    max_site_length: int
    site_proj_dim: int = 128
    d_model: int = 256
    num_layers: int = 2
    num_heads: int = 4
    transformer_ff_dim: int = 1024
    site_attention_dim: int = 128
    dropout: float = 0.1
    passage_vocab_size: int = 6
    passage_pair_vocab_size: int = 36
    subtype_vocab_size: int = 1
    passage_dim: int = 8
    subtype_dim: int = 8
    predictor_hidden_dim: int = 256
    min_log_var: float = -6.0
    max_log_var: float = 4.0
    label_weight_thresholds: tuple[float, ...] = (2.0, 4.0, 6.0)
    label_weight_values: tuple[float, ...] = (1.0, 1.3, 1.8, 2.5)
    rank_loss_weight: float = 0.0
    rank_loss_margin: float = 0.1
    rank_loss_min_label_delta: float = 0.0
    score_log_var_mode: str = "sum"


def _validate_config(config: PairSiteAttentionMinusConfig) -> None:
    positive_fields = (
        "hidden_size",
        "max_site_length",
        "site_proj_dim",
        "d_model",
        "num_layers",
        "num_heads",
        "transformer_ff_dim",
        "site_attention_dim",
        "passage_vocab_size",
        "passage_pair_vocab_size",
        "subtype_vocab_size",
        "passage_dim",
        "predictor_hidden_dim",
    )
    for name in positive_fields:
        if int(getattr(config, name)) <= 0:
            raise ValueError(f"{name} must be positive")
    if config.subtype_dim < 0:
        raise ValueError("subtype_dim must be non-negative")
    if config.d_model % config.num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads")
    if not 0.0 <= config.dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    if config.min_log_var > config.max_log_var:
        raise ValueError("min_log_var must not exceed max_log_var")
    if len(config.label_weight_values) != len(config.label_weight_thresholds) + 1:
        raise ValueError(
            "label_weight_values must have exactly one more entry than "
            "label_weight_thresholds"
        )
    if config.score_log_var_mode not in {"sum", "query"}:
        raise ValueError("score_log_var_mode must be 'sum' or 'query'")


class PairSiteAttentionScoreModel(nn.Module):
    """Predict a titer-like score from aligned serum-virus site pairs."""

    def __init__(self, config: PairSiteAttentionMinusConfig):
        super().__init__()
        self.config = config
        self.site_projection = nn.Linear(config.hidden_size, config.site_proj_dim)
        self.pair_projection = nn.Sequential(
            nn.Linear(config.site_proj_dim * 4, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.position_embedding = nn.Parameter(
            torch.empty(config.max_site_length, config.d_model)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.transformer_ff_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
            norm=nn.LayerNorm(config.d_model),
            enable_nested_tensor=False,
        )
        self.site_attention = nn.Sequential(
            nn.Linear(config.d_model, config.site_attention_dim),
            nn.Tanh(),
            nn.Linear(config.site_attention_dim, 1),
        )
        self.passage_embedding = nn.Embedding(
            config.passage_vocab_size,
            config.passage_dim,
        )
        self.passage_pair_embedding = nn.Embedding(
            config.passage_pair_vocab_size,
            config.passage_dim,
        )
        self.subtype_embedding = (
            nn.Embedding(config.subtype_vocab_size, config.subtype_dim)
            if config.subtype_dim > 0
            else None
        )
        prediction_input_dim = (
            config.d_model
            + config.passage_dim * 2
            + config.subtype_dim
            + 1
        )
        self.prediction_head = nn.Sequential(
            nn.Linear(prediction_input_dim, config.predictor_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.predictor_hidden_dim, 2),
        )
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    @staticmethod
    def _default_ids(reference: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.zeros(shape, dtype=torch.long, device=reference.device)

    def _encode_pairs(
        self,
        reference_ha: torch.Tensor,
        query_ha: torch.Tensor,
        reference_mask: torch.Tensor | None,
        query_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if reference_ha.ndim != 3:
            raise ValueError("reference_ha must have shape [batch, sites, hidden]")
        if query_ha.ndim != 4:
            raise ValueError("query_ha must have shape [batch, queries, sites, hidden]")
        batch_size, query_count, site_count, hidden_size = query_ha.shape
        if reference_ha.shape != (batch_size, site_count, hidden_size):
            raise ValueError("reference_ha and query_ha must share batch, site, and hidden dimensions")
        if hidden_size != self.config.hidden_size:
            raise ValueError(
                f"Expected hidden_size {self.config.hidden_size}, got {hidden_size}"
            )
        if site_count > self.config.max_site_length:
            raise ValueError(
                f"Input has {site_count} sites, exceeding max_site_length "
                f"{self.config.max_site_length}"
            )

        if reference_mask is None:
            reference_mask = torch.ones(
                batch_size,
                site_count,
                dtype=torch.bool,
                device=reference_ha.device,
            )
        else:
            reference_mask = reference_mask.to(device=reference_ha.device) > 0
        if query_mask is None:
            query_mask = torch.ones(
                batch_size,
                query_count,
                site_count,
                dtype=torch.bool,
                device=query_ha.device,
            )
        else:
            query_mask = query_mask.to(device=query_ha.device) > 0
        if reference_mask.shape != (batch_size, site_count):
            raise ValueError("reference_ha_mask must have shape [batch, sites]")
        if query_mask.shape != (batch_size, query_count, site_count):
            raise ValueError("query_ha_mask must have shape [batch, queries, sites]")

        pair_mask = reference_mask[:, None, :] | query_mask
        if bool((~pair_mask.any(dim=-1)).any()):
            raise ValueError("Every serum-virus pair must contain at least one valid HA site")

        reference_z = self.site_projection(reference_ha)
        reference_z = reference_z * reference_mask.unsqueeze(-1).to(reference_z.dtype)
        query_z = self.site_projection(query_ha)
        query_z = query_z * query_mask.unsqueeze(-1).to(query_z.dtype)
        reference_z = reference_z[:, None, :, :].expand(-1, query_count, -1, -1)
        pair_tokens = torch.cat(
            [
                reference_z,
                query_z,
                query_z - reference_z,
                torch.abs(query_z - reference_z),
            ],
            dim=-1,
        )
        hidden = self.pair_projection(pair_tokens)
        hidden = hidden + self.position_embedding[:site_count].view(
            1,
            1,
            site_count,
            self.config.d_model,
        )
        flat_hidden = hidden.reshape(batch_size * query_count, site_count, -1)
        flat_mask = pair_mask.reshape(batch_size * query_count, site_count)
        encoded = self.transformer(
            flat_hidden,
            src_key_padding_mask=~flat_mask,
        ).reshape(batch_size, query_count, site_count, -1)

        logits = self.site_attention(encoded).squeeze(-1)
        logits = logits.masked_fill(~pair_mask, torch.finfo(logits.dtype).min)
        alpha = torch.softmax(logits, dim=-1)
        alpha = alpha * pair_mask.to(alpha.dtype)
        alpha = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        pair_repr = torch.sum(alpha.unsqueeze(-1) * encoded, dim=-2)
        return pair_repr, alpha

    def forward(self, batch: SerumGateBatch) -> dict[str, torch.Tensor]:
        device = batch.reference_ha.device
        batch_size, query_count = batch.query_ha.shape[:2]
        pair_repr, site_attention = self._encode_pairs(
            batch.reference_ha,
            batch.query_ha,
            batch.reference_ha_mask,
            batch.query_ha_mask,
        )

        serum_passage = batch.serum_passage
        if serum_passage is None:
            serum_passage = self._default_ids(batch.reference_ha, (batch_size,))
        query_passage = batch.query_passage
        if query_passage is None:
            query_passage = self._default_ids(
                batch.reference_ha,
                (batch_size, query_count),
            )
        passage_pair = batch.passage_pair
        if passage_pair is None:
            passage_pair = (
                serum_passage[:, None] * int(self.config.passage_vocab_size)
                + query_passage
            )
        subtype = batch.subtype
        if subtype is None:
            subtype = self._default_ids(batch.reference_ha, (batch_size,))
        s_nagly = batch.s_nagly
        if s_nagly is None:
            s_nagly = torch.zeros(
                batch_size,
                query_count,
                dtype=batch.reference_ha.dtype,
                device=device,
            )

        query_passage_emb = self.passage_embedding(query_passage.long().to(device))
        passage_pair_emb = self.passage_pair_embedding(passage_pair.long().to(device))
        if self.subtype_embedding is None:
            subtype_emb = pair_repr.new_empty((batch_size, query_count, 0))
        else:
            subtype_emb = self.subtype_embedding(subtype.long().to(device))
            subtype_emb = subtype_emb[:, None, :].expand(-1, query_count, -1)
        glycan_feature = s_nagly.to(device).float().unsqueeze(-1)
        prediction_input = torch.cat(
            [
                pair_repr,
                query_passage_emb,
                passage_pair_emb,
                subtype_emb,
                glycan_feature,
            ],
            dim=-1,
        )
        prediction = self.prediction_head(prediction_input)
        mean = prediction[..., 0]
        log_var = prediction[..., 1].clamp(
            self.config.min_log_var,
            self.config.max_log_var,
        )
        return {
            "mean": mean,
            "log_var": log_var,
            "site_attention": site_attention,
            "pair_repr": pair_repr,
        }


class PairSiteAttentionMinusModel(nn.Module):
    def __init__(self, config: PairSiteAttentionMinusConfig | Mapping[str, Any]):
        super().__init__()
        if isinstance(config, Mapping):
            config = PairSiteAttentionMinusConfig(**dict(config))
        _validate_config(config)
        self.config = config
        self.score_model = PairSiteAttentionScoreModel(config)

    @staticmethod
    def _as_batch(batch: SerumGateBatch | Mapping[str, Any]) -> SerumGateBatch:
        return SerumGateModel._as_batch(batch)

    @staticmethod
    def _without_labels(batch: SerumGateBatch) -> SerumGateBatch:
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
            self_passage_pair = (
                batch.serum_passage.view(batch_size, 1)
                * int(self.config.passage_vocab_size)
                + self_query_passage
            )
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
            query_ha_mask=(
                batch.reference_ha_mask[:, None, :]
                if batch.reference_ha_mask is not None
                else None
            ),
            query_na=None,
            query_na_mask=None,
            query_passage=self_query_passage,
            passage_pair=self_passage_pair,
            s_nagly=torch.zeros(
                batch_size,
                1,
                dtype=batch.reference_ha.dtype,
                device=batch.reference_ha.device,
            ),
            labels=None,
            query_mask=self_query_mask,
        )

    def _combine_log_var(
        self,
        self_log_var: torch.Tensor,
        query_log_var: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.score_log_var_mode == "query":
            return query_log_var
        variance = torch.exp(self_log_var) + torch.exp(query_log_var)
        return torch.log(variance.clamp_min(1e-12)).clamp(
            self.config.min_log_var,
            self.config.max_log_var,
        )

    def forward(
        self,
        batch: SerumGateBatch | Mapping[str, Any],
    ) -> dict[str, torch.Tensor]:
        batch = self._as_batch(batch)
        query_out = self.score_model(self._without_labels(batch))
        self_out = self.score_model(self._self_query_batch(batch))

        query_score = query_out["mean"]
        self_score = self_out["mean"].expand_as(query_score)
        query_log_var = query_out["log_var"]
        self_log_var = self_out["log_var"].expand_as(query_log_var)
        mean = self_score - query_score
        log_var = self._combine_log_var(self_log_var, query_log_var)
        query_site_attention = query_out["site_attention"]
        self_site_attention = self_out["site_attention"].expand_as(
            query_site_attention
        )

        out: dict[str, torch.Tensor] = {
            "mean": mean,
            "log_var": log_var,
            "self_score": self_score,
            "query_score": query_score,
            "self_log_var": self_log_var,
            "query_log_var": query_log_var,
            "query_site_attention": query_site_attention,
            "self_site_attention": self_site_attention,
        }
        if batch.labels is not None:
            labels = batch.labels.float().to(mean.device)
            query_mask = (
                batch.query_mask.float().to(mean.device)
                if batch.query_mask is not None
                else None
            )
            huber = F.smooth_l1_loss(mean, labels, beta=0.5, reduction="none")
            nll = 0.5 * (
                log_var + (labels - mean).pow(2) * torch.exp(-log_var)
            )
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
            out["huber_loss"] = (
                weighted_masked_mean(huber, loss_weights, query_mask) + rank_term
            )
            out["nll_loss"] = (
                weighted_masked_mean(nll, loss_weights, query_mask) + rank_term
            )
        return out

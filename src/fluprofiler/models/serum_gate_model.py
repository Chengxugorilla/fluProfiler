"""
SerumGate model for serum-level zero-shot antigenic prediction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pooling import attention_mask


@dataclass
class SerumGateConfig:
    hidden_size: int
    latent_dim: int = 128
    theta_dim: int = 128
    passage_vocab_size: int = 6
    passage_pair_vocab_size: int = 36
    subtype_vocab_size: int = 1
    passage_dim: int = 8
    subtype_dim: int = 8
    predictor_arch: str = "conditioned_mlp"
    predictor_hidden_dim: int = 256
    distance_hidden_dim: int = 64
    calibration_hidden_dim: int = 64
    residual_hidden_dim: int = 32
    residual_scale: float = 0.25
    min_log_var: float = -6.0
    max_log_var: float = 4.0
    label_weight_thresholds: tuple[float, ...] = (2.0, 4.0, 6.0)
    label_weight_values: tuple[float, ...] = (1.0, 1.3, 1.8, 2.5)
    ha_pooling: str = "attention"
    ha_pair_mode: str = "independent"
    ha_attention_dim: int = 128
    ha_attention_heads: int = 4
    ha_attention_dropout: float = 0.1
    ha_mean_gate_init: float = 0.25
    na_branch: str = "none"
    na_pooling: str = "mean"
    na_latent_dim: int = 32
    na_hidden_dim: int = 32
    na_effect_init: float = 0.1
    rank_loss_weight: float = 0.0
    rank_loss_margin: float = 0.1
    rank_loss_min_label_delta: float = 0.0


@dataclass
class SerumGateBatch:
    reference_ha: torch.Tensor
    query_ha: torch.Tensor
    reference_ha_mask: Optional[torch.Tensor] = None
    query_ha_mask: Optional[torch.Tensor] = None
    reference_na: Optional[torch.Tensor] = None
    query_na: Optional[torch.Tensor] = None
    reference_na_mask: Optional[torch.Tensor] = None
    query_na_mask: Optional[torch.Tensor] = None
    serum_passage: Optional[torch.Tensor] = None
    query_passage: Optional[torch.Tensor] = None
    passage_pair: Optional[torch.Tensor] = None
    subtype: Optional[torch.Tensor] = None
    s_nagly: Optional[torch.Tensor] = None
    labels: Optional[torch.Tensor] = None
    query_mask: Optional[torch.Tensor] = None


def masked_mean(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return values.mean()
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def label_bin_weights(
    labels: torch.Tensor,
    thresholds: tuple[float, ...] | list[float],
    values: tuple[float, ...] | list[float],
) -> torch.Tensor:
    if len(values) != len(thresholds) + 1:
        raise ValueError("label_weight_values must have exactly one more entry than label_weight_thresholds")
    weights = torch.full_like(labels, float(values[0]), dtype=torch.float32)
    for threshold, value in zip(thresholds, values[1:]):
        weights = torch.where(labels >= float(threshold), torch.full_like(weights, float(value)), weights)
    return weights


def weighted_masked_mean(
    values: torch.Tensor,
    weights: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    combined = weights.to(values.dtype)
    if mask is not None:
        combined = combined * mask.to(values.dtype)
    return (values * combined).sum() / combined.sum().clamp_min(1.0)


def pairwise_ranking_loss(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor | None = None,
    margin: float = 0.0,
    min_label_delta: float = 0.0,
) -> torch.Tensor:
    if predictions.shape != labels.shape:
        raise ValueError("predictions and labels must have the same shape")
    label_delta = labels.unsqueeze(-1) - labels.unsqueeze(-2)
    pred_delta = predictions.unsqueeze(-1) - predictions.unsqueeze(-2)
    valid_pairs = torch.triu(
        torch.ones(label_delta.shape[-2:], dtype=torch.bool, device=labels.device),
        diagonal=1,
    )
    while valid_pairs.ndim < label_delta.ndim:
        valid_pairs = valid_pairs.unsqueeze(0)
    valid_pairs = valid_pairs & (label_delta.abs() >= float(min_label_delta)) & (label_delta != 0)
    if mask is not None:
        pair_mask = (mask.unsqueeze(-1) > 0) & (mask.unsqueeze(-2) > 0)
        valid_pairs = valid_pairs & pair_mask
    if not bool(valid_pairs.any()):
        return predictions.sum() * 0.0
    signed_margin = torch.sign(label_delta) * pred_delta
    violations = F.relu(float(margin) - signed_margin)
    return violations[valid_pairs].mean()


class HAPoolEncoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        latent_dim: int,
        pooling: str = "attention",
        attention_dim: int = 128,
        attention_heads: int = 4,
        attention_dropout: float = 0.1,
        mean_gate_init: float = 0.25,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=0.1)
        self.pooling = pooling
        if pooling not in {"attention", "mean", "lowrank_attention"}:
            raise ValueError(f"Unsupported HA pooling mode: {pooling}")
        self.attention_pooler = attention_mask(embed_size=hidden_size) if pooling == "attention" else None
        self.lowrank_pooler = (
            LowRankSelfAttentionPool(
                hidden_size=hidden_size,
                attention_dim=attention_dim,
                attention_heads=attention_heads,
                dropout=attention_dropout,
                mean_gate_init=mean_gate_init,
            )
            if pooling == "lowrank_attention"
            else None
        )
        self.projection = nn.Linear(hidden_size, latent_dim)

    @staticmethod
    def _pool(matrix: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return matrix.mean(dim=1)
        weights = mask.to(matrix.dtype).unsqueeze(-1)
        return (matrix * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def forward(self, matrix: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        matrix = self.dropout(matrix)
        if self.attention_pooler is not None:
            if mask is None:
                mask = torch.ones(matrix.shape[:2], dtype=matrix.dtype, device=matrix.device)
            pooled = self.attention_pooler(matrix, mask=mask, save_attention_path=None)
        elif self.lowrank_pooler is not None:
            pooled = self.lowrank_pooler(matrix, mask)
        else:
            pooled = self._pool(matrix, mask)
        return self.projection(pooled)


class LowRankSelfAttentionPool(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        attention_dim: int = 128,
        attention_heads: int = 4,
        dropout: float = 0.1,
        mean_gate_init: float = 0.25,
    ):
        super().__init__()
        if attention_dim <= 0:
            raise ValueError("ha_attention_dim must be positive")
        if attention_heads <= 0:
            raise ValueError("ha_attention_heads must be positive")
        if attention_dim % attention_heads != 0:
            raise ValueError("ha_attention_dim must be divisible by ha_attention_heads")
        if not (0.0 < mean_gate_init < 1.0):
            raise ValueError("ha_mean_gate_init must be in (0, 1)")
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
        gate_logit = math.log(mean_gate_init / (1.0 - mean_gate_init))
        self.attention_gate_logit = nn.Parameter(torch.tensor(gate_logit, dtype=torch.float32))

    @staticmethod
    def _masked_mean(matrix: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return matrix.mean(dim=1)
        weights = mask.to(matrix.dtype).unsqueeze(-1)
        return (matrix * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def forward(self, matrix: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        mean_pooled = self._masked_mean(matrix, mask)
        context = self.input_projection(self.norm(matrix))
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = mask <= 0
        context, _ = self.context_attention(
            context,
            context,
            context,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        logits = self.score(torch.tanh(self.dropout(context))).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(mask <= 0, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=1).unsqueeze(-1)
        attention_pooled = (matrix * weights).sum(dim=1)
        gate = torch.sigmoid(self.attention_gate_logit).to(matrix.dtype)
        return gate * attention_pooled + (1.0 - gate) * mean_pooled


class SerumGatePriorEncoder(nn.Module):
    def __init__(self, config: SerumGateConfig):
        super().__init__()
        input_dim = config.latent_dim + config.passage_dim + config.subtype_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, config.theta_dim),
            nn.ReLU(),
            nn.Linear(config.theta_dim, config.theta_dim),
        )

    def forward(
        self,
        reference_z: torch.Tensor,
        serum_passage_emb: torch.Tensor,
        subtype_emb: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(torch.cat([reference_z, serum_passage_emb, subtype_emb], dim=-1))


class SerumGateConditionedPredictor(nn.Module):
    def __init__(self, config: SerumGateConfig):
        super().__init__()
        if config.ha_pair_mode == "independent":
            ha_feature_dim = config.latent_dim * 4
        elif config.ha_pair_mode == "delta":
            ha_feature_dim = config.latent_dim
        else:
            raise ValueError(f"Unsupported HA pair mode: {config.ha_pair_mode}")
        pair_dim = (
            ha_feature_dim
            + config.passage_dim
            + config.passage_dim
            + config.subtype_dim
            + 1
        )
        self.input_layer = nn.Linear(pair_dim, config.predictor_hidden_dim)
        self.film = nn.Linear(config.theta_dim, config.predictor_hidden_dim * 2)
        self.output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(config.predictor_hidden_dim, config.predictor_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.predictor_hidden_dim, 2),
        )

    def forward(
        self,
        pair_features: torch.Tensor,
        theta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.input_layer(pair_features)
        gamma, beta = self.film(theta).chunk(2, dim=-1)
        while gamma.ndim < hidden.ndim:
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        conditioned = hidden * (1.0 + gamma) + beta
        out = self.output_layer(conditioned)
        return out[..., 0], out[..., 1]


class CalibratedMetricPredictor(nn.Module):
    def __init__(self, config: SerumGateConfig):
        super().__init__()
        if config.distance_hidden_dim <= 0:
            raise ValueError("distance_hidden_dim must be positive")
        if config.calibration_hidden_dim <= 0:
            raise ValueError("calibration_hidden_dim must be positive")
        if config.residual_hidden_dim <= 0:
            raise ValueError("residual_hidden_dim must be positive")
        self.residual_scale = float(config.residual_scale)
        self.distance_head = nn.Sequential(
            nn.Linear(config.latent_dim, config.distance_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.distance_hidden_dim, 1),
        )
        self.calibration_head = nn.Sequential(
            nn.Linear(config.theta_dim, config.calibration_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.calibration_hidden_dim, 3),
        )
        residual_input_dim = config.latent_dim + config.passage_dim + config.subtype_dim + 1
        self.residual_head = nn.Sequential(
            nn.Linear(residual_input_dim, config.residual_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.residual_hidden_dim, 1),
        )
        self._initialize_calibration()

    def _initialize_calibration(self) -> None:
        final = self.calibration_head[-1]
        if not isinstance(final, nn.Linear):
            return
        scale_one_logit = math.log(math.expm1(1.0))
        with torch.no_grad():
            final.weight.zero_()
            final.bias.copy_(torch.tensor([0.0, scale_one_logit, 0.0], dtype=final.bias.dtype))

    def forward(
        self,
        abs_delta: torch.Tensor,
        context_features: torch.Tensor,
        theta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        distance = F.softplus(self.distance_head(abs_delta).squeeze(-1))
        calibration = self.calibration_head(theta)
        bias, raw_scale, raw_log_var = calibration.unbind(dim=-1)
        scale = F.softplus(raw_scale) + 1e-6
        residual_features = torch.cat([abs_delta, context_features], dim=-1)
        residual = self.residual_scale * torch.tanh(self.residual_head(residual_features).squeeze(-1))
        mean = bias[:, None] + scale[:, None] * distance + residual
        log_var = raw_log_var[:, None].expand_as(mean)
        parts = {
            "distance": distance,
            "residual": residual,
            "serum_bias": bias,
            "serum_scale": scale,
        }
        return mean, log_var, parts


class NAPairEffect(nn.Module):
    def __init__(self, config: SerumGateConfig):
        super().__init__()
        if config.na_hidden_dim <= 0:
            raise ValueError("na_hidden_dim must be positive")
        if not (0.0 < config.na_effect_init < 1.0):
            raise ValueError("na_effect_init must be in (0, 1)")
        self.net = nn.Sequential(
            nn.Linear(config.na_latent_dim * 4, config.na_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.na_hidden_dim, 1),
        )
        gate_logit = math.log(config.na_effect_init / (1.0 - config.na_effect_init))
        self.gate_logit = nn.Parameter(torch.tensor(gate_logit, dtype=torch.float32))

    def forward(self, reference_na_z: torch.Tensor, query_na_z: torch.Tensor) -> torch.Tensor:
        reference_expanded = reference_na_z[:, None, :].expand(-1, query_na_z.shape[1], -1)
        features = torch.cat(
            [
                query_na_z,
                reference_expanded,
                query_na_z - reference_expanded,
                torch.abs(query_na_z - reference_expanded),
            ],
            dim=-1,
        )
        raw_effect = self.net(features).squeeze(-1)
        return torch.sigmoid(self.gate_logit).to(raw_effect.dtype) * raw_effect


class SerumGateModel(nn.Module):
    def __init__(self, config: SerumGateConfig | Mapping[str, Any]):
        super().__init__()
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
        self.config = config
        self.ha_encoder = HAPoolEncoder(
            config.hidden_size,
            config.latent_dim,
            pooling=config.ha_pooling,
            attention_dim=config.ha_attention_dim,
            attention_heads=config.ha_attention_heads,
            attention_dropout=config.ha_attention_dropout,
            mean_gate_init=config.ha_mean_gate_init,
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
        self.passage_pair_embedding = nn.Embedding(config.passage_pair_vocab_size, config.passage_dim)
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

    @staticmethod
    def _as_batch(batch: SerumGateBatch | Mapping[str, Any]) -> SerumGateBatch:
        if isinstance(batch, SerumGateBatch):
            return batch
        if isinstance(batch, Mapping):
            allowed = {field.name for field in fields(SerumGateBatch)}
            return SerumGateBatch(**{key: value for key, value in batch.items() if key in allowed})
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    @staticmethod
    def _default_ids(reference: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.zeros(shape, dtype=torch.long, device=reference.device)

    def _encode_query_ha(
        self,
        query_ha: torch.Tensor,
        query_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, query_count, token_count, hidden_size = query_ha.shape
        flat = query_ha.reshape(batch_size * query_count, token_count, hidden_size)
        flat_mask = None
        if query_mask is not None:
            flat_mask = query_mask.reshape(batch_size * query_count, token_count)
        encoded = self.ha_encoder(flat, flat_mask)
        return encoded.reshape(batch_size, query_count, -1)

    @staticmethod
    def _encode_query_with_encoder(
        encoder: HAPoolEncoder,
        query_matrix: torch.Tensor,
        query_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, query_count, token_count, hidden_size = query_matrix.shape
        flat = query_matrix.reshape(batch_size * query_count, token_count, hidden_size)
        flat_mask = None
        if query_mask is not None:
            flat_mask = query_mask.reshape(batch_size * query_count, token_count)
        encoded = encoder(flat, flat_mask)
        return encoded.reshape(batch_size, query_count, -1)

    def _encode_delta_ha(
        self,
        reference_ha: torch.Tensor,
        query_ha: torch.Tensor,
        reference_mask: torch.Tensor | None,
        query_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, query_count, token_count, hidden_size = query_ha.shape
        reference_expanded = reference_ha[:, None, :, :].expand(-1, query_count, -1, -1)
        flat_delta = (query_ha - reference_expanded).reshape(batch_size * query_count, token_count, hidden_size)
        flat_mask = None
        if query_mask is not None:
            combined_mask = query_mask
            if reference_mask is not None:
                combined_mask = combined_mask * reference_mask[:, None, :]
            flat_mask = combined_mask.reshape(batch_size * query_count, token_count)
        elif reference_mask is not None:
            flat_mask = reference_mask[:, None, :].expand(-1, query_count, -1).reshape(batch_size * query_count, token_count)
        encoded = self.ha_encoder(flat_delta, flat_mask)
        return encoded.reshape(batch_size, query_count, -1)

    def forward(self, batch: SerumGateBatch | Mapping[str, Any]) -> dict[str, torch.Tensor]:
        batch = self._as_batch(batch)
        device = batch.reference_ha.device
        batch_size, query_count = batch.query_ha.shape[:2]

        serum_passage = batch.serum_passage
        if serum_passage is None:
            serum_passage = self._default_ids(batch.reference_ha, (batch_size,))
        query_passage = batch.query_passage
        if query_passage is None:
            query_passage = self._default_ids(batch.reference_ha, (batch_size, query_count))
        passage_pair = batch.passage_pair
        if passage_pair is None:
            passage_pair = serum_passage[:, None] * int(self.config.passage_vocab_size) + query_passage
        subtype = batch.subtype
        if subtype is None:
            subtype = self._default_ids(batch.reference_ha, (batch_size,))
        s_nagly = batch.s_nagly
        if s_nagly is None:
            s_nagly = torch.zeros(batch_size, query_count, dtype=batch.reference_ha.dtype, device=device)

        reference_z = self.ha_encoder(batch.reference_ha, batch.reference_ha_mask)
        query_z = self._encode_query_ha(batch.query_ha, batch.query_ha_mask)
        reference_na_z = None
        query_na_z = None
        if self.config.na_branch != "none":
            if self.na_encoder is None or self.na_pair_effect is None:
                raise RuntimeError("NA branch is not initialized")
            if batch.reference_na is None or batch.query_na is None:
                raise ValueError("na_branch requires reference_na and query_na tensors")
            reference_na_z = self.na_encoder(batch.reference_na, batch.reference_na_mask)
            query_na_z = self._encode_query_with_encoder(self.na_encoder, batch.query_na, batch.query_na_mask)

        serum_passage_emb = self.passage_embedding(serum_passage.long().to(device))
        if self.subtype_embedding is None:
            subtype_emb = reference_z.new_empty((batch_size, 0))
        else:
            subtype_emb = self.subtype_embedding(subtype.long().to(device))
        theta0 = self.prior_encoder(reference_z, serum_passage_emb, subtype_emb)

        reference_z_expanded = reference_z[:, None, :].expand(-1, query_count, -1)
        subtype_expanded = subtype_emb[:, None, :].expand(-1, query_count, -1)
        query_passage_emb = self.passage_embedding(query_passage.long().to(device))
        passage_pair_emb = self.passage_pair_embedding(passage_pair.long().to(device))
        glycan_feature = s_nagly.to(device).unsqueeze(-1).float()
        predictor_parts: dict[str, torch.Tensor] = {}
        if self.config.predictor_arch == "conditioned_mlp":
            if self.config.ha_pair_mode == "independent":
                ha_pair_features = [
                    query_z,
                    reference_z_expanded,
                    query_z - reference_z_expanded,
                    torch.abs(query_z - reference_z_expanded),
                ]
            elif self.config.ha_pair_mode == "delta":
                ha_pair_features = [
                    self._encode_delta_ha(
                        batch.reference_ha,
                        batch.query_ha,
                        batch.reference_ha_mask,
                        batch.query_ha_mask,
                    )
                ]
            else:
                raise ValueError(f"Unsupported HA pair mode: {self.config.ha_pair_mode}")
            pair_features = torch.cat(
                [
                    *ha_pair_features,
                    query_passage_emb,
                    passage_pair_emb,
                    subtype_expanded,
                    glycan_feature,
                ],
                dim=-1,
            )
            mean, raw_log_var = self.predictor(pair_features, theta0)
        else:
            abs_delta = torch.abs(query_z - reference_z_expanded)
            context_features = torch.cat(
                [
                    passage_pair_emb,
                    subtype_expanded,
                    glycan_feature,
                ],
                dim=-1,
            )
            mean, raw_log_var, predictor_parts = self.predictor(abs_delta, context_features, theta0)
        if self.config.na_branch == "pair":
            assert reference_na_z is not None and query_na_z is not None and self.na_pair_effect is not None
            na_effect = self.na_pair_effect(reference_na_z, query_na_z)
            mean = mean + na_effect
            predictor_parts.update(
                {
                    "na_effect": na_effect,
                    "z_na_ref": reference_na_z,
                    "z_na_query": query_na_z,
                }
            )
        log_var = raw_log_var.clamp(self.config.min_log_var, self.config.max_log_var)

        out: dict[str, torch.Tensor] = {
            "mean": mean,
            "log_var": log_var,
            "theta0": theta0,
            "z_ref": reference_z,
            "z_query": query_z,
        }
        out.update(predictor_parts)
        if batch.labels is not None:
            labels = batch.labels.float().to(device)
            query_mask = batch.query_mask.float().to(device) if batch.query_mask is not None else None
            huber = F.smooth_l1_loss(mean, labels, beta=0.5, reduction="none")
            nll = 0.5 * (log_var + (labels - mean).pow(2) * torch.exp(-log_var))
            loss_weights = label_bin_weights(
                labels,
                self.config.label_weight_thresholds,
                self.config.label_weight_values,
            ).to(device)
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

"""
Explicit serum/reference-background plus directed mutation-set model.

This module is intentionally independent from the existing SerumGate model
and batch types. It contains no NA branch and does not require NA sequence IDs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# This model uses the training pipeline amino-acid encoding: PAD=0, 20
# canonical residues=1..20, X/unknown=21, and gap=22.
UNKNOWN_AMINO_ACID_ID = 21


@dataclass
class SerumMutationSetConfig:
    hidden_size: int
    site_dim: int = 64
    site_bottleneck_dim: int = 0
    background_dim: int = 128
    mutation_dim: int = 128
    position_dim: int = 32
    amino_acid_dim: int = 16
    presence_dim: int = 4
    max_position_embeddings: int = 400
    amino_acid_vocab_size: int = 23
    passage_vocab_size: int = 6
    use_passage_pair_feature: bool = True
    passage_pair_vocab_size: int = 36
    passage_dim: int = 8
    subtype_vocab_size: int = 1
    subtype_dim: int = 8
    theta_dim: int = 128
    mutation_attention_heads: int = 4
    mutation_attention_layers: int = 1
    mutation_ffn_dim: int = 256
    attention_dropout: float = 0.1
    attention_alpha_init: float = 0.05
    attention_tau_init: float = 8.0
    predictor_hidden_dim: int = 256
    predictor_dropout: float = 0.1
    label_weight_thresholds: tuple[float, ...] = (2.0, 4.0, 6.0)
    label_weight_values: tuple[float, ...] = (1.0, 1.3, 1.8, 2.5)
    rank_loss_weight: float = 0.0
    rank_loss_margin: float = 0.1
    rank_loss_min_label_delta: float = 0.0
    zero_init_film: bool = False
    use_film_beta: bool = True
    use_pool_mutation_count: bool = True
    use_attention_pool: bool = True
    use_predictor_mutation_count: bool = True
    use_background_to_mutation: bool = True
    bypass_mutation_transformer: bool = False
    task_bias_loss_weight: float = 0.0
    direct_background: bool = False
    use_output_identity_bias: bool = False
    serum_name_vocab_size: int = 1
    query_virus_vocab_size: int = 1


@dataclass
class SerumMutationSetBatch:
    reference_ha: torch.Tensor
    query_ha: torch.Tensor
    reference_aa: torch.Tensor
    query_aa: torch.Tensor
    reference_aligned_mask: torch.Tensor
    query_aligned_mask: torch.Tensor
    reference_embedding_mask: torch.Tensor
    query_embedding_mask: torch.Tensor
    serum_passage: Optional[torch.Tensor] = None
    query_passage: Optional[torch.Tensor] = None
    passage_pair: Optional[torch.Tensor] = None
    subtype: Optional[torch.Tensor] = None
    labels: Optional[torch.Tensor] = None
    query_mask: Optional[torch.Tensor] = None
    serum_name: Optional[torch.Tensor] = None
    query_virus: Optional[torch.Tensor] = None


def _inverse_softplus(value: float) -> float:
    if value <= 0:
        raise ValueError("softplus-constrained initial values must be positive")
    return math.log(math.expm1(value))


def _masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(tokens.dtype).unsqueeze(-1)
    return (tokens * weights).sum(dim=-2) / weights.sum(dim=-2).clamp_min(1.0)


def _weighted_masked_mean(
    values: torch.Tensor,
    weights: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    combined = weights.to(values.dtype)
    if mask is not None:
        combined = combined * mask.to(values.dtype)
    return (values * combined).sum() / combined.sum().clamp_min(1.0)


def _label_bin_weights(
    labels: torch.Tensor,
    thresholds: tuple[float, ...],
    values: tuple[float, ...],
) -> torch.Tensor:
    if len(values) != len(thresholds) + 1:
        raise ValueError("label_weight_values must have one more entry than thresholds")
    weights = torch.full_like(labels, float(values[0]), dtype=torch.float32)
    for threshold, value in zip(thresholds, values[1:]):
        weights = torch.where(labels >= float(threshold), float(value), weights)
    return weights


def _pairwise_ranking_loss(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor | None,
    margin: float,
    min_label_delta: float,
) -> torch.Tensor:
    label_delta = labels.unsqueeze(-1) - labels.unsqueeze(-2)
    prediction_delta = predictions.unsqueeze(-1) - predictions.unsqueeze(-2)
    upper = torch.triu(
        torch.ones(label_delta.shape[-2:], dtype=torch.bool, device=labels.device),
        diagonal=1,
    )
    while upper.ndim < label_delta.ndim:
        upper = upper.unsqueeze(0)
    valid = upper & (label_delta != 0) & (label_delta.abs() >= float(min_label_delta))
    if mask is not None:
        valid = valid & ((mask.unsqueeze(-1) > 0) & (mask.unsqueeze(-2) > 0))
    if not bool(valid.any()):
        return predictions.sum() * 0.0
    signed_prediction_delta = torch.sign(label_delta) * prediction_delta
    return F.relu(float(margin) - signed_prediction_delta)[valid].mean()


class DistanceBiasedSelfAttention(nn.Module):
    """Multi-head self-attention with a batch-specific RBF distance bias."""

    def __init__(
        self,
        dim: int,
        heads: int,
        dropout: float,
        alpha_init: float,
        tau_init: float,
    ):
        super().__init__()
        if dim <= 0 or heads <= 0 or dim % heads != 0:
            raise ValueError("attention dim must be positive and divisible by heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("attention dropout must be in [0, 1)")
        self.dim = int(dim)
        self.heads = int(heads)
        self.head_dim = self.dim // self.heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(self.dim, self.dim * 3)
        self.output = nn.Linear(self.dim, self.dim)
        self.dropout = nn.Dropout(dropout)
        self.raw_alpha = nn.Parameter(
            torch.full((self.heads,), _inverse_softplus(float(alpha_init)))
        )
        self.raw_tau = nn.Parameter(
            torch.full((self.heads,), _inverse_softplus(float(tau_init)))
        )

    def distance_bias(self, distances: torch.Tensor) -> torch.Tensor:
        if distances.ndim != 3 or distances.shape[-1] != distances.shape[-2]:
            raise ValueError("distances must have shape [N, K, K]")
        finite = torch.isfinite(distances)
        size = distances.shape[-1]
        diagonal = torch.eye(size, dtype=torch.bool, device=distances.device).unsqueeze(0)
        known_off_diagonal = finite & ~diagonal
        safe_distance = torch.where(finite, distances.clamp_min(0), torch.zeros_like(distances))
        alpha = F.softplus(self.raw_alpha).to(distances.dtype).view(1, self.heads, 1, 1)
        tau = (
            F.softplus(self.raw_tau).to(distances.dtype).view(1, self.heads, 1, 1)
            + torch.finfo(distances.dtype).eps
        )
        rbf = alpha * torch.exp(-safe_distance.unsqueeze(1) / tau)
        return torch.where(known_off_diagonal.unsqueeze(1), rbf, torch.zeros_like(rbf))

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        distances: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim != 3:
            raise ValueError("tokens must have shape [N, K, D]")
        n_items, token_count, _ = tokens.shape
        qkv = self.qkv(tokens).reshape(
            n_items, token_count, 3, self.heads, self.head_dim
        )
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        logits = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        logits = logits + self.distance_bias(distances)

        valid = mask > 0
        logits = logits.masked_fill(
            ~valid[:, None, None, :],
            torch.finfo(logits.dtype).min,
        )
        weights = torch.softmax(logits, dim=-1)
        weights = weights * valid[:, None, :, None].to(weights.dtype)
        weights = self.dropout(weights)
        context = torch.matmul(weights, value)
        context = context.transpose(1, 2).reshape(n_items, token_count, self.dim)
        context = self.output(context)
        context = context * valid.unsqueeze(-1).to(context.dtype)
        return context, weights


class MutationTransformerBlock(nn.Module):
    def __init__(self, config: SerumMutationSetConfig):
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.mutation_dim)
        self.attention = DistanceBiasedSelfAttention(
            dim=config.mutation_dim,
            heads=config.mutation_attention_heads,
            dropout=config.attention_dropout,
            alpha_init=config.attention_alpha_init,
            tau_init=config.attention_tau_init,
        )
        self.ffn_norm = nn.LayerNorm(config.mutation_dim)
        self.ffn = nn.Sequential(
            nn.Linear(config.mutation_dim, config.mutation_ffn_dim),
            nn.GELU(),
            nn.Dropout(config.attention_dropout),
            nn.Linear(config.mutation_ffn_dim, config.mutation_dim),
            nn.Dropout(config.attention_dropout),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        distances: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attention_out, weights = self.attention(
            self.attention_norm(tokens),
            mask,
            distances,
        )
        tokens = tokens + attention_out
        tokens = tokens + self.ffn(self.ffn_norm(tokens)) * mask.unsqueeze(-1).to(tokens.dtype)
        return tokens, weights


class MutationSetPool(nn.Module):
    def __init__(
        self,
        dim: int,
        use_mutation_count: bool = True,
        use_attention_pool: bool = True,
    ):
        super().__init__()
        self.use_mutation_count = bool(use_mutation_count)
        self.use_attention_pool = bool(use_attention_pool)
        self.score = nn.Linear(dim, 1) if self.use_attention_pool else None
        self.projection = nn.Sequential(
            nn.Linear(
                dim * (1 + int(self.use_attention_pool))
                + int(self.use_mutation_count),
                dim,
            ),
            nn.GELU(),
            nn.LayerNorm(dim),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        mutation_count: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid = mask > 0
        mean_pooled = _masked_mean(tokens, mask)
        features = [mean_pooled]
        if self.score is None:
            weights = tokens.new_zeros(mask.shape)
        else:
            logits = self.score(torch.tanh(tokens)).squeeze(-1)
            logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
            weights = torch.softmax(logits, dim=-1)
            weights = weights * valid.to(weights.dtype)
            attention_pooled = (tokens * weights.unsqueeze(-1)).sum(dim=-2)
            features.append(attention_pooled)
        if self.use_mutation_count:
            features.append(torch.log1p(mutation_count.to(tokens.dtype)).unsqueeze(-1))
        pooled = self.projection(torch.cat(features, dim=-1))
        return pooled, weights


class SerumMutationSetMinusModel(nn.Module):
    def __init__(
        self,
        config: SerumMutationSetConfig | Mapping[str, Any],
        ha_distance_matrix: torch.Tensor,
    ):
        super().__init__()
        if isinstance(config, Mapping):
            config = SerumMutationSetConfig(**dict(config))
        self._validate_config(config)
        distance = torch.as_tensor(ha_distance_matrix, dtype=torch.float32)
        self._validate_distance_matrix(distance, config.max_position_embeddings)
        self.config = config
        self.register_buffer("ha_distance_matrix", distance.clone())

        site_projection_layers: list[nn.Module] = [nn.LayerNorm(config.hidden_size)]
        if config.site_bottleneck_dim > 0:
            site_projection_layers.extend(
                [
                    nn.Linear(config.hidden_size, config.site_bottleneck_dim),
                    nn.Linear(config.site_bottleneck_dim, config.site_dim),
                ]
            )
        else:
            site_projection_layers.append(nn.Linear(config.hidden_size, config.site_dim))
        site_projection_layers.append(nn.GELU())
        self.site_projection = nn.Sequential(*site_projection_layers)
        self.background_encoder = (
            nn.Identity()
            if config.direct_background
            else nn.Sequential(
                nn.LayerNorm(config.site_dim),
                nn.Linear(config.site_dim, config.background_dim),
                nn.GELU(),
                nn.Linear(config.background_dim, config.background_dim),
            )
        )
        self.position_embedding = nn.Embedding(
            config.max_position_embeddings,
            config.position_dim,
        )
        self.amino_acid_embedding = nn.Embedding(
            config.amino_acid_vocab_size,
            config.amino_acid_dim,
        )
        self.presence_embedding = nn.Embedding(2, config.presence_dim)
        mutation_input_dim = (
            config.site_dim * 3
            + config.position_dim
            + config.amino_acid_dim * 2
            + config.presence_dim * 2
        )
        self.mutation_token_projection = nn.Sequential(
            nn.Linear(mutation_input_dim, config.mutation_dim),
            nn.GELU(),
            nn.LayerNorm(config.mutation_dim),
        )
        self.background_to_mutation = nn.Linear(
            config.background_dim,
            config.mutation_dim,
        )
        self.null_mutation = nn.Parameter(torch.zeros(config.mutation_dim))
        nn.init.normal_(self.null_mutation, mean=0.0, std=0.02)
        self.mutation_blocks = nn.ModuleList(
            MutationTransformerBlock(config)
            for _ in range(config.mutation_attention_layers)
        )
        self.mutation_pool = MutationSetPool(
            config.mutation_dim,
            use_mutation_count=config.use_pool_mutation_count,
            use_attention_pool=config.use_attention_pool,
        )

        self.passage_embedding = nn.Embedding(
            config.passage_vocab_size,
            config.passage_dim,
        )
        self.passage_pair_embedding = (
            nn.Embedding(
                config.passage_pair_vocab_size,
                config.passage_dim,
            ) if config.use_passage_pair_feature else None
        )
        self.subtype_embedding = (
            nn.Embedding(config.subtype_vocab_size, config.subtype_dim)
            if config.subtype_dim > 0
            else None
        )
        condition_input_dim = (
            config.background_dim + config.passage_dim + config.subtype_dim
        )
        self.serum_condition_encoder = nn.Sequential(
            nn.Linear(condition_input_dim, config.theta_dim),
            nn.ReLU(),
            nn.Linear(config.theta_dim, config.theta_dim),
        )
        self.film = nn.Linear(config.theta_dim, config.mutation_dim * 2)
        if config.zero_init_film:
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
        predictor_input_dim = (
            config.mutation_dim
            + config.passage_dim
            + (config.passage_dim if config.use_passage_pair_feature else 0)
            + config.subtype_dim
            + 1
        )
        self.predictor = nn.Sequential(
            nn.Linear(predictor_input_dim, config.predictor_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.predictor_dropout),
            nn.Linear(config.predictor_hidden_dim, config.predictor_hidden_dim),
            nn.GELU(),
            nn.Linear(config.predictor_hidden_dim, 1),
        )
        self.serum_name_bias = (
            nn.Embedding(config.serum_name_vocab_size, 1, padding_idx=0)
            if config.use_output_identity_bias
            else None
        )
        self.query_virus_bias = (
            nn.Embedding(config.query_virus_vocab_size, 1, padding_idx=0)
            if config.use_output_identity_bias
            else None
        )
        if config.use_output_identity_bias:
            # Index 0 is the fixed zero/unknown bucket.  Starting all known
            # identity offsets at zero makes this branch an additive-only
            # extension of the name-free model.
            nn.init.zeros_(self.serum_name_bias.weight)
            nn.init.zeros_(self.query_virus_bias.weight)

    @staticmethod
    def _validate_config(config: SerumMutationSetConfig) -> None:
        positive = {
            "hidden_size": config.hidden_size,
            "site_dim": config.site_dim,
            "background_dim": config.background_dim,
            "mutation_dim": config.mutation_dim,
            "position_dim": config.position_dim,
            "amino_acid_dim": config.amino_acid_dim,
            "presence_dim": config.presence_dim,
            "max_position_embeddings": config.max_position_embeddings,
            "amino_acid_vocab_size": config.amino_acid_vocab_size,
            "passage_vocab_size": config.passage_vocab_size,
            "passage_pair_vocab_size": config.passage_pair_vocab_size,
            "theta_dim": config.theta_dim,
            "mutation_attention_heads": config.mutation_attention_heads,
            "mutation_attention_layers": config.mutation_attention_layers,
            "mutation_ffn_dim": config.mutation_ffn_dim,
            "predictor_hidden_dim": config.predictor_hidden_dim,
            "serum_name_vocab_size": config.serum_name_vocab_size,
            "query_virus_vocab_size": config.query_virus_vocab_size,
        }
        invalid = [name for name, value in positive.items() if int(value) <= 0]
        if invalid:
            raise ValueError(f"configuration fields must be positive: {', '.join(invalid)}")
        if config.mutation_dim % config.mutation_attention_heads != 0:
            raise ValueError("mutation_dim must be divisible by mutation_attention_heads")
        if len(config.label_weight_values) != len(config.label_weight_thresholds) + 1:
            raise ValueError("label_weight_values must have one more entry than thresholds")
        if config.task_bias_loss_weight < 0:
            raise ValueError("task_bias_loss_weight must be non-negative")
        if config.site_bottleneck_dim < 0:
            raise ValueError("site_bottleneck_dim must be non-negative")
        if config.serum_name_vocab_size <= 0 or config.query_virus_vocab_size <= 0:
            raise ValueError("identity vocabulary sizes must be positive")
        if config.direct_background and config.background_dim != config.site_dim:
            raise ValueError(
                "direct_background requires background_dim to equal site_dim"
            )

    @staticmethod
    def _validate_distance_matrix(distance: torch.Tensor, required_size: int) -> None:
        if distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
            raise ValueError("HA distance matrix must be square")
        if distance.shape[0] < required_size:
            raise ValueError(
                "HA distance matrix is smaller than max_position_embeddings: "
                f"{distance.shape[0]} < {required_size}"
            )
        finite = torch.isfinite(distance)
        if bool((distance[finite] < 0).any()):
            raise ValueError("finite HA distances must be non-negative")

    @staticmethod
    def _as_batch(
        batch: SerumMutationSetBatch | Mapping[str, Any],
    ) -> SerumMutationSetBatch:
        if isinstance(batch, SerumMutationSetBatch):
            return batch
        if isinstance(batch, Mapping):
            allowed = {field.name for field in fields(SerumMutationSetBatch)}
            return SerumMutationSetBatch(
                **{key: value for key, value in batch.items() if key in allowed}
            )
        raise TypeError(f"unsupported batch type: {type(batch)}")

    @staticmethod
    def _default_ids(reference: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.zeros(shape, dtype=torch.long, device=reference.device)

    def _validate_batch(self, batch: SerumMutationSetBatch) -> None:
        if batch.reference_ha.ndim != 3 or batch.query_ha.ndim != 4:
            raise ValueError("reference_ha/query_ha must have shapes [B,L,H]/[B,Q,L,H]")
        batch_size, token_count, hidden_size = batch.reference_ha.shape
        if hidden_size != self.config.hidden_size:
            raise ValueError(
                f"expected hidden_size={self.config.hidden_size}, got {hidden_size}"
            )
        if batch.query_ha.shape[0] != batch_size:
            raise ValueError("reference and query batch sizes differ")
        if batch.query_ha.shape[2:] != (token_count, hidden_size):
            raise ValueError("reference and query HA shapes are not aligned")
        if token_count > self.config.max_position_embeddings:
            raise ValueError("batch positions exceed max_position_embeddings")
        expected_reference = (batch_size, token_count)
        expected_query = tuple(batch.query_ha.shape[:3])
        for name in (
            "reference_aa",
            "reference_aligned_mask",
            "reference_embedding_mask",
        ):
            if tuple(getattr(batch, name).shape) != expected_reference:
                raise ValueError(f"{name} must have shape {expected_reference}")
        for name in (
            "query_aa",
            "query_aligned_mask",
            "query_embedding_mask",
        ):
            if tuple(getattr(batch, name).shape) != expected_query:
                raise ValueError(f"{name} must have shape {expected_query}")

    def _reference_background(
        self,
        batch: SerumMutationSetBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected = self.site_projection(batch.reference_ha)
        projected = projected * batch.reference_embedding_mask.unsqueeze(-1).to(
            projected.dtype
        )
        pooled = _masked_mean(projected, batch.reference_embedding_mask)
        return projected, self.background_encoder(pooled)

    def _pack_mutation_tokens(
        self,
        all_tokens: torch.Tensor,
        mutation_mask: torch.Tensor,
        background: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, query_count, token_count, mutation_dim = all_tokens.shape
        flat_tokens = all_tokens.reshape(-1, token_count, mutation_dim)
        flat_mask = mutation_mask.reshape(-1, token_count)
        flat_background = (
            background[:, None, :]
            .expand(-1, query_count, -1)
            .reshape(-1, background.shape[-1])
        )
        counts = flat_mask.sum(dim=-1)
        max_mutations = max(1, int(counts.max().item()))
        packed = all_tokens.new_zeros((flat_tokens.shape[0], max_mutations, mutation_dim))
        packed_mask = all_tokens.new_zeros((flat_tokens.shape[0], max_mutations))
        positions = torch.zeros(
            flat_tokens.shape[0],
            max_mutations,
            dtype=torch.long,
            device=all_tokens.device,
        )
        null_tokens = self.null_mutation.unsqueeze(0).expand(flat_tokens.shape[0], -1)
        if self.config.use_background_to_mutation:
            null_tokens = null_tokens + self.background_to_mutation(flat_background)
        for row in range(flat_tokens.shape[0]):
            selected = torch.nonzero(flat_mask[row], as_tuple=False).squeeze(-1)
            if selected.numel() == 0:
                packed[row, 0] = null_tokens[row]
                packed_mask[row, 0] = 1.0
            else:
                length = int(selected.numel())
                packed[row, :length] = flat_tokens[row, selected]
                packed_mask[row, :length] = 1.0
                positions[row, :length] = selected
        return packed, packed_mask, positions, counts.to(all_tokens.dtype)

    def _selected_distances(
        self,
        positions: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if int(positions.max().item()) >= self.ha_distance_matrix.shape[0]:
            raise ValueError("mutation position exceeds HA distance matrix")
        distances = self.ha_distance_matrix[positions.unsqueeze(-1), positions.unsqueeze(-2)]
        pair_valid = (mask.unsqueeze(-1) > 0) & (mask.unsqueeze(-2) > 0)
        return torch.where(
            pair_valid,
            distances,
            torch.full_like(distances, float("nan")),
        )

    def encode_query_mutations(
        self,
        batch: SerumMutationSetBatch | Mapping[str, Any],
    ) -> dict[str, torch.Tensor]:
        batch = self._as_batch(batch)
        self._validate_batch(batch)
        reference_projected, background = self._reference_background(batch)
        query_projected = self.site_projection(batch.query_ha)
        query_projected = query_projected * batch.query_embedding_mask.unsqueeze(-1).to(
            query_projected.dtype
        )
        batch_size, query_count, token_count = batch.query_aa.shape
        reference_expanded = reference_projected[:, None, :, :].expand(
            -1, query_count, -1, -1
        )
        delta = query_projected - reference_expanded
        position_ids = torch.arange(token_count, device=batch.reference_ha.device)
        position_features = self.position_embedding(position_ids)[None, None].expand(
            batch_size, query_count, -1, -1
        )
        reference_aa = batch.reference_aa[:, None, :].expand(-1, query_count, -1)
        reference_presence = (
            batch.reference_embedding_mask[:, None, :]
            .expand(-1, query_count, -1)
            .long()
            .clamp(0, 1)
        )
        query_presence = batch.query_embedding_mask.long().clamp(0, 1)
        token_features = torch.cat(
            [
                reference_expanded,
                delta,
                delta.abs(),
                position_features,
                self.amino_acid_embedding(reference_aa.long()),
                self.amino_acid_embedding(batch.query_aa.long()),
                self.presence_embedding(reference_presence),
                self.presence_embedding(query_presence),
            ],
            dim=-1,
        )
        all_tokens = self.mutation_token_projection(token_features)
        if self.config.use_background_to_mutation:
            all_tokens = all_tokens + self.background_to_mutation(background)[:, None, None, :]
        mutation_mask = (
            (batch.reference_aligned_mask[:, None, :] > 0)
            & (batch.query_aligned_mask > 0)
            & (reference_aa != batch.query_aa)
            & (reference_aa != UNKNOWN_AMINO_ACID_ID)
            & (batch.query_aa != UNKNOWN_AMINO_ACID_ID)
        )
        packed, packed_mask, positions, counts = self._pack_mutation_tokens(
            all_tokens,
            mutation_mask,
            background,
        )
        distances = self._selected_distances(positions, packed_mask)
        self_attention = packed.new_zeros(
            (
                packed.shape[0],
                self.config.mutation_attention_heads,
                packed.shape[1],
                packed.shape[1],
            )
        )
        if not self.config.bypass_mutation_transformer:
            for block in self.mutation_blocks:
                packed, self_attention = block(packed, packed_mask, distances)
        pooled, pooling_attention = self.mutation_pool(packed, packed_mask, counts)
        return {
            "z_background": background,
            "z_mutation": pooled.reshape(batch_size, query_count, -1),
            "mutation_count": counts.reshape(batch_size, query_count),
            "mutation_attention": pooling_attention.reshape(
                batch_size, query_count, -1
            ),
            "mutation_self_attention": self_attention.reshape(
                batch_size,
                query_count,
                self.config.mutation_attention_heads,
                packed.shape[1],
                packed.shape[1],
            ),
            "mutation_positions": positions.reshape(batch_size, query_count, -1),
            "mutation_token_mask": packed_mask.reshape(batch_size, query_count, -1),
        }

    def _encode_empty_mutation(
        self,
        background: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = background.shape[0]
        tokens = self.null_mutation.view(1, 1, -1).expand(batch_size, 1, -1)
        if self.config.use_background_to_mutation:
            tokens = tokens + self.background_to_mutation(background).unsqueeze(1)
        mask = tokens.new_ones((batch_size, 1))
        positions = torch.zeros(batch_size, 1, dtype=torch.long, device=tokens.device)
        distances = self._selected_distances(positions, mask)
        if not self.config.bypass_mutation_transformer:
            for block in self.mutation_blocks:
                tokens, _ = block(tokens, mask, distances)
        counts = tokens.new_zeros(batch_size)
        pooled, _ = self.mutation_pool(tokens, mask, counts)
        return pooled[:, None, :], counts[:, None]

    def _condition(
        self,
        background: torch.Tensor,
        serum_passage: torch.Tensor,
        subtype: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        serum_passage_embedding = self.passage_embedding(serum_passage.long())
        if self.subtype_embedding is None:
            subtype_embedding = background.new_empty((background.shape[0], 0))
        else:
            subtype_embedding = self.subtype_embedding(subtype.long())
        theta = self.serum_condition_encoder(
            torch.cat(
                [background, serum_passage_embedding, subtype_embedding],
                dim=-1,
            )
        )
        return theta, subtype_embedding

    def _predict_score(
        self,
        z_mutation: torch.Tensor,
        mutation_count: torch.Tensor,
        theta: torch.Tensor,
        query_passage: torch.Tensor,
        passage_pair: torch.Tensor | None,
        subtype_embedding: torch.Tensor,
    ) -> torch.Tensor:
        query_count = z_mutation.shape[1]
        gamma, beta = self.film(theta).chunk(2, dim=-1)
        if not self.config.use_film_beta:
            beta = torch.zeros_like(beta)
        conditioned = (
            (1.0 + gamma[:, None, :]) * z_mutation + beta[:, None, :]
        )
        subtype_expanded = subtype_embedding[:, None, :].expand(
            -1, query_count, -1
        )
        feature_parts = [conditioned, self.passage_embedding(query_passage.long())]
        if self.passage_pair_embedding is not None:
            if passage_pair is None:
                raise ValueError(
                    "passage_pair is required when use_passage_pair_feature=True"
                )
            feature_parts.append(self.passage_pair_embedding(passage_pair.long()))
        predictor_count = (
            torch.log1p(mutation_count).unsqueeze(-1)
            if self.config.use_predictor_mutation_count
            else z_mutation.new_zeros((*mutation_count.shape, 1))
        )
        feature_parts.extend([subtype_expanded, predictor_count])
        features = torch.cat(feature_parts, dim=-1)
        return self.predictor(features).squeeze(-1)

    def forward(
        self,
        batch: SerumMutationSetBatch | Mapping[str, Any],
    ) -> dict[str, torch.Tensor]:
        batch = self._as_batch(batch)
        encoded = self.encode_query_mutations(batch)
        background = encoded["z_background"]
        batch_size, query_count = batch.query_ha.shape[:2]
        device = batch.reference_ha.device
        serum_passage = (
            batch.serum_passage
            if batch.serum_passage is not None
            else self._default_ids(batch.reference_ha, (batch_size,))
        ).to(device)
        query_passage = (
            batch.query_passage
            if batch.query_passage is not None
            else self._default_ids(batch.reference_ha, (batch_size, query_count))
        ).to(device)
        passage_pair = (
            batch.passage_pair
            if batch.passage_pair is not None
            else serum_passage[:, None] * self.config.passage_vocab_size + query_passage
        ).to(device) if self.passage_pair_embedding is not None else None
        subtype = (
            batch.subtype
            if batch.subtype is not None
            else self._default_ids(batch.reference_ha, (batch_size,))
        ).to(device)

        theta, subtype_embedding = self._condition(
            background,
            serum_passage,
            subtype,
        )
        query_score = self._predict_score(
            encoded["z_mutation"],
            encoded["mutation_count"],
            theta,
            query_passage,
            passage_pair,
            subtype_embedding,
        )
        empty_z, empty_count = self._encode_empty_mutation(background)
        self_query_passage = serum_passage[:, None]
        self_passage_pair = (
            serum_passage[:, None] * self.config.passage_vocab_size
            + self_query_passage
        ) if self.passage_pair_embedding is not None else None
        self_score = self._predict_score(
            empty_z,
            empty_count,
            theta,
            self_query_passage,
            self_passage_pair,
            subtype_embedding,
        )
        self_score = self_score.expand_as(query_score)
        mean = self_score - query_score
        if self.config.use_output_identity_bias:
            serum_name = (
                batch.serum_name
                if batch.serum_name is not None
                else self._default_ids(batch.reference_ha, (batch_size,))
            ).to(device)
            query_virus = (
                batch.query_virus
                if batch.query_virus is not None
                else self._default_ids(batch.reference_ha, (batch_size, query_count))
            ).to(device)
            identity_bias = (
                self.serum_name_bias(serum_name.long()).squeeze(-1)[:, None]
                + self.query_virus_bias(query_virus.long()).squeeze(-1)
            )
            mean = mean + identity_bias
        else:
            identity_bias = torch.zeros_like(mean)

        out = {
            **encoded,
            "mean": mean,
            "self_score": self_score,
            "query_score": query_score,
            "theta_serum": theta,
            "output_identity_bias": identity_bias,
        }
        if batch.labels is not None:
            labels = batch.labels.float().to(device)
            query_mask = (
                batch.query_mask.float().to(device)
                if batch.query_mask is not None
                else None
            )
            huber = F.smooth_l1_loss(mean, labels, beta=0.5, reduction="none")
            label_weights = _label_bin_weights(
                labels,
                self.config.label_weight_thresholds,
                self.config.label_weight_values,
            ).to(device)
            rank_loss = _pairwise_ranking_loss(
                mean,
                labels,
                query_mask,
                margin=self.config.rank_loss_margin,
                min_label_delta=self.config.rank_loss_min_label_delta,
            )
            rank_term = self.config.rank_loss_weight * rank_loss
            out["rank_loss"] = rank_loss
            weighted_huber = _weighted_masked_mean(
                huber,
                label_weights,
                query_mask,
            )
            task_bias = _weighted_masked_mean(
                mean - labels,
                torch.ones_like(labels),
                query_mask,
            )
            task_bias_loss = task_bias.square()
            out["task_bias"] = task_bias
            out["task_bias_loss"] = task_bias_loss
            out["huber_loss"] = (
                weighted_huber
                + rank_term
                + self.config.task_bias_loss_weight * task_bias_loss
            )
        return out


__all__ = [
    "DistanceBiasedSelfAttention",
    "MutationSetPool",
    "SerumMutationSetBatch",
    "SerumMutationSetConfig",
    "SerumMutationSetMinusModel",
]

"""
Metric HA antigenic-space model branch.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pooling import attention_mask


@dataclass
class MetricHAAntigenicModelConfig:
    hidden_size: int
    latent_dim: int = 128
    passage_vocab_size: int = 6
    passage_pair_vocab_size: int = 36
    subtype_vocab_size: int = 1
    epsilon: float = 1e-8
    na_lambda_max: float = 0.25
    use_serum_scale: bool = True
    use_assay_bias: bool = True
    use_na_glycan_residual: bool = True
    use_homologous_titer_in_serum_scale: bool = False
    serum_name_vocab_size: int = 0
    virus_name_vocab_size: int = 0
    ha_mismatch_dim: int = 0


@dataclass
class MetricAntigenicBatch:
    serum_ha: torch.Tensor
    virus_ha: torch.Tensor
    serum_ha_mask: Optional[torch.Tensor] = None
    virus_ha_mask: Optional[torch.Tensor] = None
    serum_passage: Optional[torch.Tensor] = None
    test_passage: Optional[torch.Tensor] = None
    passage_pair: Optional[torch.Tensor] = None
    passage_mismatch: Optional[torch.Tensor] = None
    source: Optional[torch.Tensor] = None
    s_nagly: Optional[torch.Tensor] = None
    subtype: Optional[torch.Tensor] = None
    serum_name: Optional[torch.Tensor] = None
    virus_name: Optional[torch.Tensor] = None
    ha_mismatch: Optional[torch.Tensor] = None
    use_name_bias: bool = True
    homologous_titer_z: Optional[torch.Tensor] = None
    labels: Optional[torch.Tensor] = None


class MetricHAAntigenicModel(nn.Module):
    """
    HI-derived distance = rho_A * d_HA(A, B) + assay bias + small NA residual.
    """

    def __init__(self, config: MetricHAAntigenicModelConfig | Mapping[str, Any]):
        super().__init__()
        if isinstance(config, Mapping):
            config = MetricHAAntigenicModelConfig(**dict(config))
        self.config = config

        self.ha_dropout = nn.Dropout(p=0.1)
        self.ha_pooler = attention_mask(embed_size=config.hidden_size)
        self.ha_projection = nn.Linear(config.hidden_size, config.latent_dim)

        self.ha_dim_weight_logit = nn.Parameter(torch.zeros(config.latent_dim))
        self.ha_distance_scale_logit = nn.Parameter(torch.zeros(1))

        self.rho_intercept = nn.Parameter(torch.zeros(1))
        self.rho_from_z = nn.Linear(config.latent_dim, 1, bias=False)
        self.rho_passage_effect = nn.Embedding(config.passage_vocab_size, 1)
        self.rho_homologous_weight = nn.Parameter(torch.zeros(1))

        self.passage_pair_bias = nn.Embedding(config.passage_pair_vocab_size, 1)

        self.subtype_na_glycan_logit = nn.Embedding(config.subtype_vocab_size, 1)
        self.use_name_bias = config.serum_name_vocab_size > 0 or config.virus_name_vocab_size > 0
        if self.use_name_bias:
            self.serum_name_bias = nn.Embedding(max(1, config.serum_name_vocab_size), 1, padding_idx=0)
            self.virus_name_bias = nn.Embedding(max(1, config.virus_name_vocab_size), 1, padding_idx=0)
        self.use_ha_mismatch_residual = config.ha_mismatch_dim > 0
        if self.use_ha_mismatch_residual:
            self.ha_mismatch_residual = nn.Linear(config.ha_mismatch_dim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.ha_projection.weight)
        nn.init.zeros_(self.ha_projection.bias)
        nn.init.zeros_(self.rho_from_z.weight)
        nn.init.zeros_(self.rho_passage_effect.weight)
        nn.init.zeros_(self.passage_pair_bias.weight)
        nn.init.zeros_(self.subtype_na_glycan_logit.weight)
        if self.use_name_bias:
            nn.init.zeros_(self.serum_name_bias.weight)
            nn.init.zeros_(self.virus_name_bias.weight)
        if self.use_ha_mismatch_residual:
            nn.init.zeros_(self.ha_mismatch_residual.weight)
            nn.init.zeros_(self.ha_mismatch_residual.bias)

    @staticmethod
    def _as_batch(batch: MetricAntigenicBatch | Mapping[str, Any]) -> MetricAntigenicBatch:
        if isinstance(batch, MetricAntigenicBatch):
            return batch
        if isinstance(batch, Mapping):
            allowed = {field.name for field in fields(MetricAntigenicBatch)}
            return MetricAntigenicBatch(**{key: value for key, value in batch.items() if key in allowed})
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    @staticmethod
    def _default_ids(reference: torch.Tensor, fill_value: int = 0) -> torch.Tensor:
        return torch.full(
            (reference.shape[0],),
            fill_value=fill_value,
            dtype=torch.long,
            device=reference.device,
        )

    @staticmethod
    def _centered_embedding(embedding: nn.Embedding, ids: torch.Tensor) -> torch.Tensor:
        centered_weight = embedding.weight - embedding.weight.mean(dim=0, keepdim=True)
        return torch.embedding(centered_weight, ids.long()).view(-1)

    def encode_ha(self, ha_matrix: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        ha_matrix = self.ha_dropout(ha_matrix)
        if mask is None:
            mask = torch.ones(
                ha_matrix.shape[0],
                ha_matrix.shape[1],
                dtype=ha_matrix.dtype,
                device=ha_matrix.device,
            )
        pooled = self.ha_pooler(ha_matrix, mask=mask, save_attention_path=None)
        return self.ha_projection(pooled)

    def compute_ha_distance(self, serum_z: torch.Tensor, virus_z: torch.Tensor) -> torch.Tensor:
        dim_weight = F.softplus(self.ha_dim_weight_logit)
        scale = F.softplus(self.ha_distance_scale_logit)
        weighted_sq = ((serum_z - virus_z).pow(2) * dim_weight).sum(dim=1)
        eps = float(self.config.epsilon)
        distance = torch.where(
            weighted_sq > 0,
            torch.sqrt(weighted_sq.clamp_min(0.0) + eps),
            torch.zeros_like(weighted_sq),
        )
        return scale * distance

    def compute_serum_scale(self, serum_z: torch.Tensor, batch: MetricAntigenicBatch) -> torch.Tensor:
        if not self.config.use_serum_scale:
            return torch.ones(serum_z.shape[0], dtype=serum_z.dtype, device=serum_z.device)

        serum_passage = batch.serum_passage
        if serum_passage is None:
            serum_passage = self._default_ids(serum_z)

        raw = self.rho_intercept + self.rho_from_z(serum_z).view(-1)
        raw = raw + self._centered_embedding(self.rho_passage_effect, serum_passage)
        if self.config.use_homologous_titer_in_serum_scale and batch.homologous_titer_z is not None:
            raw = raw + self.rho_homologous_weight * batch.homologous_titer_z.view(-1).to(raw.device)
        return F.softplus(raw)

    def compute_assay_bias(self, batch: MetricAntigenicBatch, reference: torch.Tensor) -> torch.Tensor:
        if not self.config.use_assay_bias:
            return torch.zeros(reference.shape[0], dtype=reference.dtype, device=reference.device)

        if batch.passage_pair is not None:
            passage_pair = batch.passage_pair
        else:
            serum_passage = batch.serum_passage if batch.serum_passage is not None else self._default_ids(reference)
            test_passage = batch.test_passage if batch.test_passage is not None else self._default_ids(reference)
            passage_pair = serum_passage.long() * int(self.config.passage_vocab_size) + test_passage.long()
        return self.passage_pair_bias(passage_pair.long().to(reference.device)).view(-1)

    def compute_na_residual(
        self,
        batch: MetricAntigenicBatch,
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        zeros = torch.zeros(reference.shape[0], dtype=reference.dtype, device=reference.device)
        if not self.config.use_na_glycan_residual:
            return zeros, zeros, zeros

        s_nagly = batch.s_nagly if batch.s_nagly is not None else zeros
        s_nagly = s_nagly.view(-1).float().to(reference.device)
        subtype = batch.subtype if batch.subtype is not None else self._default_ids(reference)
        lambda_logit = self.subtype_na_glycan_logit(subtype.long().to(reference.device)).view(-1)
        lambda_nagly = self.config.na_lambda_max * torch.sigmoid(lambda_logit)
        r_na = lambda_nagly * s_nagly
        return s_nagly, lambda_nagly, r_na

    def compute_name_bias(self, batch: MetricAntigenicBatch, reference: torch.Tensor) -> torch.Tensor:
        zeros = torch.zeros(reference.shape[0], dtype=reference.dtype, device=reference.device)
        if not self.use_name_bias or not batch.use_name_bias:
            return zeros
        if batch.serum_name is None or batch.virus_name is None:
            raise ValueError("serum_name and virus_name are required when name bias is enabled")
        serum_name = batch.serum_name.long().to(reference.device)
        virus_name = batch.virus_name.long().to(reference.device)
        return self.serum_name_bias(serum_name).view(-1) + self.virus_name_bias(virus_name).view(-1)

    def compute_ha_mismatch_residual(
        self,
        batch: MetricAntigenicBatch,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        zeros = torch.zeros(reference.shape[0], dtype=reference.dtype, device=reference.device)
        if not self.use_ha_mismatch_residual:
            return zeros
        if batch.ha_mismatch is None:
            raise ValueError("ha_mismatch is required when HA mismatch residual is enabled")
        ha_mismatch = batch.ha_mismatch.float().to(reference.device)
        return self.ha_mismatch_residual(ha_mismatch).view(-1)

    def forward(self, batch: MetricAntigenicBatch | Mapping[str, Any]) -> dict[str, torch.Tensor | None]:
        batch = self._as_batch(batch)

        serum_z = self.encode_ha(batch.serum_ha, batch.serum_ha_mask)
        virus_z = self.encode_ha(batch.virus_ha, batch.virus_ha_mask)
        d_ha = self.compute_ha_distance(serum_z, virus_z)
        rho_ha = self.compute_serum_scale(serum_z, batch)
        b_assay = self.compute_assay_bias(batch, d_ha)
        s_nagly, lambda_nagly, r_na = self.compute_na_residual(batch, d_ha)
        name_bias = self.compute_name_bias(batch, d_ha)
        ha_mismatch_residual = self.compute_ha_mismatch_residual(batch, d_ha)

        rho_times_distance = rho_ha * d_ha
        pred = rho_times_distance + b_assay + r_na + name_bias + ha_mismatch_residual

        loss = None
        if batch.labels is not None:
            labels = batch.labels.view(-1).float().to(pred.device)
            loss = F.smooth_l1_loss(pred.view(-1), labels, beta=0.5)

        return {
            "pred": pred.view(-1),
            "d_ha": d_ha.view(-1),
            "rho_ha": rho_ha.view(-1),
            "rho_ha_times_d_ha": rho_times_distance.view(-1),
            "b_assay": b_assay.view(-1),
            "s_nagly": s_nagly.view(-1),
            "r_na": r_na.view(-1),
            "name_bias": name_bias.view(-1),
            "ha_mismatch_residual": ha_mismatch_residual.view(-1),
            "lambda_nagly": lambda_nagly.view(-1),
            "z_serum_ha": serum_z,
            "z_virus_ha": virus_z,
            "loss": loss,
        }

    def assay_l2_penalty(self) -> torch.Tensor:
        return self.passage_pair_bias.weight.pow(2).mean()

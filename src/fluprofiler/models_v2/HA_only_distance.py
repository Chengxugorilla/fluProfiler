"""
HA-only distance v2 model.

This model keeps the attention-weighted HA antigen space and replaces the
downstream MLP regressor with Euclidean distance plus a passage-pair bias.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ha import fluProfiler_HA_v2
from .io import BatchInput, ModelOutput, normalize_labels_1d


class fluProfiler_HA_only_distance_v2(fluProfiler_HA_v2):
    """
    HA-only model with a simple geometric prediction head:
    pred = softplus(scale) * ||serum_HA - virus_HA||_2 + global_bias + passage_pair_bias
    """

    def __init__(
        self,
        config,
        args,
        passage_vocab_size: int = 6,
        serum_name_vocab_size: int = 0,
        virus_name_vocab_size: int = 0,
    ):
        super().__init__(
            config=config,
            args=args,
            channel_names=("serum_HA", "virus_HA"),
            passage_vocab_size=passage_vocab_size,
            passage_dim=256,
        )
        self.passage_vocab_size = passage_vocab_size
        self.distance_scale_logit = nn.Parameter(torch.zeros(1))
        self.global_bias = nn.Parameter(torch.zeros(1))
        self.passage_pair_bias = nn.Embedding(passage_vocab_size * passage_vocab_size, 1)
        nn.init.zeros_(self.passage_pair_bias.weight)
        self.use_name_bias = serum_name_vocab_size > 0 or virus_name_vocab_size > 0
        if self.use_name_bias:
            self.serum_name_bias = nn.Embedding(max(1, serum_name_vocab_size), 1, padding_idx=0)
            self.virus_name_bias = nn.Embedding(max(1, virus_name_vocab_size), 1, padding_idx=0)
            nn.init.zeros_(self.serum_name_bias.weight)
            nn.init.zeros_(self.virus_name_bias.weight)

    def _name_bias(self, batch: BatchInput, device: torch.device, batch_size: int) -> torch.Tensor:
        if not self.use_name_bias or batch.meta.get("use_name_bias", True) is False:
            return torch.zeros(batch_size, device=device)
        if "serum_name_ids" not in batch.meta or "virus_name_ids" not in batch.meta:
            raise ValueError("serum_name_ids and virus_name_ids are required when name bias is enabled")
        serum_ids = batch.meta["serum_name_ids"].long().to(device)
        virus_ids = batch.meta["virus_name_ids"].long().to(device)
        return self.serum_name_bias(serum_ids).view(-1) + self.virus_name_bias(virus_ids).view(-1)

    def _passage_pair_ids(self, passage_tokens: torch.Tensor) -> torch.Tensor:
        tokens = passage_tokens.long()
        if tokens.ndim != 2:
            raise ValueError("passage_tokens must have shape (batch, length)")
        if tokens.shape[1] >= 4:
            serum_passage = tokens[:, 1]
            virus_passage = tokens[:, 3]
        elif tokens.shape[1] >= 2:
            serum_passage = tokens[:, 0]
            virus_passage = tokens[:, 1]
        else:
            raise ValueError("passage_tokens must include serum and virus passage tokens")
        return serum_passage * self.passage_vocab_size + virus_passage

    def forward(self, batch: BatchInput | Dict, labels: Optional[torch.Tensor] = None) -> ModelOutput:
        batch = self._as_batch_input(batch)
        masks = batch.matrix_masks or {}

        serum_vector = self._encode_channel(batch.matrices["serum_HA"], masks.get("serum_HA"))
        virus_vector = self._encode_channel(batch.matrices["virus_HA"], masks.get("virus_HA"))
        distance = torch.linalg.vector_norm(serum_vector - virus_vector, dim=1)

        if batch.passage_tokens is None:
            raise ValueError("passage_tokens is required for fluProfiler_HA_only_distance_v2")
        passage_ids = self._passage_pair_ids(batch.passage_tokens).to(distance.device)
        passage_bias = self.passage_pair_bias(passage_ids).view(-1)

        name_bias = self._name_bias(batch, distance.device, distance.shape[0])
        logits = F.softplus(self.distance_scale_logit) * distance + self.global_bias + passage_bias + name_bias
        logits = logits.view(-1, 1)

        target = normalize_labels_1d(labels if labels is not None else batch.labels)
        loss = None
        if target is not None:
            target = target.to(logits.device)
            loss = self.loss_fct(logits.view(-1), target.view(-1).float())

        return ModelOutput(
            logits=logits,
            pred=logits.view(-1),
            loss=loss,
            extras={
                "serum_antigen_vector": serum_vector,
                "virus_antigen_vector": virus_vector,
                "distance": distance,
                "passage_bias": passage_bias,
                "name_bias": name_bias,
            },
        )

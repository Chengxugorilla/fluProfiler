"""
v2 HA model with unified I/O contract.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn

from ..models.losses import create_activate, create_loss_function
from ..models.pooling import attention_mask
from .io import BatchInput, ModelOutput, normalize_labels_1d


class fluProfiler_HA_v2(nn.Module):
    """
    New HA model implementation (v2) that keeps legacy behavior conceptually,
    but uses unified BatchInput/ModelOutput contracts.
    """

    def __init__(
        self,
        config,
        args,
        channel_names: Iterable[str] = ("serum_HA", "virus_HA"),
        passage_vocab_size: int = 6,
        passage_dim: int = 256,
    ):
        super().__init__()
        self.config = config
        self.args = args
        self.channel_names = tuple(channel_names)
        self.output_mode = getattr(args, "output_mode", "regression")
        self.num_labels = 1
        self.loss_reduction = getattr(config, "loss_reduction", None) or "meanmean"

        hidden_size = int(config.hidden_size)
        self.matrix_dropout = nn.Dropout(p=0.1)
        self.matrix_pooler = attention_mask(embed_size=hidden_size)

        fc_size = getattr(config, "matrix_fc_size", None)
        self.matrix_mlp, self.matrix_out_dim = self._build_matrix_mlp(hidden_size, fc_size, config)

        self.passage_encoder = nn.Sequential(
            nn.Embedding(num_embeddings=passage_vocab_size, embedding_dim=passage_dim),
            nn.ReLU(),
            nn.Linear(passage_dim, passage_dim),
        )
        self.passage_dim = passage_dim

        concat_dim = self.matrix_out_dim * len(self.channel_names) + self.passage_dim
        (
            self.dropout,
            self.hidden_layer,
            self.hidden_act,
            self.classifier,
            self.output_layer,
            self.loss_fct,
        ) = create_loss_function(
            config=config,
            args=args,
            hidden_size=concat_dim,
            classifier_size=getattr(args, "classifier_size", concat_dim),
            sigmoid=getattr(args, "sigmoid", False),
            output_mode=self.output_mode,
            num_labels=self.num_labels,
            loss_type=getattr(args, "loss_type", None),
            ignore_index=getattr(args, "ignore_index", -100),
            return_types=["dropout", "hidden_layer", "hidden_act", "classifier", "output", "loss"],
        )

    @staticmethod
    def _build_matrix_mlp(input_dim: int, fc_size, config):
        if fc_size is None or (isinstance(fc_size, (list, tuple)) and len(fc_size) == 0):
            return None, input_dim
        if isinstance(fc_size, (list, tuple)):
            dims = [int(v) for v in fc_size]
        else:
            dims = [int(fc_size)]
        layers = []
        cur = input_dim
        for out_dim in dims:
            layers.append(nn.Linear(cur, out_dim))
            layers.append(create_activate(getattr(config, "fc_activate_func", "relu")))
            cur = out_dim
        return nn.Sequential(*layers), cur

    @staticmethod
    def _fallback_mask(matrix: torch.Tensor) -> torch.Tensor:
        # Fallback behavior: all tokens valid when explicit mask is missing.
        return torch.ones(matrix.shape[0], matrix.shape[1], device=matrix.device, dtype=matrix.dtype)

    def _encode_channel(self, matrix: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        matrix = self.matrix_dropout(matrix)
        if mask is None:
            mask = self._fallback_mask(matrix)
        pooled = self.matrix_pooler(matrix, mask=mask, save_attention_path=None)
        if self.matrix_mlp is not None:
            pooled = self.matrix_mlp(pooled)
        return pooled

    @staticmethod
    def _as_batch_input(batch) -> BatchInput:
        if isinstance(batch, BatchInput):
            return batch
        if isinstance(batch, dict):
            return BatchInput(
                matrices=batch.get("matrices", {}),
                matrix_masks=batch.get("matrix_masks"),
                passage_tokens=batch.get("passage_tokens"),
                labels=batch.get("labels"),
                meta=batch.get("meta", {}) or {},
            )
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    def forward(self, batch: BatchInput | Dict, labels: Optional[torch.Tensor] = None) -> ModelOutput:
        batch = self._as_batch_input(batch)
        channel_vectors = []
        masks = batch.matrix_masks or {}
        for channel in self.channel_names:
            if channel not in batch.matrices:
                raise KeyError(f"Missing channel matrix: {channel}")
            channel_vectors.append(self._encode_channel(batch.matrices[channel], masks.get(channel)))

        concat_vector = torch.concat(channel_vectors, dim=1)

        if batch.passage_tokens is None:
            raise ValueError("passage_tokens is required for fluProfiler_HA_v2")
        passage_vector = torch.mean(self.passage_encoder(batch.passage_tokens), dim=1)
        concat_vector = torch.concat([concat_vector, passage_vector], dim=1)

        if self.dropout is not None:
            concat_vector = self.dropout(concat_vector)
        concat_vector = self.hidden_layer(concat_vector)
        concat_vector = self.hidden_act(concat_vector)

        logits = self.classifier(concat_vector)
        pred = self.output_layer(logits) if self.output_layer is not None else logits

        target = normalize_labels_1d(labels if labels is not None else batch.labels)
        loss = None
        if target is not None:
            target = target.to(logits.device)
            if self.output_mode in ["regression"]:
                loss = self.loss_fct(logits.view(-1), target.view(-1))
            else:
                loss = self.loss_fct(logits.view(-1), target.view(-1).float())

        return ModelOutput(
            logits=logits,
            pred=pred.view(-1),
            loss=loss,
            extras={"concat_vector": concat_vector},
        )

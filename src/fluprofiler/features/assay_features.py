"""
Low-capacity categorical assay bias components.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AssayBiasModule(nn.Module):
    """
    Additive categorical assay bias with centered effect groups.
    """

    def __init__(
        self,
        passage_vocab_size: int,
        mismatch_vocab_size: int,
        source_vocab_size: int = 1,
    ):
        super().__init__()
        self.serum_passage = nn.Embedding(passage_vocab_size, 1)
        self.test_passage = nn.Embedding(passage_vocab_size, 1)
        self.passage_mismatch = nn.Embedding(mismatch_vocab_size, 1)
        self.source = nn.Embedding(source_vocab_size, 1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.serum_passage.weight)
        nn.init.zeros_(self.test_passage.weight)
        nn.init.zeros_(self.passage_mismatch.weight)
        nn.init.zeros_(self.source.weight)

    @staticmethod
    def _centered_lookup(embedding: nn.Embedding, ids: torch.Tensor) -> torch.Tensor:
        centered_weight = embedding.weight - embedding.weight.mean(dim=0, keepdim=True)
        return torch.embedding(centered_weight, ids.long()).view(-1)

    def forward(
        self,
        serum_passage: torch.Tensor,
        test_passage: torch.Tensor,
        passage_mismatch: torch.Tensor,
        source: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if source is None:
            source = torch.zeros_like(serum_passage)
        return (
            self._centered_lookup(self.serum_passage, serum_passage)
            + self._centered_lookup(self.test_passage, test_passage)
            + self._centered_lookup(self.passage_mismatch, passage_mismatch)
            + self._centered_lookup(self.source, source)
        )

    def l2_penalty(self) -> torch.Tensor:
        penalty = 0.0
        for embedding in (
            self.serum_passage,
            self.test_passage,
            self.passage_mismatch,
            self.source,
        ):
            centered = embedding.weight - embedding.weight.mean(dim=0, keepdim=True)
            penalty = penalty + centered.pow(2).mean()
        return penalty

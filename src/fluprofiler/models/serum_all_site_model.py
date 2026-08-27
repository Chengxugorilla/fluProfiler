"""All-site token variant for the E2 mutation-representation ablation."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from fluprofiler.models.serum_mutation_set_model import (
    UNKNOWN_AMINO_ACID_ID,
    SerumMutationSetBatch,
    SerumMutationSetMinusModel,
)


class SerumAllSiteModel(SerumMutationSetMinusModel):
    """Use every valid aligned HA site as a token instead of mutations only."""

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
        site_mask = (
            (batch.reference_aligned_mask[:, None, :] > 0)
            & (batch.query_aligned_mask > 0)
            & (reference_aa != UNKNOWN_AMINO_ACID_ID)
            & (batch.query_aa != UNKNOWN_AMINO_ACID_ID)
        )
        packed, packed_mask, positions, counts = self._pack_mutation_tokens(
            all_tokens,
            site_mask,
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
            "mutation_attention": pooling_attention.reshape(batch_size, query_count, -1),
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

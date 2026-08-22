# Mutation-to-Query Background Cross-Attention Design

## Goal

Create an isolated MutationSet-Minus variant in which the encoded mutation
tokens attend to the complete per-site query HA context before mutation pooling.
The current MutationSet self-attention remains unchanged.

## Architecture

For each serum-query pair, existing mutation tokens are first passed through
the existing mutation self-attention blocks. Query HA site representations are
projected from `query_projected` to `mutation_dim`. A cross-attention layer then
uses mutation tokens as queries and all valid query site tokens as keys and
values:

`mutation = mutation + gate * CrossAttention(mutation, query_sites)`.

The resulting mutation tokens go through the existing `MutationSetPool` and
predictor unchanged. The query-site mask is `query_embedding_mask`; padded and
gap sites therefore cannot receive attention. Query mutation positions remain
eligible keys/values because they are part of the desired contextual sequence.

## Constraints

- Preserve the current mutation self-attention and its distance bias exactly.
- The new cross-attention has no structural-distance bias, so ND remains
  controlled by the supplied no-bias matrix and NP remains controlled by the
  existing passage-pair flag.
- Use a scalar residual gate initialized to zero. At initialization the new
  model must exactly match its base model's prediction.
- Implement in new model and training-entry files only; do not modify current
  MutationSet, FusedFiLM, or NameResidual files.
- Return cross-attention weights for analysis.

## Configuration

Add independent configuration fields for cross-attention heads and dropout,
defaulting to the current mutation attention head count and dropout. Validate
that `mutation_dim` is divisible by the head count.

## Tests

1. Gate initialized to zero preserves the base prediction exactly.
2. Invalid query sites have zero cross-attention weight.
3. Changing a valid query-background site changes the cross-attended mutation
   representation when the gate is enabled.
4. Gradients reach the cross-attention projections and residual gate.

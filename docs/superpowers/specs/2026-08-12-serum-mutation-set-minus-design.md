# Serum Mutation Set Minus Design

## Objective

Build a new zero-shot serum model that represents each prediction as the
combination of a serum/reference HA1 background and an explicit set of
directed reference-to-query mutations. The model is a parallel experiment:
the existing `SerumGate-Minus` model, its batch type, and its training entry
point remain unchanged.

The first implementation tests two hypotheses:

1. Explicit mutation-set encoding generalizes better to unseen complete
   mutation combinations than independently pooling full reference and query
   sequences.
2. A learned HA1 residue-distance attention bias improves mutation-combination
   generalization beyond mutation content and position alone.

## Scope and Isolation

Create the following new files:

- `src/fluprofiler/models/serum_mutation_set_model.py`
- `experiments/serum_gate/train_serum_mutation_set.py`
- `tests/test_serum_mutation_set_model.py`
- `tests/test_train_serum_mutation_set.py`

Do not modify:

- `src/fluprofiler/models/serum_gate_model.py`
- `src/fluprofiler/models/serum_gate_minus_model.py`
- `experiments/serum_gate/train_zero_shot_minus.py`

The new trainer may import stable data-loading, metrics, and output helpers
from the existing trainer. It must not monkey-patch the existing trainer or
existing model classes.

## Inputs

One serum task contains:

- reference HA embeddings: `[B, L, H]`
- query HA embeddings: `[B, Q, L, H]`
- aligned reference amino-acid IDs: `[B, L]`
- aligned query amino-acid IDs: `[B, Q, L]`
- valid-position masks for reference and query
- serum passage, query passage, passage-pair, subtype, glycan mismatch,
  labels, and query mask matching the existing trainer semantics

`B` is initially restricted to 1, consistent with the existing zero-shot
serum task collator. `H` is inferred from the stored HA embedding matrices.

The trainer must use aligned `serumHA` and `virusHA` strings to derive residue
IDs and mutation masks. It must align embedding rows to these strings so that
embedding index, amino-acid ID, HA1 position, and distance-matrix position
refer to the same coordinate.

The residue vocabulary contains the 20 standard amino acids plus `X`, gap,
and padding as distinct entries. A mutation token is generated when both
positions are valid and the aligned reference and query symbols differ.

## New Batch Type

Define an independent `SerumMutationSetBatch` dataclass. It carries all model
inputs and has no effect on `SerumGateBatch`. The new trainer owns conversion,
collation, and device movement for this batch.

## Distance-Matrix Contract

The training entry point accepts a required `--ha-distance-matrix` path for
the first experiment. Supported formats are `.csv`, `.tsv`, `.npy`, and
`.npz`. CSV/TSV files may contain a header and index column; the loader must
validate that the resulting numeric matrix is square.

Matrix index 0 corresponds to aligned HA1 position 0. Its size must cover all
aligned positions used by the selected dataset. The loader must reject a
smaller matrix rather than silently shifting or padding coordinates.

Finite off-diagonal values must be non-negative. Missing values remain
missing. The first `9 x 9` missing block and any other missing entries have
zero structural attention bias, so they fall back to content attention. They
must never be filled with zero as a measured distance.

The loaded matrix is registered as a non-trainable model buffer and saved in
the checkpoint through the state dict. The resolved input path and matrix
shape are recorded in the run configuration.

## Model Architecture

### Configuration

`SerumMutationSetConfig` contains the existing loss and categorical-condition
settings required by this experiment plus:

- `hidden_size`: inferred HA embedding width
- `site_dim=64`
- `background_dim=128`
- `mutation_dim=128`
- `position_dim=32`
- `amino_acid_dim=16`
- `presence_dim=4`
- `mutation_attention_heads=4`
- `mutation_attention_layers=1`
- `mutation_ffn_dim=256`
- `attention_dropout=0.1`
- `predictor_hidden_dim=256`
- `min_log_var=-6.0`
- `max_log_var=4.0`
- `score_log_var_mode="sum"`

`mutation_dim` must be divisible by `mutation_attention_heads`.

### Shared Site Projection

Reference and query site embeddings use one shared module:

```text
LayerNorm(H) -> Linear(H, 64) -> GELU
```

This prevents duplicate `H -> 64` parameter matrices.

### Reference Background

Masked-mean pool the projected reference sites and apply:

```text
LayerNorm(64) -> Linear(64, 128) -> GELU -> Linear(128, 128)
```

The output `z_background: [B, 128]` represents the complete serum/reference
background. The mutation-token branch also retains the contextual reference
embedding at every mutated position.

### Mutation Token Construction

For a mutation at aligned position `i`, construct:

```text
reference_site                    64
query_site - reference_site       64
abs(query_site - reference_site)  64
position_embedding(i)             32
reference_amino_acid_embedding    16
query_amino_acid_embedding        16
reference_presence_embedding       4
query_presence_embedding           4
                                  ---
                                  264
```

The same amino-acid embedding table is used for reference and query. Their
separate concatenation slots preserve mutation direction. Presence embeddings
distinguish residue-to-residue substitutions, deletions, and insertions.

Project the 264-dimensional vector with:

```text
Linear(264, 128) -> GELU -> LayerNorm(128)
```

Add a learned projection of `z_background` to every mutation token. Thus a
mutation is conditioned on both its local contextual reference representation
and the global serum/reference background.

### Distance-Biased Mutation Transformer

Only mutation tokens attend to one another. The initial implementation uses
one pre-normalized Transformer block with four heads:

```text
X = X + MHA(LayerNorm(X), per-head distance bias)
X = X + FFN(LayerNorm(X))
FFN = Linear(128, 256) -> GELU -> Dropout -> Linear(256, 128)
```

For mutation positions `i` and `j`, head `h` receives the additive bias:

```text
bias[h, i, j] = alpha[h] * exp(-distance[i, j] / tau[h])
```

where `alpha[h] = softplus(raw_alpha[h])` and
`tau[h] = softplus(raw_tau[h]) + epsilon`. The diagonal and missing-distance
pairs receive zero bias. Initial `alpha` is small so the model begins close to
ordinary content attention, while each head can learn a different useful
distance scale.

Use a focused custom attention module because PyTorch `MultiheadAttention`
does not provide a clean batch-specific, per-head bias interface for variable
mutation sets. The implementation must apply mutation padding masks before
softmax and avoid NaNs for padded rows.

### Mutation-Set Pooling

Pool contextual mutation tokens through two branches:

- masked mean pooling
- learned scalar attention pooling

Concatenate both 128-dimensional outputs with `log1p(mutation_count)` and
apply:

```text
Linear(257, 128) -> GELU -> LayerNorm(128)
```

The result is `z_mutation: [B, Q, 128]`.

For an empty mutation set, insert one learned `NULL_MUTATION` token with a
valid mask. Its reported mutation count remains zero. It follows the same
background conditioning, Transformer, and pooling path as non-empty sets.

### Serum Condition and Predictor

Encode the serum state from:

```text
[z_background, serum_passage_embedding, subtype_embedding]
    -> Linear(input, 128) -> ReLU -> Linear(128, 128)
```

Use this 128-dimensional state to FiLM-condition `z_mutation`:

```text
[gamma, beta] = Linear(theta_serum, 256)
z_conditioned = (1 + gamma) * z_mutation + beta
```

Concatenate `z_conditioned` with query-passage embedding, passage-pair
embedding, expanded subtype embedding, glycan mismatch, and
`log1p(mutation_count)`. Predict score and raw log variance with:

```text
Linear(input, 256) -> GELU -> Dropout(0.1)
-> Linear(256, 256) -> GELU -> Linear(256, 2)
```

Clamp log variance to the configured interval.

## Minus Semantics

Compute query and self scores through the same model path:

```text
query_score = f(reference background, reference -> query mutation set,
                serum/query experimental conditions)
self_score  = f(reference background, empty mutation set,
                serum/self experimental conditions)
mean        = self_score - query_score
```

For `score_log_var_mode="sum"`:

```text
log_var = log(exp(self_log_var) + exp(query_log_var))
```

`score_log_var_mode="query"` remains available as an ablation.

## Outputs and Losses

Return at least:

- `mean`, `log_var`
- `self_score`, `query_score`
- `self_log_var`, `query_log_var`
- `mutation_count`
- `mutation_attention`
- `z_background`, `z_mutation`

When labels are present, provide `nll_loss`, `huber_loss`, and `rank_loss`
using the same mathematical definitions, label-bin weighting, query masking,
and within-serum ranking semantics as the existing model.

## Training Entry Point

The independent CLI mirrors the existing zero-shot trainer where applicable,
including fixed splits, `--refit-train-valid`, ALL/subtype training,
embedding loading and GPU cache, label weighting, cosine scheduling, progress,
checkpoints, predictions, and per-subtype metrics for ALL runs.

Mutation-model-specific CLI arguments expose the dimensions and Transformer
settings above plus `--ha-distance-matrix`. The run config identifies the
model as `SerumMutationSet-Minus` and records all new configuration fields.

The trainer validates aligned HA columns unconditionally because mutation
identity and coordinates depend on them.

## Error Handling

Fail before training when:

- aligned HA columns are absent or inconsistent within a serum task
- a query aligned sequence length differs from its reference aligned length
- an embedding cannot be aligned to every non-gap residue
- a position exceeds `max_position_embeddings`
- the distance matrix is non-square, too small, or contains negative finite
  distances
- attention dimensions are invalid

Missing structural distances are valid data and do not trigger errors.

## Testing and Acceptance

Model tests must demonstrate:

- mutation masks and direction distinguish `K -> N` from `N -> K`
- position embeddings distinguish identical substitutions at different sites
- missing distance entries produce exactly zero structural bias
- finite near distances produce greater RBF bias than finite far distances
- empty mutation sets use the null token without NaNs
- padded mutation tokens do not affect pooled representations
- a complete forward pass returns `[B, Q]` outputs and finite losses
- gradients reach distance-scale, mutation-token, and shared site-projection
  parameters

Trainer tests must demonstrate:

- aligned strings produce correct residue IDs and mutation masks
- embedding-to-alignment coordinate mapping is exact across gaps
- supported distance formats load consistently and invalid matrices fail
- the new CLI requires a distance matrix and leaves the existing CLI unchanged
- the new collator produces the independent batch type

Acceptance requires all new tests plus existing SerumGate-Minus tests to pass
inside the `fluProfiler` Conda environment, a successful Python compile check,
and a CPU smoke forward/backward pass. No existing model or trainer file may
appear in the implementation diff.

## Deferred Work

The first experiment deliberately excludes mutation-to-full-reference
cross-attention, GNN layers, training-neighbor retrieval, serum-name IDs,
pretrained mutation embeddings, and auxiliary contrastive losses. These are
separate ablations after the explicit mutation-set baseline is measured.

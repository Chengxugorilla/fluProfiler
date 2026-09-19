# H3N2 Virus-HA Token Attention Analysis Design

## Goal

Analyze the final token-level pooling attention learned by the trained
SerumGate-Minus all-attention checkpoint. The analysis covers unique H3N2
virus HA sequences only (`seq_id_c`) and preserves all 331 embedding tokens.

## Inputs

- Checkpoint:
  `results/H1H3_HA1_v1.0/20260717_164256/SerumGate-Minus-AllAttention-latent8/whole/subtype/H3N2/checkpoints/best_model.pth`
- Dataset:
  `data/dataset/H1H3_HA1_v1.0/splited/20260717_164256/whole/whole.csv`
- Embeddings: `data/embedding/files/matrix_<seq_id_c>.pt`
- Model class: `SerumGateMinusAllAttentionModel`

The trained checkpoint is from epoch 50 and uses independent HA encoding,
pure low-rank attention pooling, latent dimension 8, attention dimension 64,
and four attention heads.

## Population and Weighting

Filter the dataset to H3N2, select virus-side HA (`seq_id_c` and `virusHA`),
and deduplicate by `seq_id_c`. Each of the 2,611 unique virus HA sequences
contributes exactly once to aggregate statistics, regardless of its number of
occurrences in the training table.

All sequences have 329 aligned HA amino-acid positions. Some contain `-` or
`X`; these tokens remain in place and are not removed or imputed.

## Attention Definition

Load the checkpoint strictly into `SerumGateMinusAllAttentionModel`, switch
the model to evaluation mode, and run inference without gradients. For each
embedding, call:

```python
model.score_model.ha_encoder.encode_with_attention(embedding, mask)
```

Use the returned 331-element softmax vector. This is the final scalar token
weight used to form the pooled HA representation. It is not the internal
331-by-331 multi-head self-attention matrix.

Token mapping is:

- token 0: beginning special token
- tokens 1 through 329: HA amino-acid positions 1 through 329
- token 330: ending special token

Special-token weights are retained in every calculation. Each per-sequence
attention vector must be finite, nonnegative, and sum to one within numerical
tolerance.

## Processing

Process unique embeddings in configurable batches on a configurable device.
Do not load all embeddings onto the GPU at once. Preserve deterministic input
order and write one output row per unique `seq_id_c`.

Aggregate each token across unique sequences with equal weight. Compute count,
mean, median, standard deviation, minimum, maximum, and the 5th, 25th, 75th,
and 95th percentiles. Also count how often each token appears in a sequence's
top 1 and top 5 attention positions.

## Outputs

Write results under:

`results/H1H3_HA1_v1.0/20260717_164256/SerumGate-Minus-AllAttention-latent8/whole/subtype/H3N2/attention_token_analysis/`

- `attention_by_sequence.csv`: one row per `seq_id_c`, sequence metadata, and
  `token_0` through `token_330`.
- `token_summary.csv`: one row per token with mapping and aggregate statistics.
- `top_tokens.csv`: tokens sorted by mean attention, including rank and summary
  statistics.
- `mean_attention_profile.png`: mean token profile with interquartile band.
- `top_tokens.png`: bar chart of the highest-mean tokens.
- `attention_heatmap.png`: all unique-sequence attention vectors, sorted by
  their attention center of mass.
- `analysis_config.json`: resolved inputs, checkpoint epoch and configuration,
  device, batch size, counts, token mapping, and validation results.

## Validation and Failure Handling

Fail before producing final outputs if the checkpoint cannot be loaded
strictly, an embedding is missing, an embedding does not have shape
`[331, 2560]`, sequence IDs map inconsistently to HA strings, attention values
are non-finite or negative, attention rows do not sum to one, or the final
unique-sequence count differs from 2,611.

Write outputs atomically through a temporary output directory so a failed run
does not leave a result directory that looks complete.

## Interpretation Boundary

The reported values describe the model's learned pooling weights. They are
useful for identifying positions emphasized when constructing HA
representations, but they are not causal effects on predicted antigenic
distance. Gradient or perturbation attribution is outside this analysis.

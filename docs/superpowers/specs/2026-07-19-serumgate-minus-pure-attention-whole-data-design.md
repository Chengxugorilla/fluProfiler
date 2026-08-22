# SerumGate-Minus Pure-Attention Whole-Data Model Design

## Goal

Build a final H3N2 SerumGate-Minus model for HA-site interpretability. The
model trains on every row from the existing `titer/seed_0` train, validation,
and test partitions. Its HA encoder uses 100% learned attention pooling and no
mean-pooling path. After training, it exports per-sequence attention weights
and a global aligned-position importance summary.

This model is an interpretation model, not an unbiased evaluation model.
Because the former test rows enter training, its fit metrics must not be
reported as held-out performance.

## Non-goals

- Do not modify the source split CSV files.
- Do not change the behavior of existing SerumGate models, pooling modes,
  training scripts, command-line defaults, or checkpoints.
- Do not train a joint H1N1/H3N2 model. The whole dataset retains both
  subtypes, but this model filters to H3N2 at training time.
- Do not add a mean-pooling contribution or a trainable attention/mean gate.
- Do not treat attention as proof of biological causality.
- Do not add an NA branch, subtype embedding, or ranking loss to this run.

## Whole Dataset

### Location and files

Create:

```text
data/dataset/H1H3_HA1_v1.0/whole/
├── all.csv
└── manifest.json
```

The builder reads, in this order:

```text
data/dataset/H1H3_HA1_v1.0/splited/20260717_164256/titer/seed_0/train.csv
data/dataset/H1H3_HA1_v1.0/splited/20260717_164256/titer/seed_0/valid.csv
data/dataset/H1H3_HA1_v1.0/splited/20260717_164256/titer/seed_0/test.csv
```

It concatenates the three frames without adding, removing, or renaming CSV
columns. In particular, `all.csv` must not contain a `source_split` column.
It retains both H1N1 and H3N2 rows and preserves source row order within each
partition. It does not perform content-based deduplication, because the source
files are the authoritative fixed partitions and silent deduplication could
change the empirical weighting.

### Validation

Before writing output, the builder must verify:

- all three source files exist;
- their column names and order are identical;
- required SerumGate columns are present;
- `label` is numeric;
- the output row count equals the sum of all three source row counts; and
- both H1N1 and H3N2 are retained when present in the source data.

The write must be atomic: generate temporary sibling files, validate them, and
replace the final files only after successful completion. Existing output is
not overwritten unless an explicit overwrite option is supplied.

### Manifest

`manifest.json` records provenance without changing `all.csv`:

- creation timestamp;
- source directory and the three source filenames;
- source file checksums;
- source row counts;
- total output row count;
- exact column list;
- subtype counts;
- exact duplicate-row count as a diagnostic only;
- builder command/options; and
- output checksum.

## Pure-Attention Architecture

### Compatibility boundary

Keep the existing `attention`, `mean`, and `lowrank_attention` modes unchanged.
Add a new explicit pooling mode named:

```text
lowrank_attention_only
```

Existing configuration defaults remain unchanged. Old checkpoints must load
through their existing code paths without new required keys.

### Pooling module

Add a separately named pure-attention pooling module rather than changing the
formula of `LowRankSelfAttentionPool`. For an HA embedding matrix
`H in R^(L x 2560)`, it computes:

```text
C = MultiHeadSelfAttention(Linear64(LayerNorm(H)))
logits_i = Linear1(tanh(Dropout(C_i)))
alpha = masked_softmax(logits)
pooled = sum_i alpha_i H_i
z = Linear8(pooled)
```

Run configuration:

- input embedding dimension: inferred from files, currently 2560;
- low-rank attention dimension: 64;
- attention heads: 4;
- attention dropout: 0.2;
- HA latent dimension: 8; and
- HA pair mode: `independent`.

The pure module must not instantiate, calculate, or mix a mean-pooled vector.
It has no attention/mean gate parameter. For every nonempty sequence, valid
site weights sum to one and masked positions have zero weight.

### SerumGate-Minus scoring

The rest of SerumGate-Minus remains structurally unchanged. The shared HA
encoder independently creates `z_r` and `z_q`. The conditioned score model
uses:

```text
[z_q, z_r, z_q - z_r, abs(z_q - z_r)]
```

together with the serum prior, passage features, and the existing glycan
feature. The same score model computes:

```text
self_score  = f(reference, reference)
query_score = f(reference, query)
distance    = self_score - query_score
```

With `score_log_var_mode=sum`, distance variance is the sum of self-score and
query-score variances. Training uses the existing weighted Gaussian NLL.

### Attention-return interface

The pure pooler exposes attention weights only when explicitly requested, for
example by a `return_attention` argument. The default forward return type and
behavior of all existing modules remain unchanged. Normal training must not
retain attention tensors in memory after the forward pass.

## Whole-Data Training

Provide a dedicated whole-data training entry point so the interpretation
workflow cannot be confused with fixed-split evaluation. It reads `all.csv`,
filters rows using `--type H3N2`, and treats every retained row as training
data.

The initial run uses:

```text
model: SerumGate-Minus
HA pooling: lowrank_attention_only
HA attention dimension: 64
HA attention heads: 4
HA attention dropout: 0.2
HA latent dimension: 8
HA pair mode: independent
serum task columns: seq_id_a,serumPassCat,serumName
maximum queries per task: 32
epochs: 200
optimizer: AdamW
learning rate: 1e-4
weight decay: 0.01
scheduler: cosine
minimum learning rate: 1e-6
loss: weighted Gaussian NLL
score variance mode: sum
NA branch: none
subtype feature: disabled by the H3N2 filter
ranking loss: disabled
```

Because there is no validation partition, the final epoch is the selected
checkpoint. The run configuration must include an explicit marker such as
`evaluation_status: "train_all_no_holdout"` and a warning that metrics are
in-sample diagnostics.

Use a new result namespace that cannot be mistaken for the fixed-split runs,
for example:

```text
results/H1H3_HA1_v1.0/20260717_164256/
└── SerumGate-Minus-PureAttn-latent8/
    └── whole/subtype/H3N2/
```

The training entry point must reject a nonempty output directory unless the
user explicitly requests overwrite behavior.

## Attention Export and Aggregation

### Interpretation input

After training, load the final checkpoint, switch to evaluation mode, and
collect unique HA sequence IDs from H3N2 rows across `seq_id_a` and `seq_id_c`.
Each unique embedding is encoded once. This avoids weighting frequently
occurring viruses more heavily merely because they appear in more assay rows.

The exported weights are the final scalar masked-softmax weights used in the
pooling sum. They are not the internal `L x L` multi-head attention matrices.

### Position mapping

Embedding rows correspond to the ungapped HA sequence. Map each token to the
existing aligned HA representation (`serumHA` for `seq_id_a`, `virusHA` for
`seq_id_c`) by walking the alignment and skipping gaps. Validate for every
unique sequence that:

- the embedding is two-dimensional;
- embedding length equals the ungapped HA length;
- a sequence ID maps to one consistent sequence/alignment; and
- all attention weights are finite and sum to one within numerical tolerance.

Fail with the sequence ID and observed lengths when mapping is ambiguous or
inconsistent. Use one-based aligned positions in human-facing outputs and
retain a zero-based token index for exact tensor correspondence.

### Files

Write under the run directory:

```text
interpretability/
├── attention_per_sequence.csv
├── attention_position_summary.csv
├── attention_top_sites.csv
├── attention_profile.png
└── interpretation_config.json
```

`attention_per_sequence.csv` contains:

```text
seq_id,role,token_index,aligned_position,amino_acid,attention_weight
```

`role` is `reference`, `query`, or `both`, based on how the unique sequence ID
appears in the H3N2 whole dataset.

`attention_position_summary.csv` contains:

```text
aligned_position,mean_attention,median_attention,std_attention,max_attention,
sequence_count,reference_sequence_count,query_sequence_count,rank
```

The primary global rank is descending mean attention across unique sequences
that have a residue at that aligned position. `attention_top_sites.csv`
contains the top 20 aligned positions, their summary statistics, observed
amino-acid distribution, and sequence count.

`attention_profile.png` is a wide academic-style plot of aligned position
against mean attention, with a variability band and labels for the top 20
positions. `interpretation_config.json` records the checkpoint checksum,
dataset checksum, filtering rule, position convention, aggregation rule, and
export timestamp.

## Error Handling

The workflow fails before expensive training or export when:

- whole-data files or required columns are missing;
- H3N2 filtering produces no rows;
- an embedding file is missing or malformed;
- attention dimension is not divisible by the number of heads;
- an embedding cannot be mapped unambiguously to its aligned sequence;
- attention values are nonfinite or fail normalization checks; or
- an output directory is already nonempty without explicit overwrite.

Errors identify the affected file or sequence ID and must not leave a result
that looks complete. Completion markers/configuration are written only after
all required artifacts succeed.

## Verification and Regression Safety

Tests must cover:

1. Whole-data construction preserves the exact source schema, concatenation
   order, total row count, and both subtypes, with no `source_split` column.
2. Manifest counts and checksums match the generated CSV.
3. Existing `attention`, `mean`, and `lowrank_attention` modes produce the same
   interfaces and remain loadable from existing-style configurations.
4. `lowrank_attention_only` contains no mean-pooling or gate parameter.
5. Pure-attention weights have the expected shape, sum to one over valid
   positions, and assign zero weight to masks.
6. Gradients reach low-rank projection, self-attention, site-score, and final
   projection parameters.
7. Training reads only `all.csv`, filters H3N2 correctly, and saves the final
   epoch with the no-holdout marker.
8. Per-sequence export deduplicates sequence IDs, maps token indices to aligned
   positions, and produces deterministic weights in evaluation mode.
9. Global aggregation weights unique sequences equally and ranks positions
   deterministically.
10. A small end-to-end smoke test builds a synthetic whole dataset, trains one
    short epoch, reloads the checkpoint, and produces all interpretation
    artifacts.

Before completion, run the focused SerumGate and dataset-builder tests plus the
existing SerumGate regression suite. No existing data or result directories
are deleted or rewritten during tests.

## Interpretation Boundary

The output supports the claim that the trained model assigned greater pooling
weight to particular aligned HA positions. It does not by itself establish a
causal biological effect. Reports and plot labels should use language such as
"attention-based site importance" rather than "causal site effect."

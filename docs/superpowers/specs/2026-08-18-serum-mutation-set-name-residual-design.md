# MutationSet-ND Name Residual Design

## Goal

Extend `SerumMutationSet-Minus-ND` with tightly regularized categorical
effects for `serumName` and `virusName`.  These effects should absorb stable,
non-sequence assay residuals while keeping the existing mutation-set pathway
responsible for sequence-derived antigenic distance.

The feature is intended to improve the H3N2 serum split, where almost every
test virus name is observed in training, without relying on future virus
names in season evaluation.  In completed H3N2 season splits, only 0.5--2.2%
of test rows have a training-seen virus name; serum-split coverage is
99.0--100.0%.

## Model

The existing model produces a distance mean `d_seq` and its existing
uncertainty outputs.  The new mean is:

```text
d_hat = d_seq + b_serum[serum_name_id] + b_virus[virus_name_id]
```

`b_serum` and `b_virus` are separate embedding tables with embedding
dimension one.  Their `UNK` rows are fixed at zero.  The residuals are added
only at the final distance-mean prediction; they do not enter site projection,
background encoding, mutation token construction, distance-biased attention,
FiLM, pooling, or the uncertainty head.

The model keeps the current distance variance unchanged and evaluates its
existing NLL or weighted-Huber loss using `d_hat` as the mean.

## Vocabularies and OOV semantics

Names are normalized by trimming leading/trailing whitespace, converting to
uppercase, and collapsing internal whitespace.  Each vocabulary is built from
the training frame only, with ID zero reserved for `UNK`.

Train, validation, and test rows use the same train-built vocabulary.  Any
unseen or missing name maps to `UNK`, whose contribution is exactly zero.
This means the virus-name branch has no direct signal for novel season viruses
and naturally falls back to the sequence model.

## Regularization

The training loss is extended with explicit ridge penalties:

```text
L_total = L_existing + lambda_serum * sum(b_serum[1:]^2)
                     + lambda_virus * sum(b_virus[1:]^2)
```

Penalties apply to the entire non-UNK tables on every optimization step, not
only the IDs present in a batch.  This is intentionally separate from global
AdamW weight decay.

The initial sweep preserves Nextflu's regularization asymmetry:

```text
lambda_serum in {1e-4, 3e-4, 1e-3, 3e-3}
lambda_virus = 10 * lambda_serum
```

The numeric values are not copied from Nextflu because its quadratic
least-squares objective is on a different scale.  The 1:10 ratio is the
transferable prior: `virusName` is more collinear with viral sequence and is
therefore shrunk more strongly.

## Data flow and configuration

The dataset and collator expose integer tensors for `serum_name_id` per task
and `query_virus_name_id` per query.  The model config gains:

- `use_name_residual: bool = false`
- `serum_name_residual_l2: float`
- `virus_name_residual_l2: float`

The disabled default must reproduce current MutationSet-ND behavior.  Run
configuration records both vocabularies, normalization policy, OOV counts per
split, and the two penalty values.

## Experiments and acceptance criteria

Run the following ablations with matched data, seed, loss, and training
schedule:

1. MutationSet-ND baseline;
2. serum-name residual only;
3. virus-name residual only;
4. both residuals, over the regularization sweep.

Primary evaluation is the H3N2 serum split over seeds 0--9, using pooled MAE
and Pearson.  Season 39--44 is the regression guard.  Report each metric for
all rows plus train-seen-name and OOV-name strata.

Acceptance requires a reproducible serum improvement against the matched ND
baseline without material degradation in season MAE or Pearson.  Diagnostics
must include name frequency versus learned residual, residual magnitude
distribution, and the fraction of predictions receiving nonzero serum or
virus corrections.

## Risks and safeguards

Categorical names can memorize assay labels rather than biology.  The final
additive-only location, train-only vocabularies, zero OOV fallback, and strong
separate L2 penalties are the safeguards.  Results must not be interpreted as
improved sequence generalization unless the OOV-name strata also improve.

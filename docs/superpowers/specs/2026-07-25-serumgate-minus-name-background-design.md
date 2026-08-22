# SerumGate-Minus name background design

## Goal

Allow `serumName` and `virusName` to account for independent experimental or
strain background effects during SerumGate-Minus training.  These fields must
not introduce an additional serum-virus interaction pathway.

## Scope

The change applies to:

- `src/fluprofiler/models/serum_gate_model.py` for batch/config support;
- `src/fluprofiler/models/serum_gate_minus_model.py` for the final distance
  adjustment; and
- `experiments/serum_gate/train_zero_shot_minus.py` for train-only vocabularies,
  dataset IDs, collators, checkpoint metadata, and CLI configuration.

The existing HA/NA encoders, passage/subtype features, conditioned score model,
uncertainty estimate, and ratio-aware self-minus-query calculation remain
unchanged.

## Architecture

With name backgrounds enabled, the predicted antigenic distance is:

`f(reference, reference) - f(reference, query) + b_serum(serumName) + b_virus(virusName)`.

`b_serum` and `b_virus` are separate `nn.Embedding(vocab_size, 1,
padding_idx=0)` tables.  They are zero-initialized, and their outputs are added
only to `SerumGateMinusModel`'s final `mean`.  They do not affect `log_var`,
the wrapped score model, or any intermediate serum-virus feature interaction.

Each name is represented as an integer ID in `SerumGateBatch`: a serum ID of
shape `(B,)` and virus IDs of shape `(B, Q)`.  The final mean adds the serum
bias broadcast across query positions and the per-query virus bias.

The feature is disabled by default.  In that mode no name column is required,
the model behavior is unchanged, and legacy checkpoints remain loadable.

## Vocabulary and data flow

When enabled, `serumName` and `virusName` are required in every split.  Two
separate vocabularies are built exclusively from the training frame.  ID `0`
is reserved for missing names and names unseen during validation/test/inference.
The dataset maps rows to these IDs, both collators place them in `SerumGateBatch`,
and `move_batch` preserves them on the selected device.

The run configuration and checkpoint metadata record the feature flag and both
name-to-ID mappings so inference can recreate the exact mapping.  The mappings
are not enlarged with validation or test names.

## Error handling

- Enabling the feature without either required column raises a clear `ValueError`.
- Empty or missing names map to ID `0` rather than becoming trainable categories.
- Disabled name backgrounds ignore any name columns.

## Tests

Tests will prove that:

1. train-only vocabularies reserve zero for missing/unseen names;
2. task datasets and both collators preserve serum/query name IDs and shapes;
3. zero-initialized name biases leave predictions unchanged;
4. setting known name biases shifts only `mean` by the expected additive amount
   while leaving `log_var` unchanged; and
5. disabled mode remains compatible with batches lacking name IDs.

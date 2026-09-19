# MutationSet Name Residual Implementation Plan

> **For implementation:** Build this as additive files only. Do not edit the
> existing `train_serum_mutation_set.py` or `serum_mutation_set_model.py`.

## Goal

Add an independent MutationSet-Minus-ND training entry point whose final
prediction includes strongly regularized discrete scalar effects for serum and
virus names.

## Tasks

1. Add tests for zero-initialized scalar effects, UNK behavior, prediction
   composition, and explicit regularization.
2. Add a model wrapper that reuses the existing sequence architecture without
   changing it and recomputes the supervised loss from the adjusted prediction.
3. Add a standalone trainer that creates normalized train-only Name vocabularies
   (train+valid after refit), supplies IDs to the wrapper, records diagnostics,
   and saves vocabularies in configurations and checkpoints.
4. Run focused unit tests, then the existing MutationSet tests to confirm the
   original implementation remains unaffected.

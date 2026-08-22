# AdaBoost Subtype Filter Design

## Goal

Add an explicit `--subtypes` option to `train_adaboost_fixed_split.py` so a run only loads and trains the requested influenza subtypes.

## CLI

- Follow the Nextflu CLI convention: `--subtypes H3N2` or `--subtypes H1N1 H3N2`.
- Define the argument with `nargs="+"` and an empty default.
- Fail through `argparse` when no subtype is supplied.

## Data Flow

1. Load the existing fixed `train.csv`, `valid.csv`, and `test.csv` files without creating a new external split.
2. Filter every loaded frame to the requested subtype set before row counts, feature construction, training, or evaluation.
3. When requested, merge the filtered valid frame into the filtered train frame.
4. Validate requested subtypes against the effective training frame; report missing and available subtype names.
5. Train one AdaBoost model per requested subtype and evaluate only the filtered test rows.

## Outputs

- Preserve the existing output files and model format.
- Record the requested subtype list in `run_config.json`.
- Make input and usable row counts describe the filtered run.

## Tests

Cover a single subtype, multiple subtypes, omitted `--subtypes`, and a requested subtype absent from the effective training data. Existing fixed-split and train-valid merge behavior must remain unchanged.

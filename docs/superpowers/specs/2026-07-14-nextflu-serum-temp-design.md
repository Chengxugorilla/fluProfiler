# Nextflu Serum Temporary Dataset Design

## Goal

Create a temporary fixed-split dataset for the legacy Nextflu baseline at
`data/dataset/H1H3_HA1_homo/splited/20260711_154610/serum_temp`.
Nextflu expects its target in a column named `label`, while the antigenic
distance target in the source dataset is stored in `diff_label`.

## Source And Output

- Source: `data/dataset/H1H3_HA1_homo/splited/20260711_154610/serum`
- Output: `data/dataset/H1H3_HA1_homo/splited/20260711_154610/serum_temp`
- Files: `train.csv`, `valid.csv`, `test.csv`, and `manifest.json`

The source directory and all files in it remain unchanged.

## Transformation

For each split CSV:

1. Read the source file while preserving row order and all columns.
2. Require `diff_label` to exist and contain numeric, non-null values.
3. Replace the output `label` values with the corresponding `diff_label`
   values.
4. Keep `diff_label` and every other column unchanged.
5. Write the transformed frame to the matching file in `serum_temp`.

Copy `manifest.json` unchanged because the split method, source data, counts,
and leakage checks are unchanged. The `_temp` directory name documents that
the CSV label mapping is specific to the Nextflu compatibility run.

## Validation

After writing all files:

- each output split has the same row count and column order as its source;
- each output `label` equals its source `diff_label` exactly;
- all columns except `label` equal their source columns;
- the copied manifest matches the source manifest.

The existing Nextflu command can then use `serum_temp` as `--data-dir` without
any code or CLI changes.

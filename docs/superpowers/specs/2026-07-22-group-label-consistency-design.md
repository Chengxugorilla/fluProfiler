# Group Label Consistency Analysis Design

## Goal

Analyze whether continuous `label` values are consistent within groups defined by
`seq_a`, `seq_c`, `serumPassCat`, and `virusPassCat`. Rank groups with large
within-group disagreement so they can be reviewed before model training.

## Input

- CSV: `data/dataset/H1H3_HA1_Crick-CNIC/processed/source.csv`
- Required columns: `seq_a`, `seq_c`, `serumPassCat`, `virusPassCat`, `label`
- Observed size: 142,943 rows and approximately 341 MB
- `serumPassCat` and `virusPassCat` contain a small number of missing values;
  missing categories remain eligible for grouping.
- `label` is treated as a continuous numeric value. Floating-point near-equality
  is therefore handled by dispersion statistics rather than exact value counts.

## Approach

Use pandas and load only the five required columns. Convert `label` to numeric,
report and exclude rows whose labels cannot be parsed, then group with
`dropna=False` so missing passage categories are not silently discarded.

For every group, calculate:

- `count`
- `mean`
- sample standard deviation (`std`)
- `min`, first quartile (`q1`), `median`, third quartile (`q3`), and `max`
- `range = max - min`
- `iqr = q3 - q1`

Single-row groups have no measurable within-group disagreement. They remain in
the complete report, with `std` filled as zero, but are excluded from the
high-disagreement report.

## Disagreement Rule

Mark a multi-row group as high disagreement when either condition is true:

- `std >= 0.5`
- `range >= 1.0`

Both thresholds are constants at the top of the script so the user can adjust
them without changing the analysis logic. Rank flagged groups by descending
`std`, then descending `range`, then descending `count`.

## Outputs

- Print input-quality counts and the 30 most inconsistent groups.
- Write a CSV containing statistics for every group.
- Write a second CSV containing only high-disagreement groups.

Output paths are derived from the input directory and do not modify the source
CSV.

## Error Handling and Verification

Fail with a clear message if the input file or required columns are missing.
Report invalid or missing labels before excluding them. Verification uses a
small synthetic DataFrame with consistent, inconsistent, singleton, and
missing-category groups, followed by a read-only run against the real CSV to
confirm that reports are produced with the expected columns and sorting.

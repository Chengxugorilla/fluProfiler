# Gradient Attribution Analysis Table Design

## Goal

Simplify `paper/Code/gradient_x_input_analysis.ipynb` around one reusable
analysis table.  Preserve the one-to-one merge of per-sample attribution data
with sample metadata and dates.

## Data flow

1. Load `whole.csv`, filter H3N2 HA1 pairs, deduplicate them, and assign the
   stable `sample_index` used by the attribution export.
2. Retain only the metadata needed for analysis, including identifiers, names,
   and dates.
3. Load `attribution_by_sample.csv` and verify its `sample_index` has the same
   order as the processed records.
4. Build `attribution_with_dates` using the existing five-key, one-to-one
   merge:

   ```python
   attribution_with_dates = attribution.merge(
       record_dates,
       on=["sample_index", "seq_id_a", "seq_id_c", "serumName", "virusName"],
       how="left",
       validate="one_to_one",
   )
   ```

5. Derive query-site summaries from that table.  A helper identifies
   `query_token_*` columns, applies an optional row filter, and returns
   sorted mean absolute attribution by token position.

## Notebook layout

- Configuration and imports
- `build_records`, `load_attributions`, `build_analysis_table`, and
  `summarize_query_sites` helpers
- Main-table construction and integrity checks
- A concise example: filter one serum and show its ranked query-site summary

## Scope

Remove unrelated and duplicate workflows: NPZ loading, checkpoint metrics,
global attribution figures, Top-25 prior-label tests, and their plots.  The
notebook will retain the same source paths, filtering, deduplication, merge
keys, and query-token aggregation semantics.

## Validation

Run the notebook code sequentially in the `fluProfiler` Conda environment;
confirm the merge is one-to-one, metadata columns have no missing values, and
the example summary contains token positions 1 through 329.

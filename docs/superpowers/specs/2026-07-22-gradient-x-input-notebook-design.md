# Gradient × Input Analysis Notebook Design

Date: 2026-07-22

## Goal

Create an executed, self-contained Jupyter notebook that reproduces and visualizes the H3N2 Gradient × Input analyses for the existing SerumGate-Minus latent-8 titer model.

## Output

The only new analysis artifact is:

`paper/Code/gradient_x_input_analysis.ipynb`

The notebook stores executed cell outputs, including figures. It must not call `savefig` or write image files.

## Inputs

The notebook reads the existing files below through a configurable result-directory cell:

- `gradient_x_input/analysis_config.json`
- `gradient_x_input/attribution_arrays.npz`
- `gradient_x_input/attribution_by_sample.csv`
- `gradient_x_input/site_summary.csv`
- sibling `metrics.csv` and `run_config.json`
- the source `whole.csv` recorded in `analysis_config.json`

It validates file existence, sample counts, token counts, array shapes, token mapping, and sample ordering before analysis.

## Analysis Sections

1. Environment, paths, and input validation.
2. Prediction and label distributions plus final model test metrics.
3. Global reference, query, and combined attribution profiles.
4. Global site rankings using `mean(abs(attribution))`, signed mean, variability, and positive fraction.
5. Comparison with canonical H3 sites 145, 155, 156, 158, 159, 189, and 193.
6. Enrichment summaries for aligned HA1 regions 139–163 and 187–193.
7. Reference/query cancellation analysis.
8. Mutated-versus-conserved site analysis using paired HA1 sequences.
9. Attribution-sum versus prediction completeness checks.
10. Season-stratified top-site analysis and interpretation limitations.

## Seasonal Figures

The notebook joins the attribution rows to the source metadata using the same deduplication key used by the attribution exporter: `seq_a`, `seq_c`, `serumPassCat`, and `virusPassCat`.

It converts `virusDate` to the existing project NH/SH season convention and restricts the main seasonal display to 2014NH through 2023SH. Missing or invalid dates are reported and excluded only from seasonal analyses.

For each season and aligned HA1 position, the displayed residue weight is:

`mean(abs(reference_attribution + query_attribution))`

The seasonal top-site set is the ordered union of the top 15 positions in each nonempty season.

Two notebook-only figures are produced:

1. A colored table grouping the seasonal top-site union into Epitope A, B, C, D, E, Unknown, and Unreported, using the position definitions already present in `paper/Code/Fig.3.ipynb`.
2. A grayscale heatmap of residue weights across seasons, with rows ordered by epitope group and position and y-axis labels colored by epitope group.

No `#` or `*` suffixes are added because the current analysis inputs do not define them.

## Plotting and Reproducibility

- Use Matplotlib and Seaborn with deterministic sorting and fixed styling.
- Call `plt.show()` for every figure and close figures after display.
- Do not create image directories or external image files.
- Keep all paths in one configuration cell.
- Use aligned HA1 positions 1–329; exclude BOS and EOS from residue rankings.
- Preserve raw, unnormalized values for tables. Heatmaps may use an explicitly labeled within-row normalization only if both raw and normalized values remain available; the primary heatmap uses raw seasonal mean absolute attribution.

## Error Handling

The notebook raises clear errors for missing files, mismatched sample order, unexpected array shapes, duplicate metadata join keys, missing required columns, and absent seasonal observations. It reports excluded-date counts rather than silently dropping rows.

## Verification

Execute the notebook with the project `fluProfiler` Conda environment and verify:

- all cells complete without errors;
- execution outputs and figures are embedded in the notebook;
- no image files are created;
- expected sample count is 24,740 and token count is 331;
- combined global top positions include the previously observed leading sites;
- the two requested seasonal figures render with nonempty seasons and residue rows;
- the notebook JSON is valid and contains no `savefig` calls.

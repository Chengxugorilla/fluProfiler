# Top-25 Prior Alignment Notebook Design

## Goal

Extend the H3N2 Gradient × Input notebook with a fully code-driven comparison between the global combined mean-absolute Top-25 residues and the structural priors in `paper/data/ABCDE.csv`.

## Inputs and Validation

- Read `paper/data/ABCDE.csv` directly from a notebook path constant.
- Require its four columns: `Amino acid residues of HA1`, `Antigenic epitope`, `Receptor binding site`, and `glycosylation site`.
- Require unique integer positions between 1 and 329, epitope values from A–E or Un, and `√` as the sole positive RBS/glycosylation marker.
- Use `top30_combined` / `combined_summary` already computed in the notebook and select the first 25 rows by descending `mean_abs`.

## Analysis

Create a 329-position annotation table with multi-label fields:

- `epitope` (A–E, Un, or absent)
- `is_epitope_prior`
- `is_receptor_binding_site`
- `is_glycosylation_site`
- `mean_abs` and global `rank_abs`
- `is_top25`

The analysis must treat epitope, receptor-binding, and glycosylation labels as overlapping; it must not assign each position to only one category.

For the overall epitope prior, each individual epitope class, RBS, and glycosylation prior, calculate:

- number of prior positions among HA1 positions 1–329;
- number and proportion of Top-25 hits;
- background proportion;
- 2×2 Fisher exact odds ratio and one-sided enrichment P value;
- mean `mean_abs` and mean global rank within the prior versus its complement.

Use a stable odds-ratio convention: report infinity when all observed non-prior Top-25 hits are zero; report NaN when a 2×2 table has a structurally undefined odds ratio.

## Notebook Outputs

Add one markdown introduction, one Top-25 annotation table, one aggregate enrichment table, and one inline Matplotlib figure. The figure shows the 25 ranked sites as horizontal bars colored by epitope label; RBS and glycosylation membership are represented by markers, with a legend.

Do not manually specify hits, write image files, use Seaborn, or modify existing analysis cells. All labels and overlaps must be derived from `ABCDE.csv`.

## Interpretation

The notebook must state that Fisher tests are descriptive enrichment tests for this selected Top-25 set and do not establish causal antigenic effects. The rank/mean-absolute comparisons provide a threshold-free complementary view.

## Verification

- Execute the notebook in the `fluProfiler` Conda environment with all outputs embedded.
- Assert no error outputs and no image-writing calls.
- Assert Top-25 has exactly 25 unique positions.
- Assert `ABCDE.csv` contains 114 unique positions, 8 RBS sites, and 6 glycosylation sites.
- Assert the aggregate table includes Overall epitope prior, A–E, Un, RBS, and Glycosylation.

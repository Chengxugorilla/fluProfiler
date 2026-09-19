# Gradient Attribution Seasonal Top-15 Design

## Goal

Extend `paper/Code/gradient_x_input_analysis.ipynb` so the integrated
`attribution_with_dates` table produces query-site Top-15 rankings per influenza
season and a union across those rankings.

## Season assignment

Use `virusDate` only.  Parse it as a calendar date and assign:

- `YYYYNH` for 1 February through 31 August of year `YYYY`.
- `YYYSH` for 1 September of year `YYYY` through 31 January of year `YYYY + 1`.

For example, 2023-09-01 through 2024-01-31 belongs to `2023SH`. Records with
missing or unparseable `virusDate` receive no season, are excluded from seasonal
ranking, and are counted in the notebook output.

## Analysis flow

1. Build `attribution_with_dates` using the existing five-key, one-to-one merge.
2. Add a `flu_season` column from `virusDate`.
3. Group the table by `flu_season`; for each group calculate mean absolute
   attribution for query positions 1–329 and retain the 15 highest positions.
4. Concatenate all seasonal Top-15 tables and group by position to make the
   union. For each union position, show included seasons, number of seasons,
   best rank, and per-season ranks.

## Outputs

- `season_top15`: one row per season-position selection, with season, rank,
  position, mean absolute attribution, and sample count.
- `season_top15_union`: one row per unique selected position, sorted by number
  of selected seasons then best rank.
- A compact season sample-count table and an excluded-record count.

## Validation

Verify all assigned labels follow the date boundaries; every populated season
has exactly 15 ranked positions; union positions equal the set union of
`season_top15`; and all source records remain in `attribution_with_dates`.

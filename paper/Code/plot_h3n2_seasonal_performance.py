"""Generate the corrected H3N2 seasonal performance rank figure."""

import numpy as np

from plot_h3n2_seasonal_performance_without_name_baselines import (
    HIGHER_BETTER,
    OUTPUT_CSV,
    load_values,
    plot_rank_matrix,
)


def main() -> None:
    values = load_values()
    values["rank"] = np.nan
    for (_, metric), index in values.groupby(["season", "metric"]).groups.items():
        values.loc[index, "rank"] = values.loc[index, "value"].rank(
            method="min", ascending=metric not in HIGHER_BETTER
        )
    values["rank"] = values["rank"].astype(int)
    values.to_csv(OUTPUT_CSV, index=False)
    plot_rank_matrix(values)
    print(f"Saved corrected figure and values: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

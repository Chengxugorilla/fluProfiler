# Figure 4 WHO vaccine-strain ranking: results package

## Scope and interpretation

This package summarizes the legacy SCMS-FiLM H3N2 vaccine-selection analysis across six seasonal panels (2023Feb to 2025Sep). It is a retrospective ranking analysis, not a strict prospective validation.

CELL and EGG candidate sera were ranked separately, while both used the same season-specific circulating test-virus panel. Lower mean predicted antigenic distance corresponds to a higher coverage score and a better rank. The 2023Feb panel retained the legacy collection-date weighting; all other panels used the unweighted mean over test viruses.

The WHO-recommended candidate was evaluated within its matching manufacturing passage category only. The figures retain WHO highlighting but do not display numerical WHO ranks.

Important implementation note: the reported candidate number is the number of ranked candidate entries. In early panels, candidates sharing the same sequence and passage can be merged by the legacy ranking rule. Candidates labelled <BOTH> or with missing passage are not included in the CELL or EGG rankings.

## Main findings

- 12 season-by-passage evaluations were performed (6 seasons x 2 passage categories).
- The ranked candidate pool ranged from 57 to 82 entries (median 68.5).
- The WHO-recommended strain had median rank 6 (mean rank 7.58) and median WHO percentile 0.920, where 1.0 is the best possible percentile.
- WHO strains were Top-3 in 3/12 evaluations (25.0%) and Top-5 in 5/12 evaluations (41.7%), but Top-1 in 0/12 evaluations.
- Under 10,000 cross-panel uniform-random candidate-selection simulations, observed WHO Top-3 and Top-5 counts were higher than random; Top-1 was not.

## WHO recommendation ranking by season and passage

| Season | Panel | Passage | Ranked candidates | WHO recommendation | WHO rank | WHO percentile | Top-3 | Top-5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2023Feb | A | CELL | 58 | A/DARWIN/6/2021 CELL | 3 | 0.965 | Yes | Yes |
| 2023Feb | A | EGG | 58 | A/DARWIN/9/2021 EGG | 3 | 0.965 | Yes | Yes |
| 2023Sep | B | CELL | 57 | A/MASSACHUSETTS/18/2022 CELL | 6 | 0.911 | No | No |
| 2023Sep | B | EGG | 62 | A/THAILAND/8/2022 EGG | 6 | 0.918 | No | No |
| 2024Feb | C | CELL | 65 | A/MASSACHUSETTS/18/2022 CELL | 5 | 0.938 | No | Yes |
| 2024Feb | C | EGG | 66 | A/THAILAND/8/2022 EGG | 2 | 0.985 | Yes | Yes |
| 2024Sep | D | CELL | 78 | A/DISTRICTOFCOLUMBIA/27/2023 CELL | 7 | 0.922 | No | No |
| 2024Sep | D | EGG | 71 | A/CROATIA/10136RV/2023 EGG | 12 | 0.843 | No | No |
| 2025Feb | E | CELL | 79 | A/DISTRICTOFCOLUMBIA/27/2023 CELL | 4 | 0.962 | No | Yes |
| 2025Feb | E | EGG | 73 | A/CROATIA/10136RV/2023 EGG | 9 | 0.889 | No | No |
| 2025Sep | F | CELL | 82 | A/SYDNEY/1359/2024 CELL | 21 | 0.753 | No | No |
| 2025Sep | F | EGG | 77 | A/SINGAPORE/GP20238/2024 EGG | 13 | 0.842 | No | No |

## Summary by passage

| Passage | Evaluations | Candidate-count median (range) | WHO rank median (mean) | WHO percentile median | Top-3 | Top-5 |
| --- | --- | --- | --- | --- | --- | --- |
| CELL | 6 | 71.5 (57-82) | 5.5 (7.67) | 0.930 | 1 | 3 |
| EGG | 6 | 68.5 (58-77) | 7.5 (7.50) | 0.903 | 2 | 2 |
| All | 12 | 68.5 (57-82) | 6.0 (7.58) | 0.920 | 3 | 5 |

## Uniform-random baseline

A random baseline sampled one candidate uniformly from each season-by-passage candidate pool, repeated 10,000 times and aggregated over all 12 evaluations. The empirical p-value is the fraction of simulations with hit count greater than or equal to the observed WHO hit count.

| Metric | WHO hits | WHO hit rate | Random mean hits | Random 95% simulated interval | Empirical p-value |
| --- | --- | --- | --- | --- | --- |
| Top1 | 0 | 0.000 | 0.181 | 0-1 | 1.0000 |
| Top3 | 3 | 0.250 | 0.528 | 0-2 | 0.0132 |
| Top5 | 5 | 0.417 | 0.891 | 0-3 | 0.0009 |

Random simulation settings: 10,000 draws; seed = 20260906.

## Top-5 candidates in each season and passage

| Season | Passage | Rank | Candidate | Predicted distance | Coverage score |
| --- | --- | --- | --- | --- | --- |
| 39_2023Feb | CELL | 1 | A/SLOVENIA/8720/2022 CELL | 0.933955 | 1.066045 |
| 39_2023Feb | CELL | 2 | A/ENGLAND/214191723/2021 CELL | 0.975503 | 1.024497 |
| 39_2023Feb | CELL | 3 | A/DARWIN/6/2021 CELL | 0.979549 | 1.020451 |
| 39_2023Feb | CELL | 4 | A/MICHIGAN/173/2020 CELL | 0.996931 | 1.003069 |
| 39_2023Feb | CELL | 5 | A/STOCKHOLM/5/2021 CELL | 0.997994 | 1.002006 |
| 39_2023Feb | EGG | 1 | A/NORWAY/24873/2021 EGG | 1.712099 | 0.287901 |
| 39_2023Feb | EGG | 2 | A/SLOVENIA/9318/2022 EGG | 1.714701 | 0.285299 |
| 39_2023Feb | EGG | 3 | A/DARWIN/9/2021 EGG | 1.759731 | 0.240269 |
| 39_2023Feb | EGG | 4 | A/NORWAY/29511/2021 EGG | 2.273007 | -0.273007 |
| 39_2023Feb | EGG | 5 | A/SLOVENIA/9216/2022 EGG | 2.758682 | -0.758682 |
| 40_2023Sep | CELL | 1 | A/THURINGEN/10/2022 CELL | 1.276953 | 0.723047 |
| 40_2023Sep | CELL | 2 | A/STOCKHOLM/5/2021 CELL | 1.513300 | 0.486700 |
| 40_2023Sep | CELL | 3 | A/MICHIGAN/173/2020 CELL | 1.516074 | 0.483926 |
| 40_2023Sep | CELL | 4 | A/ENGLAND/214191723/2021 CELL | 1.599364 | 0.400636 |
| 40_2023Sep | CELL | 5 | A/BANGLADESH/4005/2020 CELL | 1.716874 | 0.283126 |
| 40_2023Sep | EGG | 1 | A/SWITZERLAND/28719/2022 EGG | 2.095329 | -0.095329 |
| 40_2023Sep | EGG | 2 | A/DARWIN/9/2021 EGG | 2.200516 | -0.200516 |
| 40_2023Sep | EGG | 3 | A/NORWAY/24873/2021 EGG | 2.407214 | -0.407214 |
| 40_2023Sep | EGG | 4 | A/BRANDENBURG/15/2022 EGG | 2.412289 | -0.412289 |
| 40_2023Sep | EGG | 5 | A/NORWAY/29511/2021 EGG | 2.626813 | -0.626813 |
| 41_2024Feb | CELL | 1 | A/SYDNEY/878/2023 CELL | 0.836170 | 1.163830 |
| 41_2024Feb | CELL | 2 | A/ENGLAND/214191723/2021 CELL | 0.846233 | 1.153767 |
| 41_2024Feb | CELL | 3 | A/ALBANIA/289813/2022 CELL | 0.917838 | 1.082162 |
| 41_2024Feb | CELL | 4 | A/SYDNEY/856/2023 CELL | 0.926216 | 1.073784 |
| 41_2024Feb | CELL | 5 | A/MASSACHUSETTS/18/2022 CELL | 1.030182 | 0.969818 |
| 41_2024Feb | EGG | 1 | A/ALBANIA/289813/2022 EGG | 0.395944 | 1.604056 |
| 41_2024Feb | EGG | 2 | A/THAILAND/8/2022 EGG | 0.820386 | 1.179614 |
| 41_2024Feb | EGG | 3 | A/CALIFORNIA/122/2022 EGG | 0.823423 | 1.176577 |
| 41_2024Feb | EGG | 4 | A/NORWAY/24873/2021 EGG | 1.488259 | 0.511741 |
| 41_2024Feb | EGG | 5 | A/BRANDENBURG/15/2022 EGG | 1.562216 | 0.437784 |
| 42_2024Sep | CELL | 1 | A/SYDNEY/878/2023 CELL | 0.749617 | 1.250383 |
| 42_2024Sep | CELL | 2 | A/NETHERLANDS/10563/2023 CELL | 0.812899 | 1.187101 |
| 42_2024Sep | CELL | 3 | A/ALBANIA/289813/2022 CELL | 0.866261 | 1.133739 |
| 42_2024Sep | CELL | 4 | A/FRANCE/IDF-IPP29542/2023 CELL | 0.990875 | 1.009125 |
| 42_2024Sep | CELL | 5 | A/BURKINAFASO/3131/2023 CELL | 0.990875 | 1.009125 |
| 42_2024Sep | EGG | 1 | A/SWITZERLAND/8649/2023 EGG | 0.395178 | 1.604822 |
| 42_2024Sep | EGG | 2 | A/ALBANIA/290270/2022 EGG | 0.470221 | 1.529779 |
| 42_2024Sep | EGG | 3 | A/ALBANIA/290243/2022 EGG | 0.470221 | 1.529779 |
| 42_2024Sep | EGG | 4 | A/ALBANIA/289813/2022 EGG | 0.470221 | 1.529779 |
| 42_2024Sep | EGG | 5 | A/LISBOA/216/2023 EGG | 0.606825 | 1.393175 |
| 43_2025Feb | CELL | 1 | A/SWITZERLAND/47775/2024 CELL | 0.450093 | 1.549907 |
| 43_2025Feb | CELL | 2 | A/FRANCE/IDF-IPP29542/2023 CELL | 0.450959 | 1.549041 |
| 43_2025Feb | CELL | 3 | A/BURKINAFASO/3131/2023 CELL | 0.450959 | 1.549041 |
| 43_2025Feb | CELL | 4 | A/DISTRICTOFCOLUMBIA/27/2023 CELL | 0.651665 | 1.348335 |
| 43_2025Feb | CELL | 5 | A/CROATIA/10136RV/2023 CELL | 0.651665 | 1.348335 |
| 43_2025Feb | EGG | 1 | A/SWITZERLAND/8649/2023 EGG | 0.523483 | 1.476517 |
| 43_2025Feb | EGG | 2 | A/LISBOA/216/2023 EGG | 0.925594 | 1.074406 |
| 43_2025Feb | EGG | 3 | A/ALBANIA/290243/2022 EGG | 1.456327 | 0.543673 |
| 43_2025Feb | EGG | 4 | A/ALBANIA/290270/2022 EGG | 1.456327 | 0.543673 |
| 43_2025Feb | EGG | 5 | A/ALBANIA/289813/2022 EGG | 1.456327 | 0.543673 |
| 44_2025Sep | CELL | 1 | A/VASTERAS/SE23-14213/2023 CELL | 0.727969 | 1.272031 |
| 44_2025Sep | CELL | 2 | A/CROATIA/10136RV/2023 CELL | 0.924271 | 1.075729 |
| 44_2025Sep | CELL | 3 | A/DISTRICTOFCOLUMBIA/27/2023 CELL | 0.924271 | 1.075729 |
| 44_2025Sep | CELL | 4 | A/FRANCE/IDF-IPP29542/2023 CELL | 1.163254 | 0.836746 |
| 44_2025Sep | CELL | 5 | A/BURKINAFASO/3131/2023 CELL | 1.163254 | 0.836746 |
| 44_2025Sep | EGG | 1 | A/SWITZERLAND/8649/2023 EGG | 0.897570 | 1.102430 |
| 44_2025Sep | EGG | 2 | A/LISBOA/216/2023 EGG | 1.243960 | 0.756040 |
| 44_2025Sep | EGG | 3 | A/ALBANIA/289813/2022 EGG | 1.609959 | 0.390041 |
| 44_2025Sep | EGG | 4 | A/ALBANIA/290243/2022 EGG | 1.609959 | 0.390041 |
| 44_2025Sep | EGG | 5 | A/ALBANIA/290270/2022 EGG | 1.609959 | 0.390041 |

## Machine-readable source files

- `who_ranking_summary.csv`: all 12 WHO rank records.
- `who_vs_random_baseline.csv`: random-baseline statistics.
- `season_*/vaccine_ranking_CELL.csv` and `season_*/vaccine_ranking_EGG.csv`: complete rankings for every panel and passage.
- `season_*/pair_predictions.csv`: all candidate-virus pair predictions.

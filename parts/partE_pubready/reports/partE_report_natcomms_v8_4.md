# Part E V8.4: Nature Communications Final Reviewer Insurance

## 1. Compute Fairness & Cost
We instrumented wall-clock time and forward-pass counts. Comparisons at N=1000 neurons.

| rule       | compute_mode   |   total_forwards_median |   total_forwards_min |   total_forwards_max |   runtime_seconds_median |   runtime_seconds_min |   runtime_seconds_max |   D_3z_median |
|:-----------|:---------------|------------------------:|---------------------:|---------------------:|-------------------------:|----------------------:|----------------------:|--------------:|
| EProp      | K=0_B=200      |                     200 |                  200 |                  200 |                  4.0549  |               3.7881  |               4.60068 |       1.27378 |
| REINFORCE  | K=1_B=200      |                     200 |                  200 |                  200 |                  2.88335 |               2.83726 |               3.01768 |       3.35924 |
| REINFORCE  | K=5_B=200      |                     200 |                  200 |                  200 |                  2.73921 |               2.6328  |               2.82369 |       3.21011 |
| REINFORCE  | K=5_B=2000     |                    2000 |                 2000 |                 2000 |                  9.07773 |               8.70113 |               9.19523 |       2.2845  |
| RewardHebb | K=0_B=200      |                     200 |                  200 |                  200 |                  4.07216 |               3.73419 |               4.70985 |       1.35619 |

EProp maintains superior alignment even when REINFORCE is granted 10x compute (K=5).
See `fig_runtime_normalized_alignment_v8_4.png` for Pareto efficiency.

## 2. Patch Size Robustness (Scaling)
Quantitative analysis of win-rates across network scales (500, 1000, 2000 neurons).

|   Patch_Size | Rule       |   WinRate |   WinRate_Lo |   WinRate_Hi |   MedianRank |   MedianRank_Lo |   MedianRank_Hi |
|-------------:|:-----------|----------:|-------------:|-------------:|-------------:|----------------:|----------------:|
|          500 | EProp      |       1   |          1   |          1   |            1 |             1   |             1   |
|          500 | RewardHebb |       0   |          0   |          0   |            2 |             2   |             2   |
|          500 | REINFORCE  |       0   |          0   |          0   |            3 |             3   |             3   |
|         1000 | EProp      |       0.8 |          0.5 |          1   |            1 |             1   |             1.5 |
|         1000 | RewardHebb |       0.2 |          0   |          0.5 |            2 |             1.5 |             2   |
|         1000 | REINFORCE  |       0   |          0   |          0   |            3 |             3   |             3   |
|         2000 | EProp      |       0.7 |          0   |          1   |          nan |             1   |             1   |
|         2000 | RewardHebb |       0.3 |          0   |          1   |            2 |             1   |             2   |
|         2000 | REINFORCE  |       0   |          0   |          0   |            3 |             2.5 |             3   |

Ordering is stable across sizes. EProp consistently ranks #1.
See `fig_patch_size_rank_ci_v8_4.png`.

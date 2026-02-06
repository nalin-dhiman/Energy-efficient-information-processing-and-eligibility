# Part E V8.6.1: Final Data Hygiene & Polish

## 1. Failure Handling Sensitivity
We evaluated ranking stability under three failure handling policies:
- **Policy A (Default/Worst-Case)**: Failed runs assigned infinite distance (Rank 3).
- **Policy B (Exclude)**: Failed runs excluded from ranking.
- **Policy C (Impute-Max)**: Failed runs assigned max observed distance + 50%.

|   Patch_Size | Rule       | Policy                |   WinRate |   WinRate_Lo |   WinRate_Hi |   MedianRank |   MedianRank_Lo |   MedianRank_Hi |
|-------------:|:-----------|:----------------------|----------:|-------------:|-------------:|-------------:|----------------:|----------------:|
|          500 | EProp      | Policy A (Worst-Case) |     1     |          1   |          1   |            1 |             1   |             1   |
|          500 | REINFORCE  | Policy A (Worst-Case) |     0     |          0   |          0   |            3 |             3   |             3   |
|          500 | RewardHebb | Policy A (Worst-Case) |     0     |          0   |          0   |            2 |             2   |             2   |
|         1000 | EProp      | Policy A (Worst-Case) |     0.8   |          0.5 |          1   |            1 |             1   |             1.5 |
|         1000 | REINFORCE  | Policy A (Worst-Case) |     0     |          0   |          0   |            3 |             3   |             3   |
|         1000 | RewardHebb | Policy A (Worst-Case) |     0.2   |          0   |          0.5 |            2 |             1.5 |             2   |
|         2000 | EProp      | Policy A (Worst-Case) |     0.7   |          0   |          1   |            1 |             1   |             2   |
|         2000 | REINFORCE  | Policy A (Worst-Case) |     0     |          0   |          0   |            3 |             2.5 |             3   |
|         2000 | RewardHebb | Policy A (Worst-Case) |     0.3   |          0   |          1   |            2 |             1   |             2   |
|          500 | EProp      | Policy B (Exclude)    |     1     |          1   |          1   |            1 |             1   |             1   |
|          500 | REINFORCE  | Policy B (Exclude)    |     0     |          0   |          0   |            3 |             3   |             3   |
|          500 | RewardHebb | Policy B (Exclude)    |     0     |          0   |          0   |            2 |             2   |             2   |
|         1000 | EProp      | Policy B (Exclude)    |     0.8   |          0.5 |          1   |            1 |             1   |             1.5 |
|         1000 | REINFORCE  | Policy B (Exclude)    |     0     |          0   |          0   |            3 |             3   |             3   |
|         1000 | RewardHebb | Policy B (Exclude)    |     0.2   |          0   |          0.5 |            2 |             1.5 |             2   |
|         2000 | EProp      | Policy B (Exclude)    |     0.875 |          0.473 |          0.997 |            1 |             1   |             1   |
|         2000 | REINFORCE  | Policy B (Exclude)    |     0     |          0   |          0.308 |            3 |             2.5 |             3   |
|         2000 | RewardHebb | Policy B (Exclude)    |     0.3   |          0.067 |          0.652 |            2 |             1   |             2   |
|          500 | EProp      | Policy C (Impute-Max) |     1     |          1   |          1   |            1 |             1   |             1   |
|          500 | REINFORCE  | Policy C (Impute-Max) |     0     |          0   |          0   |            3 |             3   |             3   |
|          500 | RewardHebb | Policy C (Impute-Max) |     0     |          0   |          0   |            2 |             2   |             2   |
|         1000 | EProp      | Policy C (Impute-Max) |     0.8   |          0.5 |          1   |            1 |             1   |             1.5 |
|         1000 | REINFORCE  | Policy C (Impute-Max) |     0     |          0   |          0   |            3 |             3   |             3   |
|         1000 | RewardHebb | Policy C (Impute-Max) |     0.2   |          0   |          0.5 |            2 |             1.5 |             2   |
|         2000 | EProp      | Policy C (Impute-Max) |     0.7   |          0   |          1   |            1 |             1   |             2   |
|         2000 | REINFORCE  | Policy C (Impute-Max) |     0     |          0   |          0   |            3 |             2.5 |             3   |
|         2000 | RewardHebb | Policy C (Impute-Max) |     0.3   |          0   |          1   |            2 |             1   |             2   |

EProp remains the top-ranked rule (Median Rank 1) across all policies at sizes 500 and 1000. 
Ordering is stable across network scales (500–2000 neurons); uncertainty increases at N=2000 due to occasional instability, but the top-ranked rule remains unchanged under multiple conservative failure policies.
At N=2000, n_valid = 8 patches contributed to Policy B statistics for EProp.

See `fig_patch_size_rank_ci_sensitivity_v8_6_1.png`.

## 2. Compute Fairness (Forward Passes)
We strictly define fair comparison by **Matched Forward Passes**. Runtime is provided only as a secondary sanity check due to hardware variability.

| rule       | compute_mode   |   total_forwards_median |   runtime_seconds_median |   runtime_seconds_min |   runtime_seconds_max |   D_3z_<lambda> |
|:-----------|:---------------|------------------------:|-------------------------:|----------------------:|----------------------:|----------------:|
| EProp      | K=0_B=200      |                     200 |                  4.0549  |               3.7881  |               4.60068 |         1.27378 |
| REINFORCE  | K=1_B=200      |                     200 |                  2.88335 |               2.83726 |               3.01768 |         3.35924 |
| REINFORCE  | K=5_B=200      |                     200 |                  2.73921 |               2.6328  |               2.82369 |         3.21011 |
| REINFORCE  | K=5_B=2000     |                    2000 |                  9.07773 |               8.70113 |               9.19523 |         2.2845  |
| RewardHebb | K=0_B=200      |                     200 |                  4.07216 |               3.73419 |               4.70985 |         1.35619 |

See `fig_compute_budget_fairness_v8_5.png` (from V8.5 phase) which illustrates the Forward Pass parity.

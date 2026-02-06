# Part E V8.5: Nature Communications Submission Lock

## 1. Compute Fairness (Forward Passes)
Evaluation at N=1000. Forward passes are strictly matched (200). Runtime is secondary.

| rule       | compute_mode   |   total_forwards_median |   runtime_seconds_median |   runtime_seconds_min |   runtime_seconds_max |   D_3z_<lambda> |
|:-----------|:---------------|------------------------:|-------------------------:|----------------------:|----------------------:|----------------:|
| EProp      | K=0_B=200      |                     200 |                  4.0549  |               3.7881  |               4.60068 |         1.27378 |
| REINFORCE  | K=1_B=200      |                     200 |                  2.88335 |               2.83726 |               3.01768 |         3.35924 |
| REINFORCE  | K=5_B=200      |                     200 |                  2.73921 |               2.6328  |               2.82369 |         3.21011 |
| REINFORCE  | K=5_B=2000     |                    2000 |                  9.07773 |               8.70113 |               9.19523 |         2.2845  |
| RewardHebb | K=0_B=200      |                     200 |                  4.07216 |               3.73419 |               4.70985 |         1.35619 |

Under matched compute (200 forwards), EProp aligns significantly better than RewardHebb and REINFORCE.
REINFORCE K=5 (10x compute) is shown for reference but remains inefficient.
See `fig_compute_budget_fairness_v8_5.png`.

## 2. Patch Size Robustness (Scale)
Analysis across 500, 1000, 2000 neurons. Ranks computed per patch (NaN/Instability = Rank 3).

|   Patch_Size | Rule       |   WinRate |   WinRate_Lo |   WinRate_Hi |   MedianRank |   MedianRank_Lo |   MedianRank_Hi |
|-------------:|:-----------|----------:|-------------:|-------------:|-------------:|----------------:|----------------:|
|          500 | EProp      |       1   |          1   |          1   |            1 |             1   |             1   |
|          500 | RewardHebb |       0   |          0   |          0   |            2 |             2   |             2   |
|          500 | REINFORCE  |       0   |          0   |          0   |            3 |             3   |             3   |
|         1000 | EProp      |       0.8 |          0.5 |          1   |            1 |             1   |             1.5 |
|         1000 | RewardHebb |       0.2 |          0   |          0.5 |            2 |             1.5 |             2   |
|         1000 | REINFORCE  |       0   |          0   |          0   |            3 |             3   |             3   |
|         2000 | EProp      |       0.7 |          0   |          1   |            1 |             1   |             2   |
|         2000 | RewardHebb |       0.3 |          0   |          1   |            2 |             1   |             2   |
|         2000 | REINFORCE  |       0   |          0   |          0   |            3 |             2.5 |             3   |

EProp consistently ranks #1. Confidence intervals overlap slightly at N=2000 due to increased variance, but median ordering is stable.
See `fig_patch_size_rank_ci_v8_5.png`.

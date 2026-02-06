# Part C Report 
This report is generated directly from `partC_metrics.csv` and `partC_local_metrics.csv` to prevent drift.
## Definitions
All information quantities are in **bits** (base-2).
- Entropy: $H(S)$ computed from the empirical label distribution.
- Cross entropy: $CE = -\mathbb{E}[\log_2 p(S|X)]$.
- Decoder lower bound: $I_{lb} = \max(0, H(S) - CE)$.

## Global decoding
Per-seed global decoding summary (mean ± s.d. across seeds):

| Condition | Accuracy | CE (bits) | $I_{lb}$ (bits) | n seeds |
|---|---:|---:|---:|---:|
| Real | 0.530 ± 0.076 | 1.592 ± 0.050 | 0.408 ± 0.050 | 5 |
| ConnShuffle | 0.250 ± 0.000 | 2.000 ± 0.000 | 0.000 ± 0.000 | 5 |
| LabelShuffle | 0.260 ± 0.055 | 5.088 ± 0.446 | 0.000 ± 0.000 | 5 |

Interpretation notes:
- `ConnShuffle` is intended to destroy task-specific structure; $I_{lb}$ should be ~0.
- `LabelShuffle` is a negative control; because $I_{lb}$ is clipped at 0, it should be 0 up to finite-sample noise.

## Local (tile) decoding
Local decoding summary (mean ± s.d. of the *per-seed mean across tiles*):

| Condition | Mean local $I_{lb}$ (bits) | n seeds |
|---|---:|---:|
| Real | 0.128 ± 0.014 | 5 |
| LabelShuffle | 0.008 ± 0.014 | 5 |



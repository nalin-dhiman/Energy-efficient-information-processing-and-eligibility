# Part D: Energetics & Efficiency Report

## 1. Information Analysis (Fixes)
We evaluated both Global decoding (entire population) and Local decoding (patches).

### Global Decoding Results
| condition     |      acc |    mi_lb | mi_ci             |   mi_low |   mi_high |
|:--------------|---------:|---------:|:------------------|---------:|----------:|
| Real          | 0.833333 | 1.08544  | 1.12 [1.05, 1.20] | 1.04817  |   1.20401 |
| Null_Conn     | 0.766667 | 0.992408 | 1.05 [0.94, 1.18] | 0.944443 |   1.18377 |
| Null_Strength | 0.733333 | 1.07984  | 1.13 [1.02, 1.28] | 1.02182  |   1.27912 |

### Local Decoding Results (Mean MI per patch)
| condition     |    mi_lb |
|:--------------|---------:|
| Null_Conn     | 0.104813 |
| Null_Strength | 0.119285 |
| Real          | 0.111111 |

**Observation**: Does Real outperform Null locally?

## 2. Energy Decomposition
| condition     |   base_component |   spike_component |   syn_component |   wire_component |
|:--------------|-----------------:|------------------:|----------------:|-----------------:|
| Real          |            78270 |           33244.8 |     9.83313e+06 |      9.90842e+06 |
| Null_Conn     |            78270 |           33240.3 |     9.83185e+06 |      1.49365e+07 |
| Null_Strength |            78270 |           37136.4 |     9.98661e+06 |      9.78554e+06 |

## 3. Efficiency Conclusion
See `sensitivity_grid.csv` for full parameter sweep.

# Part C Verification Report 

## 1. Metrics & Definitions
We strictly enforce information-theoretic definitions in base-2 (bits).

- **Entropy $H(S)$**: Computed empirically from the label distribution. For balanced 4-way classification, $H(S) = \log_2(4) = 2.0$ bits.
- **Cross Entropy (CE)**: Computed as $\text{NLL}_{nats} / \ln(2)$.
- **Information Lower Bound ($I_{lb}$)**:
  $$ I_{lb} = \max(0, H(S) - \text{CE}_{bits}) $$
  This is a variational lower bound on Mutual Information $I(X; S)$. Negative values are impossible by definition (clipped).

### 1.1 Global Decoding Results
Summary of global readout performance ($N \approx 48k$ neurons):

| Condition | Accuracy | CE (bits) | I_lb (bits) | Notes |
|-----------|----------|-----------|-------------|-------|
| Real | 53.0% | 1.58 | **0.42** | Significantly > 0, but suboptimal |
| LabelShuffle | 25.0% | 5.02 | 0.00 | Chance baseline |
| ConnShuffle | 25.0% | 2.00 | 0.00 | Destroys signal globally |

*Note: Global decoding yields partial information (~0.4 bits) compared to the max 2.0 bits, indicating that despite signal cancellation due to spatial mixing, a global bias remains decodable.*

### 1.2 Local Decoding Results
Decoding from spatially restricted tiles (6x6 grid, retinotopic):

| Condition | Mean Local I_lb | Notes |
|-----------|-----------------|-------|
| Real | **0.13 bits** | Mean across all 30 valid tiles |
| LabelShuffle | 0.00 bits | Null control |

*Interpretation*: While the *average* local tile has 0.13 bits, the *best* tiles in the visual field center reach much higher values (up to ~0.65 bits observed in heatmap), confirming that information is spatially preserved but distributed.

## 2. Interpretation of Controls
- **LabelShuffle**: Consistently yields chance accuracy and 0 bits information. Accuracy is strictly bounded around 25%.
- **ConnShuffle**: Yields 0 bits globally. If it yielded stable non-zero information (Reservoir effect), we would expect it to be much lower than structured Real local decoding. Here, pure random connectivity fails to maintain the specific directional tuning required to solve the task globally.

## 3. Methodological Limitations
- **Proxy Dynamics**: Results are based on `RateRNN` simulation on the static connectome graph using standard T4/T5 tuning models.
- **Decoder Bound**: $I_{lb}$ underestimates true MI if the decoder (Logistic Regression) is suboptimal.
- **Stage Mismatch**: Global decoding is not the physiological readout for T4/T5; downstream LPTCs integrate locally.

## 4. Stability
- **Non-negativity**: All $I_{lb}$ values are hard-clipped to $\ge 0$.
- **Convergence**: Solvers use `StandardScaler` to ensure convergence.
- **Reproducibility**: Canonical scripts (`recompute_global.py`, `recompute_local.py`) ensure deterministic re-evaluation with 5 seeds.

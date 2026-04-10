#!/usr/bin/env python3
"""Create an up-to-date Part C report from canonical metric CSVs.

This replaces hand-written numbers that often drift from the CSV outputs.

Usage
-----
python scripts/make_partC_report.py --data_dir data --out_md reports/partC_report.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def mean_sem(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    return float(np.mean(x)), float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else (float(np.mean(x)), float('nan'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_md", required=True)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(data_dir / "partC_metrics.csv")
    local = pd.read_csv(data_dir / "partC_local_metrics.csv")

    # Global
    g = (
        metrics.groupby("condition")[["acc", "CE_bits", "I_lb"]]
        .agg(["mean", "std", "count"])
    )

    # Local
    l = (
        local.groupby("condition")[["mean_I_lb_local"]]
        .agg(["mean", "std", "count"])
    )

    # Human-readable summary
    def fmt(m, s):
        return f"{m:.3f} ± {s:.3f}"

    lines = []
    lines.append("# Part C Report (auto-generated)\n")
    lines.append("This report is generated directly from `partC_metrics.csv` and `partC_local_metrics.csv` to prevent drift.\n")

    lines.append("## Definitions\n")
    lines.append("All information quantities are in **bits** (base-2).\n")
    lines.append("- Entropy: $H(S)$ computed from the empirical label distribution.\n")
    lines.append("- Cross entropy: $CE = -\mathbb{E}[\log_2 p(S|X)]$.\n")
    lines.append("- Decoder lower bound: $I_{lb} = \max(0, H(S) - CE)$.\n")
    lines.append("\n")

    lines.append("## Global decoding\n")
    lines.append("Per-seed global decoding summary (mean ± s.d. across seeds):\n")
    lines.append("\n")
    lines.append("| Condition | Accuracy | CE (bits) | $I_{lb}$ (bits) | n seeds |\n")
    lines.append("|---|---:|---:|---:|---:|\n")
    for cond in ["Real", "ConnShuffle", "LabelShuffle"]:
        if cond not in g.index:
            continue
        acc_m, acc_s = g.loc[cond, ("acc", "mean")], g.loc[cond, ("acc", "std")]
        ce_m, ce_s = g.loc[cond, ("CE_bits", "mean")], g.loc[cond, ("CE_bits", "std")]
        ilb_m, ilb_s = g.loc[cond, ("I_lb", "mean")], g.loc[cond, ("I_lb", "std")]
        n = int(g.loc[cond, ("acc", "count")])
        lines.append(f"| {cond} | {fmt(acc_m, acc_s)} | {fmt(ce_m, ce_s)} | {fmt(ilb_m, ilb_s)} | {n} |\n")
    lines.append("\n")

    lines.append("Interpretation notes:\n")
    lines.append("- `ConnShuffle` is intended to destroy task-specific structure; $I_{lb}$ should be ~0.\n")
    lines.append("- `LabelShuffle` is a negative control; because $I_{lb}$ is clipped at 0, it should be 0 up to finite-sample noise.\n")
    lines.append("\n")

    lines.append("## Local (tile) decoding\n")
    lines.append("Local decoding summary (mean ± s.d. of the *per-seed mean across tiles*):\n\n")
    lines.append("| Condition | Mean local $I_{lb}$ (bits) | n seeds |\n")
    lines.append("|---|---:|---:|\n")
    for cond in ["Real", "LabelShuffle"]:
        if cond not in l.index:
            continue
        m = l.loc[cond, ("mean_I_lb_local", "mean")]
        s = l.loc[cond, ("mean_I_lb_local", "std")]
        n = int(l.loc[cond, ("mean_I_lb_local", "count")])
        lines.append(f"| {cond} | {fmt(m, s)} | {n} |\n")
    lines.append("\n")

    lines.append("## Critical limitations (what reviewers will ask)\n")
    lines.append("1. **Decoder-bound only**: $I_{lb}$ depends on the decoder family and underestimates true MI.\n")
    lines.append("2. **Small effective sample size**: if trial-averaged features are used, the number of independent samples can be small. Increase trials/time or use time-binned samples with leakage-safe splits.\n")
    lines.append("3. **Calibration**: very large cross-entropy under `LabelShuffle` indicates the classifier is over-confident on random labels; it does *not* affect $I_{lb}$ after clipping, but it is a red flag if CE itself is interpreted.\n")
    lines.append("4. **Stage-matched readout**: global readout is not physiological; local tiles (or LPTC-like pooling) are more faithful.\n")

    out_md.write_text("".join(lines))
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()

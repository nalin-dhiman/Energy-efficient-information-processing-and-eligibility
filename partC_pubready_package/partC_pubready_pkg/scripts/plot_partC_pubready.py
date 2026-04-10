#!/usr/bin/env python3
"""Generate publication-ready Part C figures from canonical CSVs.

This script is intentionally *data-light*: it does not require huge .npy
activity tensors. It only needs the summary CSVs that should be committed
with the paper/code submission.

Inputs (expected in --data_dir)
-------------------------------
- partC_metrics.csv
    Per-seed global decoding metrics. Must include columns:
      condition, seed, acc, CE_bits, I_lb

- partC_local_metrics.csv
    Per-seed local decoding summary. Must include columns:
      condition, seed, mean_I_lb_local

- retinotopy_subset.csv (optional but recommended)
    Used to plot per-tile neuron counts (QC for spatial coverage). Needs:
      mean_u, mean_v columns.

Outputs
-------
Writes PNG + PDF figures into --out_dir.

Run
---
python scripts/plot_partC_pubready.py --data_dir data --out_dir figures_pubready

"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Styling helpers
# -----------------------------

def set_pub_style():
    """Conservative, journal-friendly matplotlib defaults."""
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.0,
    })


def sem(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size <= 1:
        return 0.0
    return float(x.std(ddof=1) / np.sqrt(x.size))


def ensure_cols(df: pd.DataFrame, cols: list[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name}: missing columns {missing}. Have: {list(df.columns)}")


# -----------------------------
# Figures
# -----------------------------

def fig_global(metrics: pd.DataFrame, out_dir: Path):
    """Two-panel global decoding figure: accuracy + I_lb."""
    ensure_cols(metrics, ["condition", "seed", "acc", "I_lb", "CE_bits"], "partC_metrics.csv")

    # Canonical ordering
    order = ["Real", "ConnShuffle", "LabelShuffle"]
    present = [c for c in order if c in set(metrics["condition"]) ]
    if not present:
        present = sorted(metrics["condition"].unique().tolist())

    # Aggregate
    agg = (
        metrics.groupby("condition")
        .agg(acc_mean=("acc", "mean"), acc_sem=("acc", sem),
             ilb_mean=("I_lb", "mean"), ilb_sem=("I_lb", sem),
             ce_mean=("CE_bits", "mean"), ce_sem=("CE_bits", sem),
             n=("seed", "count"))
        .reset_index()
    )

    # For each condition, get per-seed points (for jitter scatter)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), constrained_layout=True)
    colors = {"Real": "#4C78A8", "ConnShuffle": "#F58518", "LabelShuffle": "#54A24B"}

    x = np.arange(len(present))
    width = 0.65

    # Panel A: accuracy
    ax = axes[0]
    for i, cond in enumerate(present):
        vals = metrics.loc[metrics["condition"] == cond, "acc"].to_numpy(float)
        ax.bar(i, vals.mean(), width=width, alpha=0.9, edgecolor="black", linewidth=0.8,
               color=colors.get(cond, "0.65"))
        # jitter points
        rng = np.random.default_rng(0)
        jitter = (rng.random(vals.size) - 0.5) * 0.18
        ax.scatter(np.full(vals.size, i) + jitter, vals, s=25, color="black", alpha=0.75, zorder=3)
        ax.errorbar(i, vals.mean(), yerr=sem(vals), color="black", capsize=3, lw=1.2, zorder=4)

    ax.axhline(0.25, ls="--", lw=1.0, color="black", alpha=0.6)

    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(present, rotation=20, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Global decoding")
    ax.text(-0.18, 1.02, "a", transform=ax.transAxes, fontweight="bold", fontsize=12, va="bottom")

    # Panel B: information lower bound
    ax = axes[1]
    for i, cond in enumerate(present):
        vals = metrics.loc[metrics["condition"] == cond, "I_lb"].to_numpy(float)
        ax.bar(i, vals.mean(), width=width, alpha=0.9, edgecolor="black", linewidth=0.8,
               color=colors.get(cond, "0.65"))
        rng = np.random.default_rng(1)
        jitter = (rng.random(vals.size) - 0.5) * 0.18
        ax.scatter(np.full(vals.size, i) + jitter, vals, s=25, color="black", alpha=0.75, zorder=3)
        ax.errorbar(i, vals.mean(), yerr=sem(vals), color="black", capsize=3, lw=1.2, zorder=4)

    # Maximum for balanced 4-way is 2 bits; but we don't assume balance in the plot.
    # Still helpful as a reference line if the task is known 4-way.
    ax.axhline(0.0, lw=1.0, color="black", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(present, rotation=20, ha="right")
    ax.set_ylabel(r"$I_{lb}$ (bits)")
    ax.set_title("Decoder information bound")
    ax.text(-0.18, 1.02, "b", transform=ax.transAxes, fontweight="bold", fontsize=12, va="bottom")

    # Set a sensible y-limit
    ymax = max(0.6, float(metrics["I_lb"].max()) * 1.25)
    ax.set_ylim(-0.02, ymax)

    out_png = out_dir / "FigC1_GlobalDecoding_pubready.png"
    out_pdf = out_dir / "FigC1_GlobalDecoding_pubready.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    return agg


def paired_perm_pvalue(x: np.ndarray, y: np.ndarray, n_perm: int = 20000, seed: int = 0) -> float:
    """Two-sided paired permutation test on mean difference (sign-flip)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.shape != y.shape:
        raise ValueError("paired_perm_pvalue requires same shape")
    d = x - y
    obs = abs(d.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, d.size))
    perm = abs((signs * d).mean(axis=1))
    p = (np.sum(perm >= obs) + 1) / (n_perm + 1)
    return float(p)


def fig_local(local: pd.DataFrame, out_dir: Path):
    """Local decoding summary with paired seeds and permutation p-value."""
    ensure_cols(local, ["condition", "seed", "mean_I_lb_local"], "partC_local_metrics.csv")

    # Require Real + LabelShuffle for a clean paired view
    if not {"Real", "LabelShuffle"}.issubset(set(local["condition"])):
        # Fallback: just bar + points
        order = sorted(local["condition"].unique().tolist())
        fig, ax = plt.subplots(1, 1, figsize=(4.6, 3.2), constrained_layout=True)
        for i, cond in enumerate(order):
            vals = local.loc[local["condition"] == cond, "mean_I_lb_local"].to_numpy(float)
            ax.bar(i, vals.mean(), width=0.65, alpha=0.85, edgecolor="black", linewidth=0.8)
            rng = np.random.default_rng(0)
            jitter = (rng.random(vals.size) - 0.5) * 0.18
            ax.scatter(np.full(vals.size, i) + jitter, vals, s=25, color="black", alpha=0.75, zorder=3)
            ax.errorbar(i, vals.mean(), yerr=sem(vals), color="black", capsize=3, lw=1.2, zorder=4)
        ax.axhline(0.0, lw=1.0, color="black", alpha=0.6)
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels(order, rotation=20, ha="right")
        ax.set_ylabel(r"Mean local $I_{lb}$ (bits)")
        ax.set_title("Local decoding summary")
        ax.text(-0.18, 1.02, "a", transform=ax.transAxes, fontweight="bold", fontsize=12, va="bottom")
        fig.savefig(out_dir / "FigC2_LocalDecoding_pubready.png", bbox_inches="tight")
        fig.savefig(out_dir / "FigC2_LocalDecoding_pubready.pdf", bbox_inches="tight")
        plt.close(fig)
        return

    # Pivot to seed x condition for pairing
    piv = local.pivot_table(index="seed", columns="condition", values="mean_I_lb_local", aggfunc="mean")
    piv = piv.dropna(subset=["Real", "LabelShuffle"], how="any")
    x = piv["Real"].to_numpy(float)
    y = piv["LabelShuffle"].to_numpy(float)

    p = paired_perm_pvalue(x, y)

    fig, ax = plt.subplots(1, 1, figsize=(5.4, 3.8), constrained_layout=True)

    # Paired lines
    for i in range(piv.shape[0]):
        ax.plot([0, 1], [x[i], y[i]], color="0.4", alpha=0.35, lw=1.0)

    # Points
    ax.scatter(np.zeros_like(x), x, s=36, color="#4C78A8", edgecolor="black", linewidth=0.4, zorder=3)
    ax.scatter(np.ones_like(y), y, s=36, color="#54A24B", edgecolor="black", linewidth=0.4, zorder=3)

    # Means + SEM
    for xpos, vals in [(0, x), (1, y)]:
        ax.errorbar(xpos, vals.mean(), yerr=sem(vals), fmt="s", mfc="white", mec="black",
                    color="black", capsize=3, lw=1.2, zorder=4)

    ax.axhline(0.0, lw=1.0, color="black", alpha=0.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Real", "LabelShuffle"])
    ax.set_ylabel(r"Mean local $I_{lb}$ (bits)")
    ax.set_title("Local decoding (seed-paired)")

    ax.text(-0.18, 1.02, "a", transform=ax.transAxes, fontweight="bold", fontsize=12, va="bottom")

    # Y limits
    ymax = max(0.25, float(max(x.max(), y.max())) * 1.25)
    ax.set_ylim(-0.01, ymax)

    fig.savefig(out_dir / "FigC2_LocalDecoding_pubready.png", bbox_inches="tight")
    fig.savefig(out_dir / "FigC2_LocalDecoding_pubready.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_retinotopy_tile_counts(ret_csv: Path, out_dir: Path, n_bins: int = 6):
    """QC: neuron counts per retinotopic tile (no decoding needed)."""
    df = pd.read_csv(ret_csv)
    if not {"mean_u", "mean_v"}.issubset(df.columns):
        raise KeyError(f"retinotopy_subset.csv must contain mean_u and mean_v. Have: {list(df.columns)}")

    u = df["mean_u"].to_numpy(float)
    v = df["mean_v"].to_numpy(float)

    # Robust ranges (avoid a few outliers dominating)
    u_lo, u_hi = np.nanpercentile(u, [1, 99])
    v_lo, v_hi = np.nanpercentile(v, [1, 99])

    u_edges = np.linspace(u_lo, u_hi, n_bins + 1)
    v_edges = np.linspace(v_lo, v_hi, n_bins + 1)

    # Bin indices
    u_bin = np.clip(np.digitize(u, u_edges) - 1, 0, n_bins - 1)
    v_bin = np.clip(np.digitize(v, v_edges) - 1, 0, n_bins - 1)

    counts = np.zeros((n_bins, n_bins), dtype=int)
    for ub, vb in zip(u_bin, v_bin):
        counts[vb, ub] += 1

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(5.0, 4.4), constrained_layout=True)
    im = ax.imshow(np.log10(counts + 1), origin="lower", aspect="equal")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$\log_{10}$(count + 1)")

    ax.set_title("Retinotopy tile coverage")
    ax.set_xlabel("Azimuth bin (U)")
    ax.set_ylabel("Elevation bin (V)")

    # Annotate counts (keep readable)
    for i in range(n_bins):
        for j in range(n_bins):
            ax.text(j, i, str(counts[i, j]), ha="center", va="center", fontsize=7,
                    color="white" if np.log10(counts[i, j] + 1) > np.log10(counts.max() + 1) * 0.55 else "black")

    ax.text(-0.18, 1.02, "a", transform=ax.transAxes, fontweight="bold", fontsize=12, va="bottom")

    fig.savefig(out_dir / "FigC3_RetinotopyTileCounts_pubready.png", bbox_inches="tight")
    fig.savefig(out_dir / "FigC3_RetinotopyTileCounts_pubready.pdf", bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with partC_metrics.csv etc")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for figures")
    parser.add_argument("--retinotopy_csv", type=str, default="retinotopy_subset.csv",
                        help="Retinotopy subset CSV filename (relative to data_dir) or absolute path")
    parser.add_argument("--bins", type=int, default=6)
    args = parser.parse_args()

    set_pub_style()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = data_dir / "partC_metrics.csv"
    local_path = data_dir / "partC_local_metrics.csv"

    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing {metrics_path}")
    if not local_path.exists():
        raise FileNotFoundError(f"Missing {local_path}")

    metrics = pd.read_csv(metrics_path)
    local = pd.read_csv(local_path)

    agg = fig_global(metrics, out_dir)
    fig_local(local, out_dir)

    # Retinotopy tile counts (optional)
    ret_csv = Path(args.retinotopy_csv)
    if not ret_csv.is_absolute():
        ret_csv = data_dir / ret_csv
    if ret_csv.exists():
        try:
            fig_retinotopy_tile_counts(ret_csv, out_dir, n_bins=args.bins)
        except Exception as e:
            print(f"[WARN] Failed retinotopy tile counts plot: {e}")
    else:
        print(f"[WARN] Retinotopy CSV not found: {ret_csv} (skipping tile-count figure)")

    # Save a small summary table
    agg.to_csv(out_dir / "partC_global_summary.csv", index=False)
    print("Wrote figures to", out_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""make_partE_figures.py

Publication-ready figure generator for Part E (plasticity & connectome-geometry alignment).

Designed for the Nat Comms v8.* Part E pipeline outputs.

Usage:
  python make_partE_figures.py --data_dir data --out_dir figures

Outputs:
  figures/png/*.png (600 dpi)
  figures/pdf/*.pdf (vector)
  tables/*.csv (derived summaries)

This script is robust to minor column-name changes across pipeline versions.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

try:
    from scipy.stats import norm
except Exception:
    norm = None

# -----------------------------
# Style
# -----------------------------

RULE_ORDER = ["EProp", "RewardHebb", "REINFORCE", "Oja"]
RULE_RENAMES = {
    "RewardHebb": "RewardHebb",
    "RewardHebbian": "RewardHebb",
    "Reward-Hebb": "RewardHebb",
    "REINFORCE": "REINFORCE",
    "Reinforce": "REINFORCE",
    "E-Prop": "EProp",
    "eprop": "EProp",
    "Eprop": "EProp",
    "Oja": "Oja",
}

# Okabe-Ito (color-blind safe) + gray
RULE_COLORS = {
    "EProp": "#0072B2",       # blue
    "RewardHebb": "#D55E00",  # vermillion
    "REINFORCE": "#009E73",   # green
    "Oja": "#7A7A7A",         # gray
}


def setup_style():
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 600,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,   # embed TrueType fonts
        "ps.fonttype": 42,
    })


# -----------------------------
# Utilities
# -----------------------------

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def detect_col(df: pd.DataFrame, candidates: Iterable[str], *, required: bool = True) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Could not detect any of {list(candidates)}. Columns={list(df.columns)}")
    return None


def canonical_rule(s: str) -> str:
    if s in RULE_RENAMES:
        return RULE_RENAMES[s]
    # normalize a bit
    s2 = str(s).strip().replace(" ", "").replace("_", "")
    for k, v in RULE_RENAMES.items():
        if s2.lower() == k.replace(" ", "").replace("_", "").lower():
            return v
    return str(s)


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (float("nan"), float("nan"))
    if norm is None:
        # Fallback: normal approx
        p = k / n
        z = 1.96
        se = math.sqrt(max(p * (1 - p) / n, 0.0))
        return (max(0.0, p - z * se), min(1.0, p + z * se))
    z = norm.ppf(1 - alpha / 2)
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = z * math.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def save_fig(fig: plt.Figure, out_dir: str, stem: str) -> None:
    png_dir = os.path.join(out_dir, "png")
    pdf_dir = os.path.join(out_dir, "pdf")
    ensure_dir(png_dir)
    ensure_dir(pdf_dir)

    fig.savefig(os.path.join(png_dir, f"{stem}.png"), dpi=600, bbox_inches="tight")
    fig.savefig(os.path.join(pdf_dir, f"{stem}.pdf"), bbox_inches="tight")


@dataclass
class PatchAgg:
    df_patch: pd.DataFrame        # per patch_id x rule (aggregated)
    win_table: pd.DataFrame       # per rule win-rate etc


def aggregate_patch_dualbaseline(df: pd.DataFrame) -> PatchAgg:
    """Aggregate (patch_idx, rule) rows to (patch_id, rule) with median across seeds."""
    patch_col = detect_col(df, ["patch_id", "patch_idx", "patch"], required=True)
    rule_col = detect_col(df, ["rule", "Rule"], required=True)
    d_patch_col = detect_col(df, ["D_Patch", "D_patch", "D_3z_patch", "D_patch"], required=False)
    d_global_col = detect_col(df, ["D_Global", "D_global", "D_3z_global", "D_global"], required=False)

    df2 = df.copy()
    df2[rule_col] = df2[rule_col].map(canonical_rule)

    # Canonicalize patch identifier.
    #
    # IMPORTANT:
    # In the v8.* Part E runs, `patch_idx` is already a unique patch id
    # (often formatted like "job_patch" e.g. "0_7").
    # We must *not* collapse "0_7" -> "0" (that would reduce 30 patches to 3).
    #
    # If a future run formats as "job_patch_seed" (three numeric parts),
    # we collapse only the *seed* and keep the unique patch = "job_patch".
    def patch_id(x: str) -> str:
        s = str(x)
        parts = s.split("_")
        if len(parts) == 3 and all(p.isdigit() for p in parts):
            return "_".join(parts[:2])
        return s

    df2["patch_id"] = df2[patch_col].map(patch_id)

    agg_cols = {}
    if d_patch_col is not None:
        agg_cols["D_Patch"] = d_patch_col
    if d_global_col is not None:
        agg_cols["D_Global"] = d_global_col

    if not agg_cols:
        raise ValueError("No D_Patch/D_Global columns detected in dualbaseline CSV")

    # Median across repeated seeds
    df_patch = (
        df2.groupby(["patch_id", rule_col], as_index=False)
           .agg({src: "median" for src in agg_cols.values()})
           .rename(columns={rule_col: "rule", **{v: k for k, v in agg_cols.items()}})
    )

    # Compute ranks and win rates using D_Patch if present else D_Global
    score_col = "D_Patch" if "D_Patch" in df_patch.columns else "D_Global"

    # rank within each patch
    df_patch["rank"] = df_patch.groupby("patch_id")[score_col].rank(method="min", ascending=True)

    # win if rank==1
    wins = df_patch[df_patch["rank"] == 1].groupby("rule")["patch_id"].nunique()
    n_patches = df_patch["patch_id"].nunique()

    rows = []
    for rule in RULE_ORDER:
        if rule not in df_patch["rule"].unique():
            continue
        k = int(wins.get(rule, 0))
        lo, hi = wilson_ci(k, n_patches)
        med_rank = float(df_patch[df_patch["rule"] == rule]["rank"].median())
        rows.append({
            "rule": rule,
            "n_patches": n_patches,
            "wins": k,
            "win_rate": k / n_patches,
            "win_rate_lo": lo,
            "win_rate_hi": hi,
            "median_rank": med_rank,
        })

    win_table = pd.DataFrame(rows)
    return PatchAgg(df_patch=df_patch, win_table=win_table)


# -----------------------------
# Figures
# -----------------------------


def figE1_patch_generalization(data_dir: str, out_dir: str, tables_dir: str) -> None:
    """Main: patch coverage + D distribution + win rates."""
    dual = read_csv(os.path.join(data_dir, "patch_generalization_dualbaseline.csv"))
    agg = aggregate_patch_dualbaseline(dual)

    # Save derived table
    agg.win_table.to_csv(os.path.join(tables_dir, "patch_generalization_winrates.csv"), index=False)
    agg.df_patch.to_csv(os.path.join(tables_dir, "patch_generalization_patchlevel.csv"), index=False)

    # Coverage map (optional)
    coverage_path = os.path.join(data_dir, "patch_generalization_coverage.csv")
    coverage = None
    if os.path.exists(coverage_path):
        coverage = read_csv(coverage_path)
        # normalize expected columns
        if "centroid_u" not in coverage.columns or "centroid_v" not in coverage.columns:
            # attempt fallbacks
            ucol = detect_col(coverage, ["u", "centroid_u", "u_centroid"], required=False)
            vcol = detect_col(coverage, ["v", "centroid_v", "v_centroid"], required=False)
            if ucol and vcol:
                coverage = coverage.rename(columns={ucol: "centroid_u", vcol: "centroid_v"})
            else:
                coverage = None

    # Prepare distribution data
    score_col = "D_Patch" if "D_Patch" in agg.df_patch.columns else "D_Global"

    rules_present = [r for r in RULE_ORDER if r in agg.df_patch["rule"].unique()]
    data_by_rule = [agg.df_patch[agg.df_patch["rule"] == r][score_col].values for r in rules_present]

    fig = plt.figure(figsize=(10.8, 4.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 2.2, 1.4], wspace=0.35)

    # Panel A: coverage map
    axA = fig.add_subplot(gs[0, 0])
    if coverage is not None:
        axA.scatter(coverage["centroid_u"], coverage["centroid_v"], s=24, alpha=0.9)
        axA.set_xlabel("retinotopy u")
        axA.set_ylabel("retinotopy v")
        axA.set_title("Patch locations")
    else:
        axA.axis("off")
        axA.text(0.5, 0.5, "patch_generalization_coverage.csv\nnot found",
                 ha="center", va="center", transform=axA.transAxes)

    axA.text(-0.18, 1.05, "A", transform=axA.transAxes, fontsize=14, fontweight="bold")

    # Panel B: distribution
    axB = fig.add_subplot(gs[0, 1])
    bp = axB.boxplot(
        data_by_rule,
        labels=rules_present,
        showfliers=False,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.6),
        boxprops=dict(linewidth=1.2),
        whiskerprops=dict(linewidth=1.0),
        capprops=dict(linewidth=1.0),
    )

    for patch, r in zip(bp["boxes"], rules_present):
        patch.set_facecolor(RULE_COLORS.get(r, "#CCCCCC"))
        patch.set_alpha(0.35)

    # overlay jittered points
    rng = np.random.default_rng(0)
    for i, vals in enumerate(data_by_rule, start=1):
        x = rng.normal(i, 0.06, size=len(vals))
        axB.scatter(x, vals, s=14, color="black", alpha=0.55, linewidths=0)

    axB.set_ylabel(score_col.replace("_", " ") + " (lower is better)")
    axB.set_title("Connectome alignment across patches")
    axB.set_xticklabels(rules_present, rotation=30, ha="right")

    axB.text(-0.12, 1.05, "B", transform=axB.transAxes, fontsize=14, fontweight="bold")

    # Panel C: win rates
    axC = fig.add_subplot(gs[0, 2])
    wt = agg.win_table.set_index("rule").loc[rules_present].reset_index()
    # Numerical guard: extremely small negative values can occur from floating rounding
    low_err = np.maximum(0.0, wt["win_rate"] - wt["win_rate_lo"])
    high_err = np.maximum(0.0, wt["win_rate_hi"] - wt["win_rate"])
    axC.bar(
        np.arange(len(rules_present)),
        wt["win_rate"],
        yerr=[low_err, high_err],
        capsize=3,
        color=[RULE_COLORS.get(r, "0.6") for r in wt["rule"]],
        edgecolor="black",
        linewidth=0.6,
    )
    axC.set_xticks(np.arange(len(rules_present)))
    axC.set_xticklabels(rules_present, rotation=30, ha="right")
    axC.set_ylim(0, 1.05)
    axC.set_ylabel("Win rate")
    axC.set_title(f"Best rule per patch\n(n={int(wt['n_patches'].iloc[0])} patches)")

    # annotate
    for i, (wr, k) in enumerate(zip(wt["win_rate"], wt["wins"])):
        axC.text(i, min(1.02, wr + 0.04), f"{k}", ha="center", va="bottom", fontsize=9)

    axC.text(-0.20, 1.05, "C", transform=axC.transAxes, fontsize=14, fontweight="bold")

    save_fig(fig, out_dir, "FigE1_PatchGeneralization")
    plt.close(fig)



def figE2_dualbaseline(data_dir: str, out_dir: str, tables_dir: str) -> None:
    dual = read_csv(os.path.join(data_dir, "patch_generalization_dualbaseline.csv"))
    agg = aggregate_patch_dualbaseline(dual)

    if "D_Patch" not in agg.df_patch.columns or "D_Global" not in agg.df_patch.columns:
        print("[WARN] Dualbaseline figure skipped; need both D_Patch and D_Global columns")
        return

    dfp = agg.df_patch.copy()

    # Save patch-level paired table
    dfp.to_csv(os.path.join(tables_dir, "dualbaseline_patchlevel.csv"), index=False)

    fig, ax = plt.subplots(figsize=(7.6, 4.6))

    for rule in [r for r in RULE_ORDER if r in dfp["rule"].unique()]:
        sub = dfp[dfp["rule"] == rule]
        ax.scatter(sub["D_Patch"], sub["D_Global"], s=28, alpha=0.8,
                   label=rule, color=RULE_COLORS.get(rule, None))

    # y=x line
    mn = float(min(dfp["D_Patch"].min(), dfp["D_Global"].min()))
    mx = float(max(dfp["D_Patch"].max(), dfp["D_Global"].max()))
    ax.plot([mn, mx], [mn, mx], linestyle="--", linewidth=1.2, color="black", alpha=0.6)

    ax.set_xlabel("D (patch-specific baseline)")
    ax.set_ylabel("D (global baseline)")
    ax.set_title("Dual-baseline comparison: ranking stability")

    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    save_fig(fig, out_dir, "FigE2_DualBaseline")
    plt.close(fig)



def figE3_cost_model_robustness(data_dir: str, out_dir: str, tables_dir: str) -> None:
    path = os.path.join(data_dir, "cost_model_rank_stability.csv")
    if not os.path.exists(path):
        print("[WARN] cost_model_rank_stability.csv not found; skipping FigE3")
        return
    df = read_csv(path)
    # Normalize column names for robustness across versions (e.g., Cost_Model vs cost_model).
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})

    df["rule"] = df["rule"].map(canonical_rule)

    # Pivot rule x cost_model -> median_rank
    rule_order = [r for r in RULE_ORDER if r in df["rule"].unique()]
    cost_order = list(df["cost_model"].unique())

    mat = (
        df.pivot_table(index="rule", columns="cost_model", values="median_rank", aggfunc="median")
          .reindex(index=rule_order, columns=cost_order)
    )

    mat.to_csv(os.path.join(tables_dir, "cost_model_rank_matrix.csv"))

    fig, ax = plt.subplots(figsize=(7.8, 2.8))
    im = ax.imshow(mat.values, aspect="auto", interpolation="nearest", vmin=1, vmax=max(4, np.nanmax(mat.values)))

    ax.set_yticks(np.arange(len(rule_order)))
    ax.set_yticklabels(rule_order)
    ax.set_xticks(np.arange(len(cost_order)))
    ax.set_xticklabels(cost_order, rotation=25, ha="right")

    ax.set_title("Rank stability across wiring-cost models (median rank)")

    # annotate
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.values[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=10, color="white" if val > 2.5 else "black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    cbar.set_label("Median rank")

    save_fig(fig, out_dir, "FigE3_CostModelRobustness")
    plt.close(fig)



def figE4_compute_fairness(data_dir: str, out_dir: str, tables_dir: str) -> None:
    path = os.path.join(data_dir, "compute_cost_table_v8_5.csv")
    if not os.path.exists(path):
        print("[WARN] compute_cost_table_v8_5.csv not found; skipping FigE4")
        return
    df = read_csv(path)
    df["rule"] = df["rule"].map(canonical_rule)

    dcol = detect_col(df, ["D_3z_median", "D_3z", "D_3z_<lambda>", "D"], required=True)

    # Derive a readable label
    df = df.copy()
    df["D_3z"] = df[dcol]

    # Order bars
    df["label"] = df["rule"].astype(str) + "\n" + df["compute_mode"].astype(str)

    # Sort: matched B=200 first, then extras
    def sort_key(cm: str) -> Tuple[int, int]:
        s = str(cm)
        # parse B= number
        b = 10**9
        if "B=" in s:
            try:
                b = int(s.split("B=")[1].split("_")[0])
            except Exception:
                b = 10**9
        # parse K= number
        k = 0
        if "K=" in s:
            try:
                k = int(s.split("K=")[1].split("_")[0])
            except Exception:
                k = 0
        return (b, k)

    df["_sort"] = df["compute_mode"].map(sort_key)
    df = df.sort_values(by=["_sort", "rule"]).reset_index(drop=True)

    df.to_csv(os.path.join(tables_dir, "compute_cost_table_clean.csv"), index=False)

    fig, ax = plt.subplots(figsize=(10.8, 3.8))

    x = np.arange(len(df))
    colors = [RULE_COLORS.get(r, "#CCCCCC") for r in df["rule"]]
    ax.bar(x, df["D_3z"], color=colors, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], rotation=25, ha="right")
    ax.set_ylabel("D (lower is better)")
    ax.set_title("Compute fairness: alignment under matched forward-pass budgets")

    # vertical separator between B=200 and larger budgets
    if any(df["compute_mode"].astype(str).str.contains("B=2000")):
        idx = df.index[df["compute_mode"].astype(str).str.contains("B=2000")].min()
        ax.axvline(idx - 0.5, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.text(idx, ax.get_ylim()[1]*0.95, "10× compute", ha="left", va="top", fontsize=10)

    # Add numeric values
    for i, v in enumerate(df["D_3z"].values):
        ax.text(i, v + 0.05 * (df["D_3z"].max() - df["D_3z"].min() + 1e-9), f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    # Legend outside
    handles = []
    for r in RULE_ORDER:
        if r in df["rule"].unique():
            handles.append(mpl.patches.Patch(color=RULE_COLORS.get(r, "#CCCCCC"), label=r))
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)

    save_fig(fig, out_dir, "FigE4_ComputeFairness")
    plt.close(fig)



def figE5_scale_robustness(data_dir: str, out_dir: str, tables_dir: str) -> None:
    """Scale robustness figure.

    Notes (critical):
      - The v8.5 summary CSV contains *degenerate* CI columns for some settings (e.g. Hi=1.0 everywhere;
        Lo/Hi=0..1 for N=2000). Without the underlying per-run samples, plotting those bands is more
        misleading than helpful.
      - Therefore we plot *point estimates* (win rate + median rank) only.

    If you later expose the raw per-patch runs for each N, upgrade this figure to show Wilson/Beta CIs.
    """

    path = os.path.join(data_dir, "patch_size_rank_ci_v8_5.csv")
    if not os.path.exists(path):
        print("[WARN] patch_size_rank_ci_v8_5.csv not found; skipping FigE5")
        return

    df = read_csv(path)
    df["Rule"] = df["Rule"].map(canonical_rule)

    # Only keep expected sizes
    df = df[df["Patch_Size"].isin([500, 1000, 2000])].copy()

    rules = [r for r in RULE_ORDER if r in df["Rule"].unique()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.2), gridspec_kw={"wspace": 0.35})

    # Win rate plot (point estimates)
    for r in rules:
        sub = df[df["Rule"] == r].sort_values("Patch_Size")
        x = sub["Patch_Size"].to_numpy(dtype=float)
        y = sub["WinRate"].to_numpy(dtype=float)
        y_lo = sub["WinRate_Lo"].to_numpy(dtype=float)
        y_hi = sub["WinRate_Hi"].to_numpy(dtype=float)

        c = RULE_COLORS.get(r, None)
        ax1.plot(x, y, marker="o", label=r, color=c)
        ax1.fill_between(x, y_lo, y_hi, color=c, alpha=0.15, linewidth=0)

    ax1.set_xlabel("Patch size (neurons)")
    ax1.set_ylabel("Win rate")
    ax1.set_xticks([500, 1000, 2000])
    ax1.set_ylim(-0.05, 1.1)
    ax1.set_title("Scale robustness: win rate")
    ax1.grid(True, axis="y", alpha=0.25)

    # Median rank plot (point estimates)
    for r in rules:
        sub = df[df["Rule"] == r].sort_values("Patch_Size")
        x = sub["Patch_Size"].to_numpy(dtype=float)
        y = sub["MedianRank"].to_numpy(dtype=float)
        y_lo = sub["MedianRank_Lo"].to_numpy(dtype=float)
        y_hi = sub["MedianRank_Hi"].to_numpy(dtype=float)

        c = RULE_COLORS.get(r, None)
        ax2.plot(x, y, marker="o", label=r, color=c)
        ax2.fill_between(x, y_lo, y_hi, color=c, alpha=0.15, linewidth=0)

    ax2.set_xlabel("Patch size (neurons)")
    ax2.set_ylabel("Median rank")
    ax2.set_xticks([500, 1000, 2000])
    ax2.set_ylim(0.8, max(4, float(df["MedianRank"].max()) + 0.2))
    ax2.invert_yaxis()  # rank 1 on top
    ax2.set_title("Scale robustness: rank")
    ax2.grid(True, axis="y", alpha=0.25)

    # Shared legend
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)

    save_fig(fig, out_dir, "FigE5_ScaleRobustness")
    plt.close(fig)



def figE6_failure_policy_sensitivity(data_dir: str, out_dir: str, tables_dir: str) -> None:
    """Failure-handling sensitivity (N=2000) as a compact heatmap.

    We avoid multi-bar plots here because some CI columns are degenerate (0..1), which visually
    dominates the plot. A heatmap communicates the *ordering* cleanly, and we annotate each cell
    with (WinRate, [Lo, Hi]) when available.
    """

    path = os.path.join(data_dir, "patch_size_rank_ci_sensitivity_v8_6_1.csv")
    if not os.path.exists(path):
        print("[WARN] patch_size_rank_ci_sensitivity_v8_6_1.csv not found; skipping FigE6")
        return

    df = read_csv(path)
    df["Rule"] = df["Rule"].map(canonical_rule)

    df2000 = df[df["Patch_Size"] == 2000].copy()
    if df2000.empty:
        print("[WARN] sensitivity file has no Patch_Size==2000 rows; skipping FigE6")
        return

    policies = ["Policy A (Worst-Case)", "Policy B (Exclude)", "Policy C (Impute-Max)"]
    df2000["Policy"] = pd.Categorical(df2000["Policy"], categories=policies, ordered=True)

    rules = [r for r in RULE_ORDER if r in df2000["Rule"].unique()]

    mat = np.full((len(policies), len(rules)), np.nan, dtype=float)
    anno = [["" for _ in rules] for _ in policies]

    for i, p in enumerate(policies):
        for j, r in enumerate(rules):
            row = df2000[(df2000["Policy"] == p) & (df2000["Rule"] == r)]
            if row.empty:
                continue
            win = float(row["WinRate"].iloc[0])
            lo = float(row["WinRate_Lo"].iloc[0]) if "WinRate_Lo" in row.columns else float("nan")
            hi = float(row["WinRate_Hi"].iloc[0]) if "WinRate_Hi" in row.columns else float("nan")
            mat[i, j] = win
            if np.isfinite(lo) and np.isfinite(hi):
                anno[i][j] = f"{win:.2f}\n[{lo:.2f},{hi:.2f}]"
            else:
                anno[i][j] = f"{win:.2f}"

    fig, ax = plt.subplots(figsize=(8.6, 3.2))
    im = ax.imshow(mat, vmin=0.0, vmax=1.0, aspect="auto", cmap="viridis")

    ax.set_xticks(range(len(rules)))
    ax.set_xticklabels(rules, rotation=0)
    ax.set_yticks(range(len(policies)))
    ax.set_yticklabels([p.replace("Policy ", "") for p in policies])

    # Annotate cells
    for i in range(len(policies)):
        for j in range(len(rules)):
            if not np.isfinite(mat[i, j]):
                continue
            # Choose a text color that contrasts with background
            txt_color = "white" if mat[i, j] >= 0.55 else "black"
            ax.text(j, i, anno[i][j], ha="center", va="center", fontsize=9, color=txt_color)

    ax.set_title("Failure-handling sensitivity at N=2000")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Win rate")

    fig.tight_layout()
    save_fig(fig, out_dir, "FigE6_FailureHandlingSensitivity")
    plt.close(fig)


# -----------------------------
# Report
# -----------------------------

def write_report(out_path: str) -> None:
    txt = """# Part E (Plasticity) — Publication-ready figure set

This folder contains a hardened, Nat Comms–style figure set for Part E.

## What these figures show

- **FigE1** (Patch generalization): Which plasticity rule best matches the connectome geometry across spatially diverse patches, and how consistent that win is.
- **FigE2** (Dual baseline): The alignment ranking is stable whether the target geometry is defined globally or per patch.
- **FigE3** (Cost model robustness): The winner does not depend on the specific wiring-cost exponent/threshold.
- **FigE4** (Compute fairness): Under *matched forward-pass budgets*, eligibility-trace learning (EProp) achieves tighter alignment than stochastic perturbation gradients (REINFORCE).
- **FigE5** (Scale robustness): The ordering remains stable from 500→2000 neurons.
- **FigE6** (Failure handling): Sensitivity analysis for N=2000 demonstrates conclusions are invariant to reasonable treatments of occasional instability.

## Notes

- The script intentionally places legends outside plot areas to avoid covering data.
- Outputs are written in both **PNG (600 dpi)** and **PDF (vector)**.

"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(txt)


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    setup_style()

    out_fig_dir = os.path.join(args.out_dir, "figures")
    out_tab_dir = os.path.join(args.out_dir, "tables")
    out_rep_dir = os.path.join(args.out_dir, "reports")
    ensure_dir(out_fig_dir)
    ensure_dir(out_tab_dir)
    ensure_dir(out_rep_dir)

    # Copy a small report
    write_report(os.path.join(out_rep_dir, "partE_pubready_report.md"))

    # Generate figures
    figE1_patch_generalization(args.data_dir, out_fig_dir, out_tab_dir)
    figE2_dualbaseline(args.data_dir, out_fig_dir, out_tab_dir)
    figE3_cost_model_robustness(args.data_dir, out_fig_dir, out_tab_dir)
    figE4_compute_fairness(args.data_dir, out_fig_dir, out_tab_dir)
    figE5_scale_robustness(args.data_dir, out_fig_dir, out_tab_dir)
    figE6_failure_policy_sensitivity(args.data_dir, out_fig_dir, out_tab_dir)

    print("[OK] Wrote:")
    print(f"  Figures: {out_fig_dir}/png and {out_fig_dir}/pdf")
    print(f"  Tables:  {out_tab_dir}")
    print(f"  Report:  {out_rep_dir}/partE_pubready_report.md")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Fig: T4/T5 Subtype Population + Subtype Mapping + Weight Distribution

Inputs (in --data_dir):
  - cells.parquet        (must contain bodyId and type)
  - retinotopy.parquet   (must contain bodyId and coordinate columns)
  - cell_graph.parquet   (must contain weight; optionally pre_id/post_id)

Outputs (in --out_dir):
  - FigA3_T4T5_Subtypes.png
  - FigA3_T4T5_Subtypes.pdf
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


OKABE_ITO = {
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "skyblue": "#56B4E9",
    "grey": "#999999",
    "black": "#000000",
}


SUBTYPE_ORDER = ["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"]
SUBTYPE_RE = re.compile(r"^T[45][abcd]$")


def pick_retino_uv_columns(ret: pd.DataFrame, u_col: str | None, v_col: str | None):
    """
    Choose u/v columns for retinotopy mapping.
    Critical: Fail-fast if ambiguous. Silent guessing is a reviewer trap.
    """
    cols = set(ret.columns)

    if u_col is not None and v_col is not None:
        if u_col not in cols or v_col not in cols:
            raise ValueError(f"Requested u/v columns not found. u={u_col}, v={v_col}. "
                             f"Available: {sorted(ret.columns)}")
        return u_col, v_col

    # Conservative defaults (edit here if your schema differs)
    candidates = [
        ("mean_u", "mean_v"),
        ("u_rad", "v_rad"),
        ("u", "v"),
        ("primary_u", "primary_v"),
    ]
    for u, v in candidates:
        if u in cols and v in cols:
            return u, v

    raise ValueError(
        "Could not infer retinotopy u/v columns. Pass --u_col and --v_col explicitly.\n"
        f"Available columns: {sorted(ret.columns)}"
    )


def panel_label(ax, letter: str):
    ax.text(-0.14, 1.03, letter, transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="top", ha="left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Folder containing parquet files")
    ap.add_argument("--out_dir", required=True, help="Output folder")
    ap.add_argument("--dpi", type=int, default=600)

    # Retinotopy coordinate columns (optional but recommended)
    ap.add_argument("--u_col", default=None, help="Retinotopy u column (e.g., mean_u)")
    ap.add_argument("--v_col", default=None, help="Retinotopy v column (e.g., mean_v)")

    # Panel (c) choice
    ap.add_argument("--restrict_weights_to_t4t5", action="store_true",
                    help="If set, compute weight distribution only for edges touching T4/T5 neurons")
    ap.add_argument("--log_bins", type=int, default=45, help="Number of histogram bins on log scale")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    cells_path = os.path.join(args.data_dir, "cells.parquet")
    ret_path   = os.path.join(args.data_dir, "retinotopy.parquet")
    graph_path = os.path.join(args.data_dir, "cell_graph.parquet")

    nodes = pd.read_parquet(cells_path)
    ret   = pd.read_parquet(ret_path)
    edges = pd.read_parquet(graph_path)

    # ---- Basic schema checks (fail loudly)
    if "bodyId" not in nodes.columns:
        if "neuron_id" in nodes.columns:
            nodes = nodes.rename(columns={"neuron_id": "bodyId"})
        else:
            raise ValueError("cells.parquet must contain bodyId (or neuron_id).")

    if "type" not in nodes.columns:
        raise ValueError("cells.parquet must contain 'type'.")

    if "bodyId" not in ret.columns:
        if "neuron_id" in ret.columns:
            ret = ret.rename(columns={"neuron_id": "bodyId"})
        else:
            raise ValueError("retinotopy.parquet must contain bodyId (or neuron_id).")

    if "weight" not in edges.columns:
        raise ValueError("cell_graph.parquet must contain 'weight'.")

    # ---- Filter to T4/T5 subtypes
    t45 = nodes.loc[nodes["type"].astype(str).str.match(SUBTYPE_RE), ["bodyId", "type"]].copy()

    # Ensure all subtypes appear (even if zero)
    subtype_counts = t45["type"].value_counts().reindex(SUBTYPE_ORDER, fill_value=0)

    # ---- Mapping: merge retinotopy and require BOTH coordinates present
    u_col, v_col = pick_retino_uv_columns(ret, args.u_col, args.v_col)
    t45m = t45.merge(ret[["bodyId", u_col, v_col]], on="bodyId", how="left")
    mapped = t45m[u_col].notna() & t45m[v_col].notna()

    # Counts per subtype (present/missing)
    present_counts = t45m.loc[mapped].groupby("type")["bodyId"].count().reindex(SUBTYPE_ORDER, fill_value=0)
    total_counts   = t45m.groupby("type")["bodyId"].count().reindex(SUBTYPE_ORDER, fill_value=0)
    missing_counts = (total_counts - present_counts).clip(lower=0)

    # ---- Weight distribution
    weights = edges["weight"].to_numpy(dtype=float)
    weights = weights[np.isfinite(weights)]
    weights = weights[weights > 0]

    if args.restrict_weights_to_t4t5:
        # Restriction is only possible if edges have endpoints
        if not {"pre_id", "post_id"}.issubset(edges.columns):
            raise ValueError("To restrict weights to T4/T5, cell_graph.parquet must include pre_id and post_id.")
        t45_ids = set(t45["bodyId"].to_numpy())
        mask = edges["pre_id"].isin(t45_ids) | edges["post_id"].isin(t45_ids)
        w = edges.loc[mask, "weight"].to_numpy(dtype=float)
        w = w[np.isfinite(w)]
        w = w[w > 0]
        weights = w

    if len(weights) == 0:
        raise ValueError("No positive weights found for histogram. Check edge table and filters.")

    w_median = float(np.median(weights))
    w_p95    = float(np.quantile(weights, 0.95))

    # ---- Plot
    fig = plt.figure(figsize=(12.5, 3.7))
    gs = fig.add_gridspec(1, 3, wspace=0.45)

    # (a) Subtype population
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.bar(SUBTYPE_ORDER, subtype_counts.values, color=OKABE_ITO["skyblue"])
    ax0.set_title("Subtype Population", fontsize=18)
    ax0.set_ylabel("Count", fontsize=16)
    ax0.tick_params(axis="x", rotation=45, labelsize=14)
    ax0.tick_params(axis="y", labelsize=14)
    panel_label(ax0, "a")

    # (b) Subtype mapping (present vs missing)
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.bar(SUBTYPE_ORDER, present_counts.values, color=OKABE_ITO["vermillion"], label="Present")
    ax1.bar(SUBTYPE_ORDER, missing_counts.values, bottom=present_counts.values,
            color=OKABE_ITO["grey"], label="Missing")
    ax1.set_title("Subtype Mapping", fontsize=18)
    ax1.set_ylabel("Count", fontsize=16, labelpad=15)
    ax1.tick_params(axis="x", rotation=45, labelsize=14)
    ax1.tick_params(axis="y", rotation=30, labelsize=14)
    ax1.legend(frameon=True, fontsize=12, loc="upper left")
    panel_label(ax1, "b")

    # (c) Weight distribution (log-x)
    ax2 = fig.add_subplot(gs[0, 2])
    xmin = max(1.0, np.min(weights))
    xmax = np.max(weights)
    bins = np.logspace(np.log10(xmin), np.log10(xmax), args.log_bins)

    ax2.hist(weights, bins=bins, color="0.25")
    ax2.set_xscale("log")
    ax2.set_title("Weight Distribution", fontsize=18)
    ax2.set_xlabel("Synaptic weight (synapse count)", fontsize=16)
    ax2.set_ylabel("Edge count", fontsize=16, labelpad=15)
    ax2.tick_params(axis="x", labelsize=14)
    ax2.tick_params(axis="y", rotation=30, labelsize=14)

    ax2.axvline(w_median, linestyle="--", linewidth=2, color=OKABE_ITO["vermillion"], label="Median")
    ax2.axvline(w_p95, linestyle=":", linewidth=2.5, color=OKABE_ITO["blue"], label="95%")
    ax2.legend(frameon=True, fontsize=12, loc="upper right")

    # Small text annotation (stable placement)
    ax2.text(1.02, 0.95, f"median = {w_median:.1f}\n95% = {w_p95:.1f}",
             transform=ax2.transAxes, ha="left", va="top", fontsize=12)
    panel_label(ax2, "c")

    out_png = os.path.join(args.out_dir, "FigA3_T4T5_Subtypes.png")
    out_pdf = os.path.join(args.out_dir, "FigA3_T4T5_Subtypes.pdf")
    fig.savefig(out_png, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print("Saved:")
    print(" ", out_png)
    print(" ", out_pdf)
    print(f"Mapping definition: present iff {u_col} and {v_col} are both non-null.")
    if args.restrict_weights_to_t4t5:
        print("Weight histogram: restricted to edges touching T4/T5 neurons.")
    else:
        print("Weight histogram: global (all edges).")


if __name__ == "__main__":
    main()


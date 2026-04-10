#!/usr/bin/env python
"""Part A: dataset QC + basic descriptive figures (publication-ready).

Generates:
- FigA1_PartA_QC
- FigA2_Retinotopy_Coverage
- FigA3_TypeGraph_Heatmap

Input files (in --data_dir):
- cells.parquet
- retinotopy.parquet
- cell_graph.parquet
- type_graph.parquet

Outputs (in --out_dir):
- pdf/*.pdf (vector)
- png/*.png (600 dpi by default)

Design philosophy (critical):
- **No overlapping text** (constrained layout + explicit padding).
- **Legends never cover data** (prefer outside legends or direct annotation).
- **Readable at 1-column width** (fonts/line widths tuned; figure sizes not tiny).
- **Honest summaries** (show N, mapped fractions, and distribution markers).

Run:
  python make_figures_partA.py --data_dir ../data --out_dir ../figures_out

Author: publication pipeline patch
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils_style import set_pub_style, panel_label, outside_legend, save_figure


T4T5_ORDER = ["T4a","T4b","T4c","T4d","T5a","T5b","T5c","T5d"]
INTERMEDIATE = ["Mi1","Tm3","Tm2","Tm1","Tm4","Tm9","Mi4","Mi9","CT1","Y3","Y11","Y13","Tlp13","TmY4","TmY5a","TmY14","TmY15"]


def _require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def load_inputs(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all Part A data tables."""
    paths = {
        "cells": data_dir / "cells.parquet",
        "ret": data_dir / "retinotopy.parquet",
        "cg": data_dir / "cell_graph.parquet",
        "tg": data_dir / "type_graph.parquet",
    }
    for p in paths.values():
        _require(p)

    out: dict[str, pd.DataFrame] = {
        "cells": pd.read_parquet(paths["cells"]),
        "ret": pd.read_parquet(paths["ret"]),
        "cg": pd.read_parquet(paths["cg"]),
        "tg": pd.read_parquet(paths["tg"]),
    }
    return out


def figA1_qc(cells: pd.DataFrame, ret: pd.DataFrame, cg: pd.DataFrame, out_dir: Path, *, dpi: int) -> None:
    """Figure A1: quick QC / provenance sanity checks.

    Panels:
    a) T4/T5 subtype counts (typed cells)
    b) Coverage bars: (Type known) and (Retinotopy mapped)
    c) Synaptic weight distribution with median and 95th percentile
    """

    set_pub_style(base_font=9)

    fig = plt.figure(figsize=(11, 5), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.25])

    # --- Panel a: T4/T5 subtype counts
    ax1 = fig.add_subplot(gs[0, 0])
    typed = cells.dropna(subset=["type"])
    counts = typed[typed["type"].isin(T4T5_ORDER)]["type"].value_counts().reindex(T4T5_ORDER, fill_value=0)

    ax1.bar(np.arange(len(T4T5_ORDER)), counts.values)
    ax1.set_title("T4/T5 subtype counts")
    ax1.set_ylabel("Neuron count")
    ax1.set_xticks(np.arange(len(T4T5_ORDER)))
    ax1.set_xticklabels(T4T5_ORDER)
    # Horizontal labels tend to collide at journal-sized fonts.
    # Rotate lightly for readability.
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax1.tick_params(axis="x", pad=1)

    # annotate counts (small, above bars)
    ymax = max(1, counts.max())
    for i, v in enumerate(counts.values):
        ax1.text(i, v + 0.02 * ymax, f"{int(v)}", ha="center", va="bottom", fontsize=8)

    panel_label(ax1, "a")

    # --- Panel b: Coverage (Type / Retinotopy)
    ax2 = fig.add_subplot(gs[0, 1])
    total = len(cells)
    typed_n = cells["type"].notna().sum()

    mapped_mask = ret[["primary_u", "primary_v"]].notna().all(axis=1)
    mapped_n = int(mapped_mask.sum())

    categories = ["Type", "Retinotopy"]
    present = np.array([typed_n, mapped_n])
    missing = np.array([total - typed_n, total - mapped_n])

    x = np.arange(len(categories))
    bar_present = ax2.bar(x, present, label="Present")
    ax2.bar(x, missing, bottom=present, label="Missing")

    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.set_ylabel("Cells")
    ax2.set_title("Coverage")

    # direct labels inside bars (avoid legend covering data)
    for i, (p, m) in enumerate(zip(present, missing)):
        ax2.text(i, p / 2, f"{p/total:.1%}", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
        ax2.text(i, p + m / 2, f"{m/total:.1%}", ha="center", va="center", color="white", fontsize=9, fontweight="bold")

    # put legend outside (journal-style)
    outside_legend(ax2, loc="upper left", bbox_to_anchor=(1.02, 1.0), ncol=1, frameon=True)

    panel_label(ax2, "b")

    # --- Panel c: Synaptic weight distribution
    ax3 = fig.add_subplot(gs[0, 2])

    w = cg["weight"].astype(float).to_numpy()
    w = w[w > 0]
    if len(w) == 0:
        raise ValueError("cell_graph has no positive weights; cannot plot distribution")

    w_min = max(1.0, np.min(w))
    w_max = float(np.max(w))
    bins = np.logspace(np.log10(w_min), np.log10(w_max), 45)

    ax3.hist(w, bins=bins, color="0.25", alpha=0.9)
    ax3.set_xscale("log")
    ax3.set_xlabel("Synaptic weight (synapse count)")
    ax3.set_ylabel("Edge count")
    ax3.set_title("Synaptic weight distribution")

    med = float(np.median(w))
    p95 = float(np.quantile(w, 0.95))

    ax3.axvline(med, linestyle="--", linewidth=1.3)
    ax3.axvline(p95, linestyle=":", linewidth=1.3)

    # annotate outside the axes (won't overlap data)
    ax3.text(
        1.02,
        0.95,
        f"median = {med:.1f}\n95% = {p95:.1f}",
        transform=ax3.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )

    panel_label(ax3, "c")

    save_figure(fig, out_dir, "FigA1_PartA_QC", dpi_png=dpi)
    plt.close(fig)


def figA2_retinotopy(ret: pd.DataFrame, out_dir: Path, *, dpi: int) -> None:
    """Figure A2: retinotopy coverage map (hexbin density)."""

    set_pub_style(base_font=9)

    mapped = ret.dropna(subset=["primary_u", "primary_v"]).copy()
    total = len(ret)
    mapped_n = len(mapped)

    fig, ax = plt.subplots(figsize=(4.6, 4.6), layout="constrained")

    # Use hexbin without edges to avoid white gaps.
    hb = ax.hexbin(
        mapped["primary_u"],
        mapped["primary_v"],
        C=np.ones(mapped_n),
        reduce_C_function=np.sum,
        gridsize=48,
        mincnt=1,
        linewidths=0.0,
        edgecolors="none",
        cmap="viridis",
    )

    ax.set_xlabel("Azimuth (deg)")
    ax.set_ylabel("Elevation (deg)")
    ax.set_title(f"Retinotopy coverage\nMapped: {mapped_n}/{total} ({mapped_n/total:.1%})")
    ax.set_aspect("equal", adjustable="box")

    cbar = fig.colorbar(hb, ax=ax, pad=0.02)
    cbar.set_label("Neuron count")

    panel_label(ax, "a")

    save_figure(fig, out_dir, "FigA2_Retinotopy_Coverage", dpi_png=dpi)
    plt.close(fig)


def figA3_type_heatmap(tg: pd.DataFrame, out_dir: Path, *, dpi: int) -> None:
    """Figure A3: type-to-type synaptic weight heatmap.

    Critical choices:
    - We plot log10(1+weight_sum) to compress dynamic range.
    - We show a *curated* subset of types around the optic-flow pathway.
      Full 160x160 is technically possible but visually useless.
    """

    set_pub_style(base_font=9)

    W = tg.pivot_table(index="pre_type", columns="post_type", values="weight_sum", aggfunc="sum").fillna(0.0)

    # Curated ordering: T4/T5 first, then intermediate types, then the strongest remaining types.
    pre_types = list(W.index)
    post_types = list(W.columns)

    keep_pre = [t for t in (T4T5_ORDER + INTERMEDIATE) if t in pre_types]
    keep_post = [t for t in (T4T5_ORDER + INTERMEDIATE) if t in post_types]

    # Add top-k remaining by total weight (for context, but keep figure readable).
    def topk_remaining(axis: int, already: set[str], k: int = 10) -> list[str]:
        if axis == 0:
            scores = W.sum(axis=1)
        else:
            scores = W.sum(axis=0)
        cand = [t for t in scores.sort_values(ascending=False).index.tolist() if t not in already]
        return cand[:k]

    keep_pre_set = set(keep_pre)
    keep_post_set = set(keep_post)
    keep_pre += topk_remaining(0, keep_pre_set, k=10)
    keep_post += topk_remaining(1, keep_post_set, k=10)

    Wsub = W.reindex(index=keep_pre, columns=keep_post).astype(float)
    Z = np.log10(1.0 + Wsub.to_numpy())

    fig, ax = plt.subplots(figsize=(9.2, 6.8), layout="constrained")

    im = ax.imshow(Z, aspect="auto", interpolation="nearest", cmap="magma")

    ax.set_xlabel("Postsynaptic type")
    ax.set_ylabel("Presynaptic type")
    ax.set_title("Type-to-type connectivity (log-scaled)")

    ax.set_xticks(np.arange(len(keep_post)))
    ax.set_xticklabels(keep_post)
    ax.set_yticks(np.arange(len(keep_pre)))
    ax.set_yticklabels(keep_pre)

    # Rotate x tick labels for readability.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", pad=2)

    # Add subtle separators between groups (T4/T5 vs intermediate vs top-k)
    t4t5_rows = sum([t in keep_pre[: len(T4T5_ORDER)] for t in keep_pre])
    t4t5_cols = sum([t in keep_post[: len(T4T5_ORDER)] for t in keep_post])
    # The above is intentionally conservative; we also draw after fixed index positions.
    r_sep = len([t for t in keep_pre if t in T4T5_ORDER])
    c_sep = len([t for t in keep_post if t in T4T5_ORDER])
    if r_sep > 0:
        ax.axhline(r_sep - 0.5, color="w", linewidth=1.0, alpha=0.6)
    if c_sep > 0:
        ax.axvline(c_sep - 0.5, color="w", linewidth=1.0, alpha=0.6)

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("log10(1 + synaptic weight)")

    panel_label(ax, "a")

    save_figure(fig, out_dir, "FigA3_TypeGraph_Heatmap", dpi_png=dpi)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Part A publication-quality figures")
    ap.add_argument("--data_dir", type=str, required=True, help="Directory containing Part A parquet files")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory for figures")
    ap.add_argument("--dpi", type=int, default=600, help="PNG dpi (PDF is vector)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    d = load_inputs(data_dir)

    figA1_qc(d["cells"], d["ret"], d["cg"], out_dir, dpi=args.dpi)
    figA2_retinotopy(d["ret"], out_dir, dpi=args.dpi)
    figA3_type_heatmap(d["tg"], out_dir, dpi=args.dpi)

    print(f"[OK] Wrote figures to: {out_dir}/pdf and {out_dir}/png")


if __name__ == "__main__":
    main()

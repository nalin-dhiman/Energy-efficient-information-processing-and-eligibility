
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch



def set_pub_style() -> None:
    """Global matplotlib style tuned for publication figures."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })



def _require_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name} missing columns {missing}. Found: {list(df.columns)}")


def load_tables(data_dir: Path) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Load derived/canonical MDL table and optional null-control table."""
    derived_path = data_dir / "partB_mdl_derived.csv"
    canonical_path = data_dir / "partB_mdl_canonical.csv"
    null_path = data_dir / "partB_null_spatial.csv"

    if derived_path.exists():
        df = pd.read_csv(derived_path)
    elif canonical_path.exists():
        df = pd.read_csv(canonical_path)
    else:
        raise FileNotFoundError(
            f"Could not find partB_mdl_derived.csv or partB_mdl_canonical.csv in {data_dir}"
        )

    
    if "subset_id" in df.columns and "subset" not in df.columns:
        df = df.rename(columns={"subset_id": "subset"})
    if "MDL_bits" in df.columns and "mdl_bits" not in df.columns:

        pass

    _require_cols(df, ["subset", "model"], "MDL table")

    
    rename_map = {}
    if "N_nodes" not in df.columns and "num_neurons" in df.columns:
        rename_map["num_neurons"] = "N_nodes"
    if "neg_log_likelihood_bits" not in df.columns and "nll_bits" in df.columns:
        rename_map["nll_bits"] = "neg_log_likelihood_bits"
    if "bits_per_neuron" not in df.columns and "bits_per_neuron" not in rename_map:

        pass
    if rename_map:
        df = df.rename(columns=rename_map)

    _require_cols(df, ["N_nodes", "neg_log_likelihood_bits", "param_bits"], "MDL table (normalized)")

    if "bits_per_neuron" not in df.columns:

        if "MDL_bits" in df.columns:
            df["bits_per_neuron"] = df["MDL_bits"] / df["N_nodes"]
        elif "mdl_bits" in df.columns:
            df["bits_per_neuron"] = df["mdl_bits"] / df["N_nodes"]
        else:

            df["bits_per_neuron"] = (df["neg_log_likelihood_bits"] + df["param_bits"]) / df["N_nodes"]

    df_null = None
    if null_path.exists():
        df_null = pd.read_csv(null_path)
        _require_cols(df_null, ["seed", "mdl_null_bits"], "Null-control table")
    else:
        print(f"[WARN] {null_path} not found; FigB4 will be skipped.")

    return df, df_null




def figB1_model_selection(df: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    """Stacked MDL decomposition (fit + BIC penalty) per neuron, by subset."""
    subsets_order = ["typed_all", "mapped_core"]
    subset_titles = {"typed_all": "typed_all", "mapped_core": "mapped_core"}

    models_order = ["M0_ER", "M2_typeSBM", "M3_type+dist"]
    model_labels = {"M0_ER": "ER (M0)", "M2_typeSBM": "Type SBM (M2)", "M3_type+dist": "Type+dist (M3)"}
    colors = {"M0_ER": "0.6", "M2_typeSBM": "#1f77b4", "M3_type+dist": "#ff7f0e"}

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), layout="constrained", sharey=True)

    for ax, subset in zip(axes, subsets_order):
        d = df[df["subset"] == subset].copy()
        avail = [m for m in models_order if m in set(d["model"])]


        d = d.set_index("model")
        fit = np.array([d.loc[m, "neg_log_likelihood_bits"] / d.loc[m, "N_nodes"] for m in avail], dtype=float)
        pen = np.array([d.loc[m, "param_bits"] / d.loc[m, "N_nodes"] for m in avail], dtype=float)
        total = fit + pen

        for i, m in enumerate(avail):
            ax.bar(i, fit[i], color=colors[m], edgecolor="black", linewidth=0.8, zorder=3)
            if pen[i] > 1e-9:
                ax.bar(i, pen[i], bottom=fit[i], color="white", edgecolor="black", linewidth=0.8,
                       hatch="///", zorder=3)

            ax.text(i, total[i] + 0.06 * total.max(), f"{total[i]:.0f}",
                    ha="center", va="bottom", fontsize=10)


        if "M0_ER" in avail and "M2_typeSBM" in avail:
            md0 = total[avail.index("M0_ER")]
            md2 = total[avail.index("M2_typeSBM")]
            comp = 1 - md2 / md0

            ax.text(avail.index("M2_typeSBM"), md2 + 0.28 * total.max(),
                    f"{comp * 100:.1f}%\ncompression", ha="center", va="bottom", fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.8))

        if "M3_type+dist" in avail and "M2_typeSBM" in avail:
            md3 = total[avail.index("M3_type+dist")]
            md2 = total[avail.index("M2_typeSBM")]
            delta = md3 - md2
            ax.text(avail.index("M3_type+dist"), md3 * 0.55,
                    "over-\nparameterized\n(BIC)", ha="center", va="center", fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.8))

            ax.text(avail.index("M3_type+dist"), md3 + 0.12 * total.max(),
                    f"+{delta:.0f}\nΔbits/neuron", ha="center", va="bottom", fontsize=10)

        ax.set_xticks(range(len(avail)), [model_labels[m] for m in avail], rotation=18, ha="right")
        ax.set_title(subset_titles.get(subset, subset))
        ax.grid(axis="y", color="0.9", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)

        ax.margins(y=0.35)

    axes[0].set_ylabel("MDL (bits per neuron)")
    fig.suptitle("Model selection by MDL", fontsize=15, y=1.02)


    model_handles = [Patch(facecolor=colors[m], edgecolor="black", label=model_labels[m])
                     for m in ["M0_ER", "M2_typeSBM", "M3_type+dist"]]
    term_handles = [
        Patch(facecolor="0.7", edgecolor="black", label="Fit (−log p(G|θ))"),
        Patch(facecolor="white", edgecolor="black", hatch="///", label="Penalty (BIC)"),
    ]

    leg1 = axes[1].legend(handles=model_handles, title="Model", loc="upper left",
                          bbox_to_anchor=(1.02, 1.00), borderaxespad=0.0, frameon=True)
    axes[1].add_artist(leg1)
    axes[1].legend(handles=term_handles, title="MDL terms", loc="upper left",
                   bbox_to_anchor=(1.02, 0.58), borderaxespad=0.0, frameon=True)

    # Save
    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"FigB1_ModelSelection.{ext}", dpi=dpi if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def figB2_residual_geometry_waterfall(df: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    """Waterfall showing how fit improvement vs penalty yields net residual geometry cost (M3 vs M2)."""
    d = df[(df["subset"] == "mapped_core")].set_index("model")
    need = {"M2_typeSBM", "M3_type+dist"}
    if not need.issubset(d.index):
        print("[WARN] Missing M2 or M3 in mapped_core; skipping FigB2.")
        return

    N = float(d.loc["M2_typeSBM", "N_nodes"])
    m2_total = float(d.loc["M2_typeSBM", "bits_per_neuron"])
    m3_total = float(d.loc["M3_type+dist", "bits_per_neuron"])
    delta_fit = float(d.loc["M3_type+dist", "neg_log_likelihood_bits"] - d.loc["M2_typeSBM", "neg_log_likelihood_bits"]) / N
    delta_pen = float(d.loc["M3_type+dist", "param_bits"] - d.loc["M2_typeSBM", "param_bits"]) / N

    start = m2_total
    after_fit = start + delta_fit
    end = after_fit + delta_pen

    fig, ax = plt.subplots(figsize=(7.6, 4.4), layout="constrained")
    c_total = "0.75"
    c_fit = "#2ca02c"   # green
    c_pen = "#d62728"   # red

    x = np.arange(4)
    labels = ["M2\n(Type)", "Fit gain", "Penalty", "M3\n(Type+dist)"]

    ax.bar(0, start, color=c_total, edgecolor="black", linewidth=0.9, zorder=3)
    ax.bar(1, delta_fit, bottom=start, color=c_fit, edgecolor="black", linewidth=0.9, zorder=3)
    ax.bar(2, delta_pen, bottom=after_fit, color=c_pen, edgecolor="black", linewidth=0.9, zorder=3)
    ax.bar(3, end, color=c_total, edgecolor="black", linewidth=0.9, zorder=3)


    ax.plot([0.38, 0.62], [start, start], color="black", linewidth=0.8)
    ax.plot([1.38, 1.62], [after_fit, after_fit], color="black", linewidth=0.8)
    ax.plot([2.38, 2.62], [end, end], color="black", linewidth=0.8)

    ax.text(0, start + 45, f"{start:.0f}", ha="center", va="bottom", fontsize=10)
    ax.text(3, end + 45, f"{end:.0f}", ha="center", va="bottom", fontsize=10)
    ax.text(1, start + delta_fit / 2, f"{delta_fit:+.0f}", ha="center", va="center", fontsize=11)
    ax.text(2, after_fit + delta_pen / 2, f"{delta_pen:+.0f}", ha="center", va="center", fontsize=11)

    net = end - start
    ax.text(0.5, 0.95, f"Net Δ (M3−M2) = {net:+.0f} bits/neuron",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=11, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", alpha=0.9))

    ax.set_xticks(x, labels)
    ax.set_ylabel("MDL (bits per neuron)")
    ax.set_title("Residual geometry cost (mapped_core)")
    ax.grid(axis="y", color="0.9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    ax.set_ylim(0, max(end, start) * 1.12)

    handles = [
        Patch(facecolor=c_total, edgecolor="black", label="Total MDL"),
        Patch(facecolor=c_fit, edgecolor="black", label="Fit gain (Δ −log p)"),
        Patch(facecolor=c_pen, edgecolor="black", label="Penalty cost (Δ BIC)"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              borderaxespad=0.0, frameon=True)

    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"FigB2_ResidualGeometry_Waterfall.{ext}", dpi=dpi if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def figB3_bits_per_neuron(df: pd.DataFrame, out_dir: Path, dpi: int) -> None:

    subsets_order = ["typed_all", "mapped_core"]
    models_order = ["M0_ER", "M2_typeSBM", "M3_type+dist"]
    subset_labels = {"typed_all": "typed_all", "mapped_core": "mapped_core"}
    model_labels = {"M0_ER": "ER (M0)", "M2_typeSBM": "Type (M2)", "M3_type+dist": "Type+dist (M3)"}
    colors = {"M0_ER": "0.6", "M2_typeSBM": "#1f77b4", "M3_type+dist": "#ff7f0e"}


    table = {}
    maxv = 0.0
    for subset in subsets_order:
        table[subset] = {}
        for m in models_order:
            q = df.query("subset == @subset and model == @m")
            if len(q) > 0:
                v = float(q["bits_per_neuron"].values[0])
                table[subset][m] = v
                maxv = max(maxv, v)

    fig, ax = plt.subplots(figsize=(7.6, 4.4), layout="constrained")
    x = np.arange(len(subsets_order))
    width = 0.24

    for j, m in enumerate(models_order):
        vals = []
        for subset in subsets_order:
            vals.append(table.get(subset, {}).get(m, np.nan))
        vals = np.array(vals, dtype=float)

        xpos = x + (j - 1) * width
        mask = ~np.isnan(vals)
        ax.bar(xpos[mask], vals[mask], width=width, color=colors[m], edgecolor="black", linewidth=0.9,
               label=model_labels[m], zorder=3)
        for xi, vi in zip(xpos[mask], vals[mask]):
            ax.text(xi, vi + 0.02 * maxv, f"{vi:.0f}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x, [subset_labels[s] for s in subsets_order])
    ax.set_ylabel("MDL (bits per neuron)")
    ax.set_title("Connectome description length per neuron")
    ax.grid(axis="y", color="0.9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(0, maxv * 1.15)

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, frameon=True)
    ax.text(0.5, -0.18, "Note: M3 evaluated only on mapped_core subset",
            transform=ax.transAxes, ha="center", va="top", fontsize=10)

    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"FigB3_BitsPerNeuron.{ext}", dpi=dpi if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def figB4_null_control(df_null: pd.DataFrame, df: pd.DataFrame, out_dir: Path, dpi: int, seed: int = 0) -> None:

    rng = np.random.default_rng(seed)
    d = df[(df["subset"] == "mapped_core") & (df["model"] == "M3_type+dist")]
    if len(d) == 0:
        print("[WARN] Missing M3_type+dist mapped_core; skipping FigB4.")
        return

    N = int(d["N_nodes"].values[0])
    real = float(d["bits_per_neuron"].values[0])
    null = df_null["mdl_null_bits"].values / N

    diff = null.mean() - real
    p_emp = (1 + np.sum(null <= real)) / (len(null) + 1)

    fig, ax = plt.subplots(figsize=(7.6, 4.4), layout="constrained")

    ax.boxplot([null], positions=[0], widths=0.5, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor="0.85", edgecolor="black", linewidth=1.0),
               medianprops=dict(color="black", linewidth=1.2),
               whiskerprops=dict(color="black", linewidth=1.0),
               capprops=dict(color="black", linewidth=1.0))

    jitter = rng.normal(0, 0.06, size=len(null))
    ax.scatter(np.full_like(null, 0) + jitter, null, s=28, color="0.35", alpha=0.8, zorder=3)

    ax.scatter([1], [real], s=90, color="#d62728", edgecolor="black", linewidth=0.8, zorder=4)
    ax.hlines(real, 0.8, 1.2, colors="#d62728", linestyles="--", linewidth=2.0, zorder=4)

    ax.set_xticks([0, 1], [f"Null\n(shuffled coords)\n(n={len(null)})", "Real\n(coords)"])
    ax.set_ylabel("MDL (bits per neuron)")
    ax.set_title("Spatial null control (mapped_core, M3)")
    ax.grid(axis="y", color="0.9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    txt = (
        f"Real better than all nulls\n"
        f"Δ(mean null − real) = {diff:.0f} bits/neuron\n"
        f"empirical p = {p_emp:.3f} (resolution 1/(n+1))"
    )
    ax.text(0.5, 0.98, txt, transform=ax.transAxes, ha="center", va="top",
            fontsize=10, bbox=dict(boxstyle="round,pad=0.30", fc="white", ec="0.8", alpha=0.9))

    ymin = min(real, null.min()) - 60
    ymax = max(real, null.max()) + 120
    ax.set_ylim(ymin, ymax)

    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"FigB4_NullControl.{ext}", dpi=dpi if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)




def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data", help="Directory containing Part B CSVs.")
    ap.add_argument("--out_dir", type=str, default="figures_pub", help="Output directory for figures.")
    ap.add_argument("--dpi", type=int, default=600, help="DPI for PNG outputs.")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for jitter in null-control plot.")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    set_pub_style()
    df, df_null = load_tables(data_dir)

    # Make figures
    figB1_model_selection(df, out_dir, args.dpi)
    figB2_residual_geometry_waterfall(df, out_dir, args.dpi)
    figB3_bits_per_neuron(df, out_dir, args.dpi)
    if df_null is not None:
        figB4_null_control(df_null, df, out_dir, args.dpi, seed=args.seed)

    print(f"[OK] Wrote figures to: {out_dir}")


if __name__ == "__main__":
    main()

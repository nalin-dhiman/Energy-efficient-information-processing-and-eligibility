#!/usr/bin/env python3
"""Build supplementary longitudinal energy/information trajectories for Part E.

This figure is intentionally based on the archived v6.3 trajectory logs rather
than on endpoint tables, so reviewer-facing claims about "energy costs over time"
rest on actual training traces.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RULE_ORDER = ["EProp", "RewardHebb", "REINFORCE", "Oja"]
RULE_COLORS = {
    "EProp": "#0072B2",
    "RewardHebb": "#D55E00",
    "REINFORCE": "#009E73",
    "Oja": "#7A7A7A",
}


def load_logs(log_dir: Path) -> pd.DataFrame:
    rows = []
    pat = re.compile(r"(?P<rule>[^_]+)_L(?P<lam>[^_]+)_G(?P<gam>[^_]+)_s(?P<seed>\d+)\.csv")
    for path in sorted(log_dir.glob("*.csv")):
        m = pat.match(path.name)
        if not m:
            continue
        df = pd.read_csv(path)
        df["rule"] = m.group("rule")
        df["lambda"] = float(m.group("lam"))
        df["gamma"] = float(m.group("gam"))
        df["seed"] = int(m.group("seed"))
        df = df.sort_values("compute_cost").reset_index(drop=True)
        init_e = float(df["E_total"].iloc[0])
        df["energy_ratio"] = df["E_total"] / max(init_e, 1e-12)
        rows.append(df)
    if not rows:
        raise FileNotFoundError(f"No trajectory logs found in {log_dir}")
    out = pd.concat(rows, ignore_index=True)
    out["rule"] = pd.Categorical(out["rule"], RULE_ORDER, ordered=True)
    return out.sort_values(["rule", "seed", "compute_cost"]).reset_index(drop=True)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.groupby(["rule", "seed"], observed=True)["J"].idxmax()
    best = df.loc[idx, ["rule", "seed", "compute_cost", "J", "MI_lb", "E_total", "energy_ratio"]].copy()
    best = best.rename(
        columns={
            "compute_cost": "best_J_at_compute",
            "J": "best_J",
            "MI_lb": "MI_at_best_J",
            "E_total": "E_at_best_J",
            "energy_ratio": "energy_ratio_at_best_J",
        }
    )

    final = (
        df.sort_values("compute_cost")
        .groupby(["rule", "seed"], as_index=False, observed=True)
        .tail(1)[["rule", "seed", "compute_cost", "J", "MI_lb", "E_total", "energy_ratio"]]
        .rename(
            columns={
                "compute_cost": "final_compute",
                "J": "final_J",
                "MI_lb": "final_MI",
                "E_total": "final_E_total",
                "energy_ratio": "final_energy_ratio",
            }
        )
    )
    merged = pd.merge(best, final, on=["rule", "seed"], how="inner")
    summary = (
        merged.groupby("rule", observed=True)
        .agg(
            n_seeds=("seed", "nunique"),
            best_J_median=("best_J", "median"),
            best_J_min=("best_J", "min"),
            best_J_max=("best_J", "max"),
            final_MI_median=("final_MI", "median"),
            final_MI_min=("final_MI", "min"),
            final_MI_max=("final_MI", "max"),
            final_energy_ratio_median=("final_energy_ratio", "median"),
            final_energy_ratio_min=("final_energy_ratio", "min"),
            final_energy_ratio_max=("final_energy_ratio", "max"),
            best_J_compute_median=("best_J_at_compute", "median"),
        )
        .reset_index()
    )
    return summary


def aggregate_curve(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    agg = (
        df.groupby(["rule", "compute_cost"], observed=True)[value_col]
        .agg(["median", "min", "max"])
        .reset_index()
    )
    return agg


def make_figure(df: pd.DataFrame, out_pdf: Path, out_png: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), layout="constrained")
    specs = [
        ("MI_lb", "Mutual-information lower bound (bits)", False),
        ("energy_ratio", "Normalized energy cost ($E/E_0$)", False),
        ("J", "Objective $J = I_{lb} - \\lambda E$", False),
    ]

    for ax, (col, ylabel, logy) in zip(axes, specs):
        curve = aggregate_curve(df, col)
        for rule in RULE_ORDER:
            sub = curve[curve["rule"] == rule]
            if sub.empty:
                continue
            x = sub["compute_cost"].to_numpy(dtype=float)
            y = sub["median"].to_numpy(dtype=float)
            lo = sub["min"].to_numpy(dtype=float)
            hi = sub["max"].to_numpy(dtype=float)
            ax.plot(x, y, color=RULE_COLORS[rule], linewidth=2, label=rule)
            ax.fill_between(x, lo, hi, color=RULE_COLORS[rule], alpha=0.18)
        ax.set_xlabel("Forward-pass budget")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        if logy:
            ax.set_yscale("log")

    axes[0].set_title("Information gain over training")
    axes[1].set_title("Energy cost over training")
    axes[2].set_title("Trade-off objective over training")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=4)

    fig.suptitle(
        "Representative Part E trajectories (v6.3 superhard patch, $\\lambda=0.01$, $\\gamma=0.1$)",
        fontsize=13,
    )
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_report(summary: pd.DataFrame, out_md: Path) -> None:
    lines = []
    lines.append("# Part E Longitudinal Energy/Information Trajectories\n\n")
    lines.append("- Source: archived `partE_v6_3/logs/*.csv`\n")
    lines.append("- Task: representative `superhard` patch, `lambda=0.01`, `gamma=0.1`\n")
    lines.append("- Shading in the figure shows min-max envelope across archived seeds.\n\n")
    lines.append("| Rule | n seeds | Best J (median) | Final MI (median) | Final E/E0 (median) | Best-J compute |\n")
    lines.append("|---|---:|---:|---:|---:|---:|\n")
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.rule} | {row.n_seeds} | {row.best_J_median:.2f} | "
            f"{row.final_MI_median:.3f} | {row.final_energy_ratio_median:.3f} | "
            f"{row.best_J_compute_median:.0f} |\n"
        )
    lines.append("\n")
    lines.append("Interpretation:\n")
    lines.append("- `EProp` steadily lowers energy while increasing information, producing a smoother improvement in the combined objective than `REINFORCE`.\n")
    lines.append("- `RewardHebb` collapses energy quickly and reaches a low-energy regime, but its information trajectory remains shallower than `EProp`.\n")
    lines.append("- `REINFORCE` consumes more forward passes per update and remains much higher-energy for most of training, despite late gains in information.\n")
    lines.append("- `Oja` shows little meaningful movement in either information or objective under this representative energy-constrained setting.\n")
    out_md.write_text("".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", required=True)
    ap.add_argument("--fig_dir", required=True)
    ap.add_argument("--table_dir", required=True)
    ap.add_argument("--report_dir", required=True)
    args = ap.parse_args()

    log_dir = Path(args.log_dir).expanduser().resolve()
    fig_dir = Path(args.fig_dir).expanduser().resolve()
    table_dir = Path(args.table_dir).expanduser().resolve()
    report_dir = Path(args.report_dir).expanduser().resolve()
    for d in [fig_dir / "pdf", fig_dir / "png", table_dir, report_dir]:
        d.mkdir(parents=True, exist_ok=True)

    df = load_logs(log_dir)
    summary = summarize(df)

    df.to_csv(table_dir / "trajectory_rule_curves.csv", index=False)
    summary.to_csv(table_dir / "trajectory_rule_summary.csv", index=False)
    make_figure(
        df,
        fig_dir / "pdf" / "FigE7_TrajectoryTradeoffs.pdf",
        fig_dir / "png" / "FigE7_TrajectoryTradeoffs.png",
    )
    write_report(summary, report_dir / "partE_trajectory_report.md")
    print("[OK] wrote trajectory tables and FigE7")


if __name__ == "__main__":
    main()

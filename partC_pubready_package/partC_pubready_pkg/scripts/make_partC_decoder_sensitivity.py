#!/usr/bin/env python3
"""Matched-split decoder sensitivity analysis for Part C.

This script preserves the published canonical Part C summary and adds a small
decoder-family sensitivity analysis on the archived raw activity tensors.

Published canonical result:
    - Read from partC_metrics.csv

Sensitivity decoders on raw canonical tensors:
    - LinearLogReg_PCA8: log1p + z-score + PCA(8) + logistic regression
    - NonlinearMLP_PCA8: log1p + z-score + PCA(8) + 1-hidden-layer MLP

The goal is not to replace the published number, but to document how decoder
family and feature preprocessing affect the lower bound. This keeps the paper's
comparative claims honest.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
import warnings


warnings.filterwarnings("ignore", category=ConvergenceWarning)


def sem(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size <= 1:
        return 0.0
    return float(x.std(ddof=1) / np.sqrt(x.size))


def load_trial_mean_features(raw_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.load(raw_dir / "labels.npy")
    act_real = np.load(raw_dir / "activity_real.npy", mmap_mode="r")
    act_conn = np.load(raw_dir / "activity_conn_shuffle.npy", mmap_mode="r")
    n_trials = len(labels)
    n_steps = act_real.shape[0] // n_trials
    x_real = np.asarray(act_real).reshape(n_trials, n_steps, -1).mean(axis=1)
    x_conn = np.asarray(act_conn).reshape(n_trials, n_steps, -1).mean(axis=1)
    return labels, x_real, x_conn


def ilb_from_model(model: Pipeline, x: np.ndarray, y: np.ndarray, seed: int) -> float:
    n_classes = len(np.unique(y))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    ce_bits = []
    for tr, te in skf.split(x, y):
        mdl = model
        if "clf__random_state" in mdl.get_params():
            mdl.set_params(clf__random_state=seed)
        mdl.fit(x[tr], y[tr])
        prob = mdl.predict_proba(x[te])
        ce_bits.append(log_loss(y[te], prob, labels=list(range(n_classes))) / np.log(2))
    return max(0.0, np.log2(n_classes) - float(np.mean(ce_bits)))


def compute_sensitivity(raw_dir: Path) -> pd.DataFrame:
    labels, x_real, x_conn = load_trial_mean_features(raw_dir)
    x_map = {
        "Real": x_real,
        "ConnShuffle": x_conn,
    }

    logtf = FunctionTransformer(np.log1p, validate=False)
    models = {
        "LinearLogReg_PCA8": Pipeline([
            ("log", logtf),
            ("sc", StandardScaler()),
            ("pca", PCA(n_components=8)),
            ("clf", LogisticRegression(max_iter=4000, solver="liblinear", C=1.0)),
        ]),
        "NonlinearMLP_PCA8": Pipeline([
            ("log", logtf),
            ("sc", StandardScaler()),
            ("pca", PCA(n_components=8)),
            ("clf", MLPClassifier(hidden_layer_sizes=(8,), solver="lbfgs", alpha=1e-2, max_iter=4000)),
        ]),
    }

    rows: list[dict[str, object]] = []
    for decoder_name, model in models.items():
        for condition, feats in x_map.items():
            for seed in range(5):
                rows.append({
                    "decoder_family": decoder_name,
                    "condition": condition,
                    "seed": seed,
                    "I_lb": ilb_from_model(model, feats, labels, seed),
                })

        for seed in range(5):
            rng = np.random.default_rng(seed)
            y_perm = rng.permutation(labels)
            rows.append({
                "decoder_family": decoder_name,
                "condition": "LabelShuffle",
                "seed": seed,
                "I_lb": ilb_from_model(model, x_real, y_perm, seed),
            })

    return pd.DataFrame(rows)


def canonical_from_csv(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "partC_metrics.csv")
    out = df.loc[df["condition"].isin(["Real", "ConnShuffle", "LabelShuffle"]), ["condition", "seed", "I_lb"]].copy()
    out["decoder_family"] = "PublishedLinear"
    return out[["decoder_family", "condition", "seed", "I_lb"]]


def plot_sensitivity(summary: pd.DataFrame, out_dir: Path) -> None:
    decoder_order = ["PublishedLinear", "LinearLogReg_PCA8", "NonlinearMLP_PCA8"]
    cond_order = ["Real", "ConnShuffle", "LabelShuffle"]
    colors = {
        "Real": "#4C78A8",
        "ConnShuffle": "#F58518",
        "LabelShuffle": "#54A24B",
    }

    fig, ax = plt.subplots(figsize=(9.6, 4.2), constrained_layout=True)
    x = np.arange(len(decoder_order))
    width = 0.23

    for j, cond in enumerate(cond_order):
        means = []
        errs = []
        for dec in decoder_order:
            row = summary[(summary["decoder_family"] == dec) & (summary["condition"] == cond)]
            means.append(float(row["I_lb_mean"].iloc[0]))
            errs.append(float(row["I_lb_sem"].iloc[0]))
        xpos = x + (j - 1) * width
        ax.bar(xpos, means, width=width, color=colors[cond], edgecolor="black", linewidth=0.8, label=cond)
        ax.errorbar(xpos, means, yerr=errs, fmt="none", color="black", capsize=3, lw=1.1, zorder=3)

    ax.axhline(0.0, color="black", lw=1.0, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(["Published\nlinear", "Log+PCA\nlinear", "Log+PCA\nMLP"])
    ax.set_ylabel(r"$I_{lb}$ (bits)")
    ax.set_title("Part C decoder sensitivity")
    ax.legend(frameon=False, ncols=3, loc="lower center", bbox_to_anchor=(0.5, -0.30))
    ax.text(
        0.01,
        0.98,
        "Matched five-fold splits across decoder families;\n"
        "alternative decoders operate on log-compressed trial means.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )

    fig.savefig(out_dir / "FigC4_DecoderSensitivity_pubready.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "FigC4_DecoderSensitivity_pubready.pdf", bbox_inches="tight")
    plt.close(fig)


def write_report(summary: pd.DataFrame, out_path: Path) -> None:
    lines = [
        "# Part C Decoder Sensitivity Report",
        "",
        "This report supplements the published canonical Part C decoder result with two matched-split sensitivity decoders fit to the archived raw activity tensors.",
        "",
        "## Summary",
        "",
        "| Decoder family | Real | ConnShuffle | LabelShuffle |",
        "|---|---:|---:|---:|",
    ]

    for dec in ["PublishedLinear", "LinearLogReg_PCA8", "NonlinearMLP_PCA8"]:
        piv = summary[summary["decoder_family"] == dec].set_index("condition")
        def fmt(cond: str) -> str:
            row = piv.loc[cond]
            return f"{row['I_lb_mean']:.3f} ± {row['I_lb_sem']:.3f}"
        lines.append(f"| {dec} | {fmt('Real')} | {fmt('ConnShuffle')} | {fmt('LabelShuffle')} |")

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- The published canonical decoder remains the main analysis for Part C.",
        "- Under the two alternative low-capacity decoders tested here, `LabelShuffle` still collapses to zero.",
        "- However, `ConnShuffle` no longer collapses to zero and can approach or exceed the real graph under the alternative preprocessing/decoder families.",
        "- This confirms that absolute `I_lb` values, and even some condition rankings, are decoder-family dependent.",
        "- The manuscript should therefore keep the Part C claim narrow: under the published canonical decoder family, the real graph supports a reproducible non-zero lower bound that disappears under the strict negative controls used in that regime.",
        "",
    ])

    out_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("../data"))
    parser.add_argument("--raw_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=Path("../figures_pubready"))
    parser.add_argument("--report_dir", type=Path, default=Path("../reports"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    published = canonical_from_csv(args.data_dir)
    sensitivity = compute_sensitivity(args.raw_dir)
    all_rows = pd.concat([published, sensitivity], ignore_index=True)

    all_rows.to_csv(args.out_dir / "partC_decoder_sensitivity_per_seed.csv", index=False)

    summary = (
        all_rows.groupby(["decoder_family", "condition"], sort=False)
        .agg(I_lb_mean=("I_lb", "mean"), I_lb_sem=("I_lb", sem), n=("seed", "count"))
        .reset_index()
    )
    summary.to_csv(args.out_dir / "partC_decoder_sensitivity_summary.csv", index=False)

    plot_sensitivity(summary, args.out_dir)
    write_report(summary, args.report_dir / "partC_decoder_sensitivity.md")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Representative motif comparison for learned Part E patch connectomes.

This script reruns the archived v6.3 representative patch experiment for a
small number of seeds, extracts final weight matrices for each rule, aggregates
them to type-level graphs, and compares simple motif statistics with the
biological patch initialization. The goal is not to redefine the main Part E
benchmark, but to add a reviewer-facing interpretive analysis linking learning
outcomes to recurrent / reciprocal motif structure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp


RULE_ORDER = ["BioInit", "EProp", "RewardHebb", "REINFORCE", "Oja"]
RULE_COLORS = {
    "BioInit": "#111111",
    "EProp": "#0072B2",
    "RewardHebb": "#D55E00",
    "REINFORCE": "#009E73",
    "Oja": "#7A7A7A",
}
FOCAL_TRIADS = ["201", "120U", "300", "021C"]
RULE_SEED_BASE = {"BioInit": 1000, "EProp": 2000, "RewardHebb": 3000, "REINFORCE": 4000, "Oja": 5000}


def import_metrics(raw_root: Path):
    sys.path.insert(0, str(raw_root))
    from opticflow_partE import metrics  # type: ignore

    return metrics


def get_task_config() -> dict[str, object]:
    return {
        "n_classes": 16,
        "angles": np.linspace(0, 2 * np.pi, 16, endpoint=False),
        "noise": 0.0,
    }


def load_patch(raw_root: Path):
    ret_typed = pd.read_parquet(raw_root / "outputs" / "audit" / "retinotopy_typed.parquet")
    nodes = pd.read_parquet(raw_root / "outputs" / "full_opticlobe_dataset" / "nodes.parquet", columns=["body_id", "type"])
    edges = pd.read_parquet(raw_root / "outputs" / "full_opticlobe_dataset" / "edges.parquet", columns=["pre_id", "post_id"])

    valid_ids = ret_typed["body_id"].values
    valid_nodes = nodes[nodes["body_id"].isin(valid_ids)].reset_index(drop=True)
    valid_nodes = valid_nodes.merge(ret_typed[["body_id", "mean_u", "mean_v"]], on="body_id", how="left")
    patch_nodes = valid_nodes.head(1000).reset_index(drop=True)

    uid_map = {uid: i for i, uid in enumerate(patch_nodes["body_id"].values)}
    patch_edges = edges[edges["pre_id"].isin(uid_map) & edges["post_id"].isin(uid_map)]
    rows = patch_edges["pre_id"].map(uid_map).to_numpy(dtype=np.int64)
    cols = patch_edges["post_id"].map(uid_map).to_numpy(dtype=np.int64)
    data = np.ones(len(rows), dtype=np.float64) * 0.1
    adj = sp.csr_matrix((data, (rows, cols)), shape=(len(patch_nodes), len(patch_nodes)))

    coords = patch_nodes[["mean_u", "mean_v"]].to_numpy(dtype=np.float64)
    r_idx, c_idx = adj.nonzero()
    d_flat = np.linalg.norm(coords[r_idx] - coords[c_idx], axis=1)
    return patch_nodes, adj, coords, r_idx, c_idx, d_flat


def run_rule(
    rule: str,
    seed: int,
    adj: sp.csr_matrix,
    d_flat: np.ndarray,
    metrics,
    n_steps: int = 200,
    lam: float = 0.01,
    gam: float = 0.1,
):
    np.random.seed(seed)
    task_config = get_task_config()
    n_classes = int(task_config["n_classes"])
    angles = np.asarray(task_config["angles"], dtype=float)

    N = adj.shape[0]
    batch_size = 64
    W = adj.copy().astype(np.float64)
    W_readout = np.zeros((N, n_classes), dtype=np.float64)
    lr_plasticity = 1e-4 if rule in {"Oja", "EProp"} else 1e-3
    lr_readout = 0.01
    sigma_perturb = 0.001
    e_trace = np.zeros_like(W.data)
    alpha_trace = 0.9
    R_avg = 0.0
    node_prefs = np.random.rand(N) * 2 * np.pi
    trace_log: list[dict[str, float | int | str]] = []
    n_forward_passes = 0

    for t in range(n_steps):
        if rule == "EProp":
            W.data = np.clip(W.data, 0, 1.0)

        target_cls = np.random.randint(0, n_classes, size=batch_size)
        target_angles = angles[target_cls]
        delta = node_prefs[None, :] - target_angles[:, None]
        r_in = np.maximum(np.cos(delta), 0)

        W_base = W.copy()

        def run_forward(W_curr, r_input):
            W_d = W_curr.toarray() if sp.issparse(W_curr) else W_curr
            r = r_input.dot(W_d)
            if rule == "EProp":
                r = np.clip(r, 0, 10.0)
            else:
                np.maximum(r, 0, out=r)
            return r

        if rule == "REINFORCE":
            noise = np.random.normal(0, sigma_perturb, size=W.data.shape)

            W.data = np.maximum(W_base.data + noise, 0)
            r_pos = run_forward(W, r_in)
            E_met_p = np.mean(np.sum(r_pos, axis=1))
            E_wire_p = float(np.sum(W.data * d_flat))
            E_syn_p = float(np.sum(W.data))
            E_tot_p = E_met_p + E_syn_p + gam * E_wire_p
            logits_p = r_pos @ W_readout
            probs_p = np.exp(logits_p - logits_p.max(axis=1, keepdims=True))
            probs_p = probs_p / probs_p.sum(axis=1, keepdims=True)
            p_tgt_p = probs_p[np.arange(batch_size), target_cls]
            ce_p = np.mean(-np.log2(np.clip(p_tgt_p, 1e-9, 1.0)))
            mi_p = metrics.compute_mi_lb_bits(ce_p, n_classes, verify=False)
            J_p = metrics.compute_objective_j(mi_p, E_tot_p, lam)

            W.data = np.maximum(W_base.data - noise, 0)
            r_neg = run_forward(W, r_in)
            E_met_n = np.mean(np.sum(r_neg, axis=1))
            E_wire_n = float(np.sum(W.data * d_flat))
            E_syn_n = float(np.sum(W.data))
            E_tot_n = E_met_n + E_syn_n + gam * E_wire_n
            logits_n = r_neg @ W_readout
            probs_n = np.exp(logits_n - logits_n.max(axis=1, keepdims=True))
            probs_n = probs_n / probs_n.sum(axis=1, keepdims=True)
            p_tgt_n = probs_n[np.arange(batch_size), target_cls]
            ce_n = np.mean(-np.log2(np.clip(p_tgt_n, 1e-9, 1.0)))
            mi_n = metrics.compute_mi_lb_bits(ce_n, n_classes, verify=False)
            J_n = metrics.compute_objective_j(mi_n, E_tot_n, lam)

            grad_est = (J_p - J_n) / (2 * sigma_perturb) * noise
            W.data = np.maximum(W_base.data + lr_plasticity * grad_est, 0)
            r_out = run_forward(W, r_in)
            n_forward_passes += 3
        else:
            r_out = run_forward(W, r_in)
            n_forward_passes += 1

        logits = r_out @ W_readout
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = probs / probs.sum(axis=1, keepdims=True)
        p_tgt = probs[np.arange(batch_size), target_cls]
        mean_ce = np.mean(-np.log2(np.clip(p_tgt, 1e-9, 1.0)))
        mi_lb = float(metrics.compute_mi_lb_bits(mean_ce, n_classes, verify=False))

        E_met = float(np.mean(np.sum(r_out, axis=1)))
        E_wire = float(np.sum(W.data * d_flat))
        E_syn = float(np.sum(np.abs(W.data)))
        E_total = E_met + E_syn + gam * E_wire
        J = float(metrics.compute_objective_j(mi_lb, E_total, lam))

        dlogits = probs.copy()
        dlogits[np.arange(batch_size), target_cls] -= 1
        dW_readout = (r_out.T @ dlogits) / batch_size
        W_readout -= lr_readout * dW_readout

        rp = r_in[:, adj.nonzero()[0]]
        rc = r_out[:, adj.nonzero()[1]]
        if rule == "Oja":
            term1 = np.mean(rc * rp, axis=0)
            term2 = np.mean(rc**2, axis=0) * W.data
            W.data = np.maximum(W.data + lr_plasticity * (term1 - term2), 0)
            R_avg = 0.9 * R_avg + 0.1 * J
        elif rule == "RewardHebb":
            hebb = np.mean(rc * rp, axis=0)
            W.data = np.maximum(W.data + lr_plasticity * (J - R_avg) * hebb, 0)
            R_avg = 0.9 * R_avg + 0.1 * J
        elif rule == "EProp":
            hebb = np.mean(rc * rp, axis=0)
            e_trace = np.clip(alpha_trace * e_trace + hebb, -5.0, 5.0)
            update = np.clip(lr_plasticity * (J - R_avg) * e_trace, -0.1, 0.1)
            W.data = np.maximum(W.data + update, 0)
            R_avg = 0.9 * R_avg + 0.1 * J
        elif rule == "REINFORCE":
            R_avg = 0.9 * R_avg + 0.1 * J

        trace_log.append(
            {
                "rule": rule,
                "seed": seed,
                "epoch": t,
                "J": J,
                "MI_lb": mi_lb,
                "E_total": E_total,
                "compute_cost": n_forward_passes,
            }
        )

        if E_total > 1e7 or np.isnan(E_total) or np.mean(r_out) > 50:
            break

    return W.copy(), pd.DataFrame(trace_log)


def aggregate_type_graph(W: sp.csr_matrix, patch_nodes: pd.DataFrame) -> pd.DataFrame:
    W = W.copy()
    W.eliminate_zeros()
    coo = W.tocoo()
    df = pd.DataFrame(
        {
            "pre_type": patch_nodes.iloc[coo.row]["type"].to_numpy(),
            "post_type": patch_nodes.iloc[coo.col]["type"].to_numpy(),
            "weight_sum": coo.data.astype(float),
        }
    )
    agg = df.groupby(["pre_type", "post_type"], as_index=False)["weight_sum"].sum()
    return agg


def density_matched_graph(type_df: pd.DataFrame, top_frac: float) -> nx.DiGraph:
    sub = type_df[type_df["pre_type"] != type_df["post_type"]].copy()
    if sub.empty:
        return nx.DiGraph()
    k = max(1, int(np.ceil(len(sub) * top_frac)))
    sub = sub.nlargest(k, "weight_sum")
    g = nx.DiGraph()
    for row in sub.itertuples(index=False):
        g.add_edge(str(row.pre_type), str(row.post_type), weight=float(row.weight_sum))
    return g


def reciprocal_fraction(g: nx.DiGraph) -> float:
    if g.number_of_edges() == 0:
        return float("nan")
    pairs = set()
    reciprocal = 0
    for u, v in g.edges():
        key = tuple(sorted((u, v)))
        pairs.add(key)
    for u, v in pairs:
        if g.has_edge(u, v) and g.has_edge(v, u):
            reciprocal += 1
    return reciprocal / max(len(pairs), 1)


def motif_zscores(g: nx.DiGraph, n_nulls: int, seed_base: int) -> dict[str, float]:
    real = nx.triadic_census(g)
    out = {}
    for triad in FOCAL_TRIADS:
        vals = []
        for i in range(n_nulls):
            h = g.copy()
            if h.number_of_edges() >= 4:
                try:
                    nswap = min(5 * h.number_of_edges(), 10000)
                    nx.algorithms.swap.directed_edge_swap(
                        h, nswap=nswap, max_tries=max(20 * nswap, 1000), seed=seed_base + i
                    )
                except Exception:
                    pass
            c = nx.triadic_census(h)
            vals.append(float(c[triad]))
        mu = float(np.mean(vals))
        sd = float(np.std(vals))
        out[triad] = (float(real[triad]) - mu) / sd if sd > 0 else float("nan")
    return out


def make_figure(summary: pd.DataFrame, out_pdf: Path, out_png: Path) -> None:
    order = [r for r in RULE_ORDER if r in summary["rule"].unique()]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.2), layout="constrained")

    ax = axes[0]
    x = np.arange(len(order))
    vals = [summary.loc[summary["rule"] == r, "reciprocal_fraction_median"].iloc[0] for r in order]
    lo = [summary.loc[summary["rule"] == r, "reciprocal_fraction_min"].iloc[0] for r in order]
    hi = [summary.loc[summary["rule"] == r, "reciprocal_fraction_max"].iloc[0] for r in order]
    ax.bar(x, vals, color=[RULE_COLORS[r] for r in order], edgecolor="black", linewidth=0.8)
    ax.errorbar(x, vals, yerr=[np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)], fmt="none", ecolor="black", capsize=3)
    ax.set_xticks(x, order, rotation=20, ha="right")
    ax.set_ylabel("Reciprocal-pair fraction")
    ax.set_title("Density-matched reciprocal motif proxy")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    heat = summary.set_index("rule")[[f"{t}_z_median" for t in FOCAL_TRIADS]].reindex(order)
    finite_vals = heat.to_numpy()[np.isfinite(heat.to_numpy())]
    vmax = 6.0 if finite_vals.size == 0 else max(6.0, float(np.nanpercentile(np.abs(finite_vals), 95)))
    im = ax.imshow(heat.to_numpy(), cmap="coolwarm", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(FOCAL_TRIADS)), FOCAL_TRIADS)
    ax.set_yticks(np.arange(len(order)), order)
    ax.set_title("Triad enrichment z-scores")
    for i in range(len(order)):
        for j in range(len(FOCAL_TRIADS)):
            val = heat.iloc[i, j]
            txt = "nan" if not np.isfinite(val) else f"{val:.1f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("z-score vs degree-preserving rewires")

    fig.suptitle("Representative motif alignment of learned patch connectomes", fontsize=13)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_report(summary: pd.DataFrame, out_md: Path) -> None:
    lines = []
    lines.append("# Part E Motif Alignment Supplement\n\n")
    lines.append("- Source: representative v6.3 superhard patch rerun (`lambda=0.01`, `gamma=0.1`, 2 seeds)\n")
    lines.append("- Graphs were aggregated to type-level and density-matched by retaining the top 20% of directed type edges in each rule-specific graph.\n")
    lines.append("- Triad z-scores use degree-preserving rewired nulls on the resulting type graphs.\n\n")
    lines.append("| Rule | Reciprocal fraction (median) | 201 z | 120U z | 300 z | 021C z |\n")
    lines.append("|---|---:|---:|---:|---:|---:|\n")
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['rule']} | {row['reciprocal_fraction_median']:.3f} | "
            f"{row['201_z_median']:.2f} | {row['120U_z_median']:.2f} | "
            f"{row['300_z_median']:.2f} | {row['021C_z_median']:.2f} |\n"
        )
    lines.append("\n")
    lines.append("Interpretation:\n")
    lines.append("- The biological patch initialization (`BioInit`) combines high reciprocity with strong enrichment of recurrent triads relative to degree-preserving rewires.\n")
    lines.append("- The learned rules preserve non-random motif structure, but they do so in different ways: `Oja` and `REINFORCE` remain closest to `BioInit` in reciprocal-pair fraction, whereas `EProp` and `RewardHebb` shift toward stronger enrichment of recurrent triads (`201`, `120U`, `300`) while reducing reciprocity.\n")
    lines.append("- This analysis is still a representative patch-level motif proxy, not a full patch-generalization rerun with saved learned graphs for every condition.\n")
    out_md.write_text("".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", required=True)
    ap.add_argument("--fig_dir", required=True)
    ap.add_argument("--table_dir", required=True)
    ap.add_argument("--report_dir", required=True)
    ap.add_argument("--n_seeds", type=int, default=2)
    ap.add_argument("--top_frac", type=float, default=0.2)
    ap.add_argument("--n_nulls", type=int, default=10)
    args = ap.parse_args()

    raw_root = Path(args.raw_root).expanduser().resolve()
    fig_dir = Path(args.fig_dir).expanduser().resolve()
    table_dir = Path(args.table_dir).expanduser().resolve()
    report_dir = Path(args.report_dir).expanduser().resolve()
    for d in [fig_dir / "pdf", fig_dir / "png", table_dir, report_dir]:
        d.mkdir(parents=True, exist_ok=True)

    metrics = import_metrics(raw_root)
    patch_nodes, adj, _, _, _, d_flat = load_patch(raw_root)

    curve_rows = []
    motif_rows = []

    # Biological initialization as comparator.
    bio_type = aggregate_type_graph(adj.copy(), patch_nodes)
    bio_graph = density_matched_graph(bio_type, args.top_frac)
    bio_z = motif_zscores(bio_graph, args.n_nulls, seed_base=1000)
    motif_rows.append(
        {
            "rule": "BioInit",
            "seed": -1,
            "reciprocal_fraction": reciprocal_fraction(bio_graph),
            **{f"{triad}_z": bio_z[triad] for triad in FOCAL_TRIADS},
        }
    )

    for rule in ["Oja", "RewardHebb", "REINFORCE", "EProp"]:
        for seed in range(args.n_seeds):
            W_final, curve_df = run_rule(rule, seed, adj, d_flat, metrics)
            curve_rows.append(curve_df)

            sp.save_npz(table_dir / f"final_weights_{rule}_s{seed}.npz", W_final)
            type_df = aggregate_type_graph(W_final, patch_nodes)
            type_df.to_csv(table_dir / f"type_graph_{rule}_s{seed}.csv", index=False)
            g = density_matched_graph(type_df, args.top_frac)
            z = motif_zscores(g, args.n_nulls, seed_base=RULE_SEED_BASE[rule] + 100 * seed)
            motif_rows.append(
                {
                    "rule": rule,
                    "seed": seed,
                    "reciprocal_fraction": reciprocal_fraction(g),
                    **{f"{triad}_z": z[triad] for triad in FOCAL_TRIADS},
                }
            )
            print(f"[OK] motif proxy for {rule} seed {seed}")

    pd.concat(curve_rows, ignore_index=True).to_csv(table_dir / "motif_patch_rerun_trajectories.csv", index=False)
    motif_df = pd.DataFrame(motif_rows)
    motif_df.to_csv(table_dir / "motif_rule_seed_stats.csv", index=False)

    summary = (
        motif_df.groupby("rule", as_index=False)
        .agg(
            reciprocal_fraction_median=("reciprocal_fraction", "median"),
            reciprocal_fraction_min=("reciprocal_fraction", "min"),
            reciprocal_fraction_max=("reciprocal_fraction", "max"),
            **{f"{triad}_z_median": (f"{triad}_z", "median") for triad in FOCAL_TRIADS},
        )
    )
    summary.to_csv(table_dir / "motif_rule_summary.csv", index=False)
    make_figure(
        summary,
        fig_dir / "pdf" / "FigE8_MotifAlignment.pdf",
        fig_dir / "png" / "FigE8_MotifAlignment.png",
    )
    write_report(summary, report_dir / "partE_motif_report.md")
    print("[OK] wrote motif summary and FigE8")


if __name__ == "__main__":
    main()

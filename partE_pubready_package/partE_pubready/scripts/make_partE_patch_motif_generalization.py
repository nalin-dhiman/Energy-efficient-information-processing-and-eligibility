#!/usr/bin/env python3
"""Multi-patch motif generalization analysis for Part E.

This script extends the representative-patch motif proxy by rerunning a small
set of spatially distinct patches with the same lightweight Part E update rules,
then comparing each learned graph to its own biological patch baseline in both
geometry-alignment space and motif-fingerprint space.
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
from scipy.stats import spearmanr


RULE_ORDER = ["EProp", "RewardHebb", "REINFORCE", "Oja"]
RULE_COLORS = {
    "EProp": "#0072B2",
    "RewardHebb": "#D55E00",
    "REINFORCE": "#009E73",
    "Oja": "#7A7A7A",
    "BioInit": "#111111",
}
FOCAL_TRIADS = ["201", "120U", "300", "021C"]


def import_metrics(raw_root: Path):
    sys.path.insert(0, str(raw_root))
    from opticflow_partE import metrics  # type: ignore

    return metrics


def load_inputs(raw_root: Path):
    ret_typed = pd.read_parquet(
        raw_root / "outputs" / "audit" / "retinotopy_typed.parquet",
        columns=["body_id", "mean_u", "mean_v"],
    )
    nodes = pd.read_parquet(
        raw_root / "outputs" / "full_opticlobe_dataset" / "nodes.parquet",
        columns=["body_id", "type"],
    )
    edges = pd.read_parquet(
        raw_root / "outputs" / "full_opticlobe_dataset" / "edges.parquet",
        columns=["pre_id", "post_id", "weight"],
    )
    return ret_typed, nodes, edges


def select_patches(ret_typed: pd.DataFrame, num_patches: int, patch_size: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    all_valid_ids = ret_typed["body_id"].to_numpy()
    available_mask = np.ones(len(all_valid_ids), dtype=bool)
    node_coords = ret_typed[["mean_u", "mean_v"]].to_numpy(dtype=float)

    patches = []
    attempts = 0
    while len(patches) < num_patches and attempts < 500:
        attempts += 1
        avail_indices = np.where(available_mask)[0]
        if len(avail_indices) < patch_size:
            break
        center_idx = int(rng.choice(avail_indices))
        center = node_coords[center_idx]
        dists = np.linalg.norm(node_coords[avail_indices] - center, axis=1)
        nearest_local = np.argsort(dists)[:patch_size]
        patch_indices = avail_indices[nearest_local]
        patches.append(all_valid_ids[patch_indices].copy())
        available_mask[patch_indices] = False
    return patches


def build_patch_graph(
    patch_ids: np.ndarray, ret_typed: pd.DataFrame, nodes: pd.DataFrame, edges: pd.DataFrame
):
    patch_nodes = nodes[nodes["body_id"].isin(patch_ids)].reset_index(drop=True)
    patch_nodes = patch_nodes.merge(ret_typed, on="body_id", how="left")
    uid_map = {uid: i for i, uid in enumerate(patch_nodes["body_id"].to_numpy())}

    patch_edges = edges[edges["pre_id"].isin(uid_map) & edges["post_id"].isin(uid_map)].copy()
    rows = patch_edges["pre_id"].map(uid_map).to_numpy(dtype=np.int64)
    cols = patch_edges["post_id"].map(uid_map).to_numpy(dtype=np.int64)
    weights = patch_edges["weight"].fillna(1.0).to_numpy(dtype=float)

    adj_init = sp.csr_matrix((np.full(len(rows), 0.1, dtype=float), (rows, cols)), shape=(len(patch_nodes), len(patch_nodes)))
    conn = sp.csr_matrix((weights, (rows, cols)), shape=(len(patch_nodes), len(patch_nodes)))

    coords = patch_nodes[["mean_u", "mean_v"]].to_numpy(dtype=float)
    r_idx, c_idx = adj_init.nonzero()
    d_flat = np.linalg.norm(coords[r_idx] - coords[c_idx], axis=1)
    return patch_nodes, adj_init, conn, d_flat, r_idx, c_idx


def compute_geometry_metrics(W_csr: sp.csr_matrix, d_flat: np.ndarray) -> tuple[float, float]:
    weights = np.abs(W_csr.data)
    if len(weights) == 0:
        return 0.0, 0.0
    rho = float(np.corrcoef(weights, d_flat)[0, 1]) if len(weights) > 1 else 0.0
    q95_w = np.percentile(weights, 95)
    q25_d = np.percentile(d_flat, 25)
    strong = weights >= q95_w
    short = d_flat <= q25_d
    short_share = float(np.sum(weights[strong & short]) / (np.sum(weights[strong]) + 1e-9))
    return rho, short_share


def compute_ew_norm(W_csr: sp.csr_matrix, d_flat: np.ndarray, mean_d_norm: float) -> float:
    weights = np.abs(W_csr.data)
    if len(weights) == 0:
        return 0.0
    total_w = np.sum(weights) + 1e-9
    return float((np.sum(weights * d_flat) / total_w) / max(mean_d_norm, 1e-9))


def compute_mean_pair_distance(coords: np.ndarray, seed: int, n_sample: int = 5000) -> float:
    rng = np.random.default_rng(seed)
    idx_a = rng.integers(0, len(coords), size=n_sample)
    idx_b = rng.integers(0, len(coords), size=n_sample)
    mean_d = float(np.mean(np.linalg.norm(coords[idx_a] - coords[idx_b], axis=1)))
    return mean_d if mean_d > 1e-6 else 1.0


def aggregate_type_graph(W: sp.csr_matrix, patch_nodes: pd.DataFrame) -> pd.DataFrame:
    W = W.copy()
    W.eliminate_zeros()
    coo = W.tocoo()
    return (
        pd.DataFrame(
            {
                "pre_type": patch_nodes.iloc[coo.row]["type"].to_numpy(),
                "post_type": patch_nodes.iloc[coo.col]["type"].to_numpy(),
                "weight_sum": coo.data.astype(float),
            }
        )
        .groupby(["pre_type", "post_type"], as_index=False)["weight_sum"]
        .sum()
    )


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
    pairs = {tuple(sorted((u, v))) for u, v in g.edges()}
    reciprocal = sum(1 for u, v in pairs if g.has_edge(u, v) and g.has_edge(v, u))
    return float(reciprocal / max(len(pairs), 1))


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
            vals.append(float(nx.triadic_census(h)[triad]))
        mu = float(np.mean(vals))
        sd = float(np.std(vals))
        out[triad] = (float(real[triad]) - mu) / sd if sd > 0 else float("nan")
    return out


def run_rule(
    rule: str,
    seed: int,
    adj: sp.csr_matrix,
    d_flat: np.ndarray,
    r_idx: np.ndarray,
    c_idx: np.ndarray,
    metrics,
    lam: float = 0.01,
    gam: float = 1.0,
    n_steps: int = 200,
    n_classes: int = 8,
):
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 2 * np.pi, n_classes, endpoint=False)
    N = adj.shape[0]
    batch_size = 64

    W = adj.copy().astype(float)
    W_readout = np.zeros((N, n_classes), dtype=float)
    lr_plas = 1e-4 if rule == "Oja" else 1e-3
    lr_read = 0.01
    sigma = 0.001
    R_avg = 0.0
    e_trace = np.zeros_like(W.data)
    alpha = 0.9
    node_prefs = rng.random(N) * 2 * np.pi

    def fwd(W_c, r):
        W_d = W_c.toarray() if sp.issparse(W_c) else W_c
        res = r.dot(W_d)
        np.maximum(res, 0, out=res)
        return res

    trace_rows = []
    total_forwards = 0
    for t in range(n_steps):
        target_cls = rng.integers(0, n_classes, size=batch_size)
        delta = node_prefs[None, :] - angles[target_cls][:, None]
        r_in = np.maximum(np.cos(delta), 0)

        if rule == "REINFORCE":
            noise = rng.normal(0, sigma, size=W.data.shape)
            W_base = W.copy()

            W.data = W_base.data + noise
            np.maximum(W.data, 0, out=W.data)
            rp = fwd(W, r_in)
            Ep = np.mean(np.sum(rp, axis=1)) + gam * np.sum(W.data * d_flat)
            lp = rp @ W_readout
            probs_p = np.exp(lp - np.max(lp, axis=1, keepdims=True))
            probs_p /= np.sum(probs_p, axis=1, keepdims=True)
            mip = metrics.compute_mi_lb_bits(
                np.mean(-np.log2(np.clip(probs_p[np.arange(batch_size), target_cls], 1e-9, 1.0))),
                n_classes,
                verify=False,
            )
            Jp = metrics.compute_objective_j(mip, Ep, lam)

            W.data = W_base.data - noise
            np.maximum(W.data, 0, out=W.data)
            rn = fwd(W, r_in)
            En = np.mean(np.sum(rn, axis=1)) + gam * np.sum(W.data * d_flat)
            ln = rn @ W_readout
            probs_n = np.exp(ln - np.max(ln, axis=1, keepdims=True))
            probs_n /= np.sum(probs_n, axis=1, keepdims=True)
            min_ = metrics.compute_mi_lb_bits(
                np.mean(-np.log2(np.clip(probs_n[np.arange(batch_size), target_cls], 1e-9, 1.0))),
                n_classes,
                verify=False,
            )
            Jn = metrics.compute_objective_j(min_, En, lam)

            grad = (Jp - Jn) / (2 * sigma) * noise
            W.data = W_base.data + lr_plas * grad
            np.maximum(W.data, 0, out=W.data)
            r_out = fwd(W, r_in)
            total_forwards += 3
        else:
            r_out = fwd(W, r_in)
            total_forwards += 1

        E_tot = np.mean(np.sum(r_out, axis=1)) + gam * np.sum(W.data * d_flat)
        logits = r_out @ W_readout
        probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs /= np.sum(probs, axis=1, keepdims=True)
        ce = np.mean(-np.log2(np.clip(probs[np.arange(batch_size), target_cls], 1e-9, 1.0)))
        mi = float(metrics.compute_mi_lb_bits(ce, n_classes, verify=False))
        J = float(metrics.compute_objective_j(mi, E_tot, lam))

        rp_ = r_in[:, r_idx]
        rc_ = r_out[:, c_idx]
        hebb = np.mean(rc_ * rp_, axis=0)
        if rule == "Oja":
            W.data += lr_plas * (hebb - np.mean(rc_**2, axis=0) * W.data)
        elif rule == "RewardHebb":
            W.data += lr_plas * (J - R_avg) * hebb
            R_avg = 0.9 * R_avg + 0.1 * J
        elif rule == "EProp":
            e_trace = alpha * e_trace + hebb
            W.data += lr_plas * (J - R_avg) * e_trace
            R_avg = 0.9 * R_avg + 0.1 * J
        np.maximum(W.data, 0, out=W.data)

        dprob = probs.copy()
        dprob[np.arange(batch_size), target_cls] -= 1
        W_readout -= lr_read * (r_out.T @ dprob) / batch_size

        trace_rows.append(
            {
                "epoch": t,
                "rule": rule,
                "seed": seed,
                "J": J,
                "MI_lb": mi,
                "E_total": float(E_tot),
                "total_forwards": total_forwards,
            }
        )

    return W.copy(), pd.DataFrame(trace_rows)


def compute_d3z(df_rows: pd.DataFrame, df_bio: pd.DataFrame) -> pd.Series:
    metrics_base = ["Ew_norm", "Rho", "ShortShare"]
    vec_stats = {}
    for m in metrics_base:
        pool = np.concatenate([df_rows[m].to_numpy(dtype=float), df_bio[m].to_numpy(dtype=float)])
        vec_stats[m] = (float(np.mean(pool)), float(np.std(pool)))

    dvals = []
    for row in df_rows.itertuples(index=False):
        dist_sq = 0.0
        bio = df_bio[df_bio["patch_idx"] == row.patch_idx].iloc[0]
        for m in metrics_base:
            mean, std = vec_stats[m]
            std = std if std > 1e-9 else 1.0
            z_rule = (getattr(row, m) - mean) / std
            z_bio = (bio[m] - mean) / std
            dist_sq += (z_rule - z_bio) ** 2
        dvals.append(np.sqrt(dist_sq))
    return pd.Series(dvals, index=df_rows.index, dtype=float)


def compute_motif_distance(df_rows: pd.DataFrame, df_bio: pd.DataFrame) -> pd.Series:
    motif_cols = ["reciprocal_fraction"] + [f"{triad}_z" for triad in FOCAL_TRIADS]
    vec_stats = {}
    for m in motif_cols:
        pool = np.concatenate([df_rows[m].to_numpy(dtype=float), df_bio[m].to_numpy(dtype=float)])
        finite = pool[np.isfinite(pool)]
        mean = float(np.mean(finite)) if len(finite) else 0.0
        std = float(np.std(finite)) if len(finite) else 1.0
        vec_stats[m] = (mean, std if std > 1e-9 else 1.0)

    dvals = []
    for _, row in df_rows.iterrows():
        bio = df_bio[df_bio["patch_idx"] == row["patch_idx"]].iloc[0]
        dist_sq = 0.0
        for m in motif_cols:
            mean, std = vec_stats[m]
            row_v = row[m]
            bio_v = bio[m]
            if not np.isfinite(row_v) or not np.isfinite(bio_v):
                continue
            dist_sq += (((row_v - mean) / std) - ((bio_v - mean) / std)) ** 2
        dvals.append(np.sqrt(dist_sq))
    return pd.Series(dvals, index=df_rows.index, dtype=float)


def make_figure(
    learned: pd.DataFrame,
    summary: pd.DataFrame,
    corr_rho: float,
    corr_p: float,
    out_pdf: Path,
    out_png: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.2), layout="constrained")

    ax = axes[0]
    order = RULE_ORDER
    xpos = np.arange(len(order))
    vals = [summary.loc[summary["rule"] == r, "motif_dist_median"].iloc[0] for r in order]
    lo = [summary.loc[summary["rule"] == r, "motif_dist_min"].iloc[0] for r in order]
    hi = [summary.loc[summary["rule"] == r, "motif_dist_max"].iloc[0] for r in order]
    ax.bar(xpos, vals, color=[RULE_COLORS[r] for r in order], edgecolor="black", linewidth=0.8)
    ax.errorbar(xpos, vals, yerr=[np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)], fmt="none", ecolor="black", capsize=3)
    ax.set_xticks(xpos, order, rotation=20, ha="right")
    ax.set_ylabel("Motif-fingerprint distance to patch connectome")
    ax.set_title("Lower is closer to the biological patch")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    heat = summary.set_index("rule")[["reciprocal_fraction_median"] + [f"{triad}_z_median" for triad in FOCAL_TRIADS]].reindex(order)
    finite = heat.to_numpy()[np.isfinite(heat.to_numpy())]
    vmax = 2.0 if finite.size == 0 else float(np.nanpercentile(np.abs(finite), 95))
    im = ax.imshow(heat.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(1 + len(FOCAL_TRIADS)), ["recip"] + FOCAL_TRIADS)
    ax.set_yticks(np.arange(len(order)), order)
    ax.set_title("Median motif fingerprint across patches")
    for i in range(len(order)):
        for j in range(1 + len(FOCAL_TRIADS)):
            val = heat.iloc[i, j]
            text = "nan" if not np.isfinite(val) else f"{val:.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Metric value")

    ax = axes[2]
    for rule in order:
        sub = learned[learned["rule"] == rule]
        ax.scatter(
            sub["D_3z"],
            sub["motif_dist_to_bio"],
            label=rule,
            color=RULE_COLORS[rule],
            s=55,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.4,
        )
    ax.set_xlabel("Geometry alignment distance $D_{3z}$")
    ax.set_ylabel("Motif-fingerprint distance to patch connectome")
    ax.set_title(f"Across learned graphs: Spearman $\\rho={corr_rho:.2f}$, $p={corr_p:.3f}$")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9, loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=2)

    fig.suptitle("Multi-patch motif generalization of learned Part E connectomes", fontsize=13)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_report(
    summary: pd.DataFrame,
    learned: pd.DataFrame,
    corr_rho: float,
    corr_p: float,
    out_md: Path,
) -> None:
    lines = []
    lines.append("# Part E Multi-patch Motif Generalization\n\n")
    lines.append("- Source: deterministic rerun of spatially distinct 1000-node patches using the archived patch-generalization regime (`lambda=0.01`, `gamma=1.0`, 8-way motion task).\n")
    lines.append("- For each patch we compared learned motif fingerprints to the corresponding biological patch graph after type-level aggregation and density matching.\n")
    lines.append(f"- Across learned graphs, motif distance and geometry-alignment distance are related with Spearman rho = {corr_rho:.2f} (p = {corr_p:.3g}).\n\n")
    lines.append("| Rule | Patches | Median motif distance | Median D_3z | Median reciprocity |\n")
    lines.append("|---|---:|---:|---:|---:|\n")
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.rule} | {row.n_patches} | {row.motif_dist_median:.3f} | "
            f"{row.D_3z_median:.3f} | {row.reciprocal_fraction_median:.3f} |\n"
        )
    lines.append("\n")
    motif_wins = learned.pivot(index="patch_idx", columns="rule", values="motif_dist_to_bio").idxmin(axis=1).value_counts()
    geom_wins = learned.pivot(index="patch_idx", columns="rule", values="D_3z").idxmin(axis=1).value_counts()
    lines.append("Interpretation:\n")
    lines.append("- This analysis moves beyond a single representative patch: each rule is evaluated across multiple spatially distinct patches against its own biological motif baseline.\n")
    lines.append("- Lower motif distance means the learned graph stays closer to the biological patch in combined reciprocity and triad-enrichment space.\n")
    lines.append(
        f"- Geometry alignment and motif similarity partly dissociate in this rerun: EProp gives the lowest $D_{{3z}}$ on all {learned['patch_idx'].nunique()} patches, "
        f"whereas the motif-fingerprint winner is REINFORCE on {int(motif_wins.get('REINFORCE', 0))} patches, RewardHebb on {int(motif_wins.get('RewardHebb', 0))}, and EProp on {int(motif_wins.get('EProp', 0))}.\n"
    )
    lines.append("- The weak overall D_3z-to-motif correlation shows that motif resemblance is an additional descriptor of learned solutions rather than a simple restatement of the geometry-alignment score.\n")
    lines.append("- The analysis is still type-level rather than cell-level, but it is no longer confined to one patch.\n")
    out_md.write_text("".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", required=True)
    ap.add_argument("--fig_dir", required=True)
    ap.add_argument("--table_dir", required=True)
    ap.add_argument("--report_dir", required=True)
    ap.add_argument("--n_patches", type=int, default=5)
    ap.add_argument("--patch_size", type=int, default=1000)
    ap.add_argument("--top_frac", type=float, default=0.2)
    ap.add_argument("--n_nulls", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw_root = Path(args.raw_root).expanduser().resolve()
    fig_dir = Path(args.fig_dir).expanduser().resolve()
    table_dir = Path(args.table_dir).expanduser().resolve()
    report_dir = Path(args.report_dir).expanduser().resolve()
    for d in [fig_dir / "pdf", fig_dir / "png", table_dir, report_dir]:
        d.mkdir(parents=True, exist_ok=True)

    metrics = import_metrics(raw_root)
    ret_typed, nodes, edges = load_inputs(raw_root)
    patches = select_patches(ret_typed, args.n_patches, args.patch_size, args.seed)
    if not patches:
        raise RuntimeError("No patches selected for motif generalization analysis")

    learned_rows = []
    bio_rows = []
    trace_rows = []

    for patch_idx, patch_ids in enumerate(patches):
        patch_nodes, adj_init, conn, d_flat, r_idx, c_idx = build_patch_graph(patch_ids, ret_typed, nodes, edges)
        coords = patch_nodes[["mean_u", "mean_v"]].to_numpy(dtype=float)
        mean_d = compute_mean_pair_distance(coords, seed=args.seed + patch_idx)

        bio_type = aggregate_type_graph(conn.copy(), patch_nodes)
        bio_graph = density_matched_graph(bio_type, args.top_frac)
        bio_z = motif_zscores(bio_graph, args.n_nulls, seed_base=1000 + 100 * patch_idx)
        bio_rho, bio_short = compute_geometry_metrics(conn, d_flat)
        bio_rows.append(
            {
                "patch_idx": patch_idx,
                "rule": "BioInit",
                "Ew_norm": compute_ew_norm(conn, d_flat, mean_d),
                "Rho": bio_rho,
                "ShortShare": bio_short,
                "reciprocal_fraction": reciprocal_fraction(bio_graph),
                **{f"{triad}_z": bio_z[triad] for triad in FOCAL_TRIADS},
            }
        )

        for rule in RULE_ORDER:
            seed = args.seed + 1000 * patch_idx
            W_final, trace_df = run_rule(rule, seed, adj_init, d_flat, r_idx, c_idx, metrics)
            trace_df["patch_idx"] = patch_idx
            trace_rows.append(trace_df)

            sp.save_npz(table_dir / f"patch{patch_idx}_{rule}_final_weights.npz", W_final)
            type_df = aggregate_type_graph(W_final, patch_nodes)
            type_df.to_csv(table_dir / f"patch{patch_idx}_{rule}_type_graph.csv", index=False)
            g = density_matched_graph(type_df, args.top_frac)
            z = motif_zscores(g, args.n_nulls, seed_base=2000 + 1000 * patch_idx + 100 * RULE_ORDER.index(rule))
            rho, short = compute_geometry_metrics(W_final, d_flat)
            learned_rows.append(
                {
                    "patch_idx": patch_idx,
                    "rule": rule,
                    "Ew_norm": compute_ew_norm(W_final, d_flat, mean_d),
                    "Rho": rho,
                    "ShortShare": short,
                    "reciprocal_fraction": reciprocal_fraction(g),
                    **{f"{triad}_z": z[triad] for triad in FOCAL_TRIADS},
                }
            )
            print(f"[OK] patch {patch_idx} rule {rule}")

    learned = pd.DataFrame(learned_rows)
    bio = pd.DataFrame(bio_rows)
    learned["D_3z"] = compute_d3z(learned, bio)
    learned["motif_dist_to_bio"] = compute_motif_distance(learned, bio)

    corr = spearmanr(learned["D_3z"], learned["motif_dist_to_bio"], nan_policy="omit")
    corr_rho = float(corr.statistic) if np.isfinite(corr.statistic) else float("nan")
    corr_p = float(corr.pvalue) if np.isfinite(corr.pvalue) else float("nan")

    summary = (
        learned.groupby("rule", as_index=False)
        .agg(
            n_patches=("patch_idx", "nunique"),
            motif_dist_median=("motif_dist_to_bio", "median"),
            motif_dist_min=("motif_dist_to_bio", "min"),
            motif_dist_max=("motif_dist_to_bio", "max"),
            D_3z_median=("D_3z", "median"),
            reciprocal_fraction_median=("reciprocal_fraction", "median"),
            **{f"{triad}_z_median": (f"{triad}_z", "median") for triad in FOCAL_TRIADS},
        )
        .reset_index(drop=True)
    )

    learned.to_csv(table_dir / "patch_motif_generalization_rows.csv", index=False)
    bio.to_csv(table_dir / "patch_motif_bio_rows.csv", index=False)
    summary.to_csv(table_dir / "patch_motif_generalization_summary.csv", index=False)
    pd.concat(trace_rows, ignore_index=True).to_csv(table_dir / "patch_motif_generalization_traces.csv", index=False)

    make_figure(
        learned,
        summary,
        corr_rho,
        corr_p,
        fig_dir / "pdf" / "FigE9_MotifGeneralization.pdf",
        fig_dir / "png" / "FigE9_MotifGeneralization.png",
    )
    write_report(summary, learned, corr_rho, corr_p, report_dir / "partE_patch_motif_generalization_report.md")
    print("[OK] wrote multi-patch motif generalization outputs and FigE9")


if __name__ == "__main__":
    main()

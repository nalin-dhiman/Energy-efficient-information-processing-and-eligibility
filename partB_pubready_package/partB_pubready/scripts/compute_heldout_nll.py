#!/usr/bin/env python3
"""
Held-out Poisson NLL diagnostics for Part B structural models.

This script combines two evaluation modes on the mapped-core subset:

1. Exact dyad-split evaluation for M0_ER and M2_typeSBM.
   Because these models depend only on ordered type-pair counts, the full
   held-out Poisson NLL can be computed exactly from blockwise train/test
   sufficient statistics without enumerating all N^2 dyads.

2. Monte Carlo dyad-split evaluation for M3_type+dist.
   The distance-augmented model depends on ordered type pair and distance bin.
   Exhaustive enumeration over all mapped-core dyads is too expensive in the
   archived release, so M3 is evaluated on a large random sample of dyads. A
   coordinate-shuffled M3 null is evaluated on the same sampled dyads.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln


EPS = 1e-12


@dataclass
class GraphData:
    n_nodes: int
    n_types: int
    type_labels: np.ndarray
    coords: np.ndarray
    edge_rows: np.ndarray
    edge_cols: np.ndarray
    edge_weights: np.ndarray
    edge_pair_idx: np.ndarray
    edge_flat_sorted: np.ndarray
    edge_weight_sorted: np.ndarray
    zero_counts_by_pair: np.ndarray
    total_edge_count: int


def load_mapped_core(raw_root: Path) -> GraphData:
    full_dir = raw_root / "outputs" / "full_opticlobe_dataset"
    audit_path = raw_root / "outputs" / "audit" / "retinotopy_typed.parquet"

    nodes = pd.read_parquet(full_dir / "nodes.parquet", columns=["body_id", "type"])
    edges = pd.read_parquet(full_dir / "edges.parquet", columns=["pre_id", "post_id", "weight"])
    ret_typed = pd.read_parquet(audit_path, columns=["body_id", "mean_u", "mean_v"])

    valid_ids = set(ret_typed["body_id"].astype(np.int64))
    subset_nodes = nodes[nodes["body_id"].isin(valid_ids)].copy()
    subset_nodes = subset_nodes.merge(ret_typed, on="body_id", how="left")
    subset_nodes = subset_nodes.reset_index(drop=True)

    uid_map = {uid: i for i, uid in enumerate(subset_nodes["body_id"].astype(np.int64).tolist())}
    valid_edges = edges[edges["pre_id"].isin(uid_map) & edges["post_id"].isin(uid_map)].copy()

    rows = valid_edges["pre_id"].map(uid_map).to_numpy(dtype=np.int64, copy=False)
    cols = valid_edges["post_id"].map(uid_map).to_numpy(dtype=np.int64, copy=False)
    weights = valid_edges["weight"].to_numpy(dtype=np.float64, copy=False)

    type_labels, uniques = pd.factorize(subset_nodes["type"], sort=True)
    type_labels = type_labels.astype(np.int64, copy=False)
    coords = subset_nodes[["mean_u", "mean_v"]].to_numpy(dtype=np.float64, copy=False)

    n_nodes = len(subset_nodes)
    n_types = len(uniques)

    edge_pair_idx = type_labels[rows] * n_types + type_labels[cols]
    block_sizes = np.bincount(type_labels, minlength=n_types).astype(np.int64)
    dyads_by_pair = (block_sizes[:, None] * block_sizes[None, :]).reshape(-1)
    edge_count_by_pair = np.bincount(edge_pair_idx, minlength=n_types * n_types).astype(np.int64)
    zero_counts_by_pair = dyads_by_pair - edge_count_by_pair

    edge_flat = rows * n_nodes + cols
    order = np.argsort(edge_flat)

    return GraphData(
        n_nodes=n_nodes,
        n_types=n_types,
        type_labels=type_labels,
        coords=coords,
        edge_rows=rows,
        edge_cols=cols,
        edge_weights=weights,
        edge_pair_idx=edge_pair_idx,
        edge_flat_sorted=edge_flat[order],
        edge_weight_sorted=weights[order],
        zero_counts_by_pair=zero_counts_by_pair,
        total_edge_count=len(weights),
    )


def sample_weights(graph: GraphData, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    flat = rows * graph.n_nodes + cols
    idx = np.searchsorted(graph.edge_flat_sorted, flat)
    hit = (idx < graph.edge_flat_sorted.size) & (graph.edge_flat_sorted[idx] == flat)
    weights = np.zeros(flat.shape[0], dtype=np.float64)
    weights[hit] = graph.edge_weight_sorted[idx[hit]]
    return weights


def poisson_nll_bits(n_test: np.ndarray, w_test: np.ndarray, logfact_test: np.ndarray, lam: np.ndarray) -> float:
    lam = np.clip(lam, EPS, None)
    nll_nats = np.sum(n_test * lam - w_test * np.log(lam) + logfact_test)
    return float(nll_nats / np.log(2.0))


def exact_m0_m2_split(graph: GraphData, rng: np.random.Generator, test_frac: float) -> dict[str, float]:
    edge_test_mask = rng.random(graph.edge_weights.size) < test_frac
    edge_train_mask = ~edge_test_mask

    pair_train = graph.edge_pair_idx[edge_train_mask]
    pair_test = graph.edge_pair_idx[edge_test_mask]
    weight_train = graph.edge_weights[edge_train_mask]
    weight_test = graph.edge_weights[edge_test_mask]

    num_pairs = graph.n_types * graph.n_types
    edge_train_count = np.bincount(pair_train, minlength=num_pairs).astype(np.int64)
    edge_test_count = np.bincount(pair_test, minlength=num_pairs).astype(np.int64)
    weight_train_sum = np.bincount(pair_train, weights=weight_train, minlength=num_pairs).astype(np.float64)
    weight_test_sum = np.bincount(pair_test, weights=weight_test, minlength=num_pairs).astype(np.float64)
    logfact_test_sum = np.bincount(pair_test, weights=gammaln(weight_test + 1.0), minlength=num_pairs).astype(np.float64)

    zero_test_count = rng.binomial(graph.zero_counts_by_pair, test_frac)
    zero_train_count = graph.zero_counts_by_pair - zero_test_count

    n_train = edge_train_count + zero_train_count
    n_test = edge_test_count + zero_test_count

    lam_m2 = np.zeros(num_pairs, dtype=np.float64)
    nonzero = n_train > 0
    lam_m2[nonzero] = weight_train_sum[nonzero] / n_train[nonzero]
    m2_total = poisson_nll_bits(n_test.astype(np.float64), weight_test_sum, logfact_test_sum, lam_m2)

    n_train_m0 = float(n_train.sum())
    n_test_m0 = float(n_test.sum())
    lam_m0 = max(float(weight_train_sum.sum()) / max(n_train_m0, 1.0), EPS)
    m0_total = float((n_test_m0 * lam_m0 - weight_test_sum.sum() * np.log(lam_m0) + logfact_test_sum.sum()) / np.log(2.0))

    return {
        "n_test_dyads_exact": n_test_m0,
        "n_test_positive_dyads_exact": float(edge_test_count.sum()),
        "M0_ER_nll_bits_total": m0_total,
        "M0_ER_nll_bits_per_test_dyad": m0_total / max(n_test_m0, 1.0),
        "M2_typeSBM_nll_bits_total": m2_total,
        "M2_typeSBM_nll_bits_per_test_dyad": m2_total / max(n_test_m0, 1.0),
    }


def compute_distance_bins(graph: GraphData, rng: np.random.Generator, n_probe: int, n_bins: int) -> np.ndarray:
    rows = rng.integers(0, graph.n_nodes, size=n_probe, dtype=np.int64)
    cols = rng.integers(0, graph.n_nodes, size=n_probe, dtype=np.int64)
    dists = np.linalg.norm(graph.coords[rows] - graph.coords[cols], axis=1)
    hi = float(np.percentile(dists, 99))
    if not np.isfinite(hi) or hi <= 0:
        hi = float(dists.max())
    if hi <= 0:
        hi = 1.0
    return np.linspace(0.0, hi, n_bins + 1)


def fit_and_eval_sampled(cell_train: np.ndarray, cell_test: np.ndarray, w_train: np.ndarray, w_test: np.ndarray, num_cells: int) -> float:
    n_train = np.bincount(cell_train, minlength=num_cells).astype(np.float64)
    n_test = np.bincount(cell_test, minlength=num_cells).astype(np.float64)
    weight_train_sum = np.bincount(cell_train, weights=w_train, minlength=num_cells).astype(np.float64)
    weight_test_sum = np.bincount(cell_test, weights=w_test, minlength=num_cells).astype(np.float64)
    logfact_test_sum = np.bincount(cell_test, weights=gammaln(w_test + 1.0), minlength=num_cells).astype(np.float64)

    lam = np.zeros(num_cells, dtype=np.float64)
    nonzero = n_train > 0
    lam[nonzero] = weight_train_sum[nonzero] / n_train[nonzero]
    total = poisson_nll_bits(n_test, weight_test_sum, logfact_test_sum, lam)
    return total


def sampled_m3_split(
    graph: GraphData,
    rng: np.random.Generator,
    bins: np.ndarray,
    n_dyads: int,
    train_frac: float,
) -> dict[str, float]:
    rows = rng.integers(0, graph.n_nodes, size=n_dyads, dtype=np.int64)
    cols = rng.integers(0, graph.n_nodes, size=n_dyads, dtype=np.int64)
    weights = sample_weights(graph, rows, cols)

    train_mask = rng.random(n_dyads) < train_frac
    test_mask = ~train_mask

    t_pre = graph.type_labels[rows]
    t_post = graph.type_labels[cols]
    pair_idx = (t_pre * graph.n_types + t_post).astype(np.int64, copy=False)
    n_cells = graph.n_types * graph.n_types * (len(bins) - 1)

    dists = np.linalg.norm(graph.coords[rows] - graph.coords[cols], axis=1)
    dist_bins = np.clip(np.digitize(dists, bins, right=False) - 1, 0, len(bins) - 2)
    pairdist_idx = pair_idx * (len(bins) - 1) + dist_bins

    m3_total = fit_and_eval_sampled(
        pairdist_idx[train_mask],
        pairdist_idx[test_mask],
        weights[train_mask],
        weights[test_mask],
        n_cells,
    )

    # Coordinate-shuffle null on the same sampled dyads.
    perm = rng.permutation(graph.n_nodes)
    shuffled_coords = graph.coords[perm]
    dists_null = np.linalg.norm(shuffled_coords[rows] - shuffled_coords[cols], axis=1)
    dist_bins_null = np.clip(np.digitize(dists_null, bins, right=False) - 1, 0, len(bins) - 2)
    pairdist_idx_null = pair_idx * (len(bins) - 1) + dist_bins_null

    m3_null_total = fit_and_eval_sampled(
        pairdist_idx_null[train_mask],
        pairdist_idx_null[test_mask],
        weights[train_mask],
        weights[test_mask],
        n_cells,
    )

    n_test = float(test_mask.sum())
    return {
        "n_test_dyads_sampled": n_test,
        "n_test_positive_dyads_sampled": float((weights[test_mask] > 0).sum()),
        "M3_type+dist_nll_bits_total": m3_total,
        "M3_type+dist_nll_bits_per_test_dyad": m3_total / max(n_test, 1.0),
        "M3_coordShuffle_nll_bits_total": m3_null_total,
        "M3_coordShuffle_nll_bits_per_test_dyad": m3_null_total / max(n_test, 1.0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", required=True, help="Root of optic-lobe-v1.1-neuprint-tables")
    ap.add_argument("--out_dir", required=True, help="Directory for CSV outputs")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--n_dyads_m3", type=int, default=1000000)
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--n_bins", type=int, default=10)
    ap.add_argument("--probe_dyads", type=int, default=250000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    raw_root = Path(args.raw_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    graph = load_mapped_core(raw_root)
    bins = compute_distance_bins(graph, np.random.default_rng(args.seed), args.probe_dyads, args.n_bins)

    split_rows = []
    for split in range(args.n_splits):
        rng = np.random.default_rng(args.seed + split + 1)
        exact = exact_m0_m2_split(graph, rng, test_frac=1.0 - args.train_frac)
        sampled = sampled_m3_split(graph, rng, bins, args.n_dyads_m3, train_frac=args.train_frac)
        row = {"split": split, **exact, **sampled}
        split_rows.append(row)
        print(
            f"[split {split}] "
            f"M0={row['M0_ER_nll_bits_per_test_dyad']:.4f}, "
            f"M2={row['M2_typeSBM_nll_bits_per_test_dyad']:.4f}, "
            f"M3={row['M3_type+dist_nll_bits_per_test_dyad']:.4f}, "
            f"M3_shuffle={row['M3_coordShuffle_nll_bits_per_test_dyad']:.4f}"
        )

    split_df = pd.DataFrame(split_rows)
    split_df.to_csv(out_dir / "TableB_HeldoutDyadNLL_splits.csv", index=False)

    summary_rows = [
        {
            "model": "M0_ER",
            "evaluation": "exact dyad split",
            "metric": "heldout_poisson_nll_bits_per_test_dyad",
            "mean": float(split_df["M0_ER_nll_bits_per_test_dyad"].mean()),
            "sd": float(split_df["M0_ER_nll_bits_per_test_dyad"].std(ddof=1)),
        },
        {
            "model": "M2_typeSBM",
            "evaluation": "exact dyad split",
            "metric": "heldout_poisson_nll_bits_per_test_dyad",
            "mean": float(split_df["M2_typeSBM_nll_bits_per_test_dyad"].mean()),
            "sd": float(split_df["M2_typeSBM_nll_bits_per_test_dyad"].std(ddof=1)),
        },
        {
            "model": "M3_type+dist",
            "evaluation": "sampled dyad split",
            "metric": "heldout_poisson_nll_bits_per_test_dyad",
            "mean": float(split_df["M3_type+dist_nll_bits_per_test_dyad"].mean()),
            "sd": float(split_df["M3_type+dist_nll_bits_per_test_dyad"].std(ddof=1)),
        },
        {
            "model": "M3_coordShuffle",
            "evaluation": "sampled dyad split",
            "metric": "heldout_poisson_nll_bits_per_test_dyad",
            "mean": float(split_df["M3_coordShuffle_nll_bits_per_test_dyad"].mean()),
            "sd": float(split_df["M3_coordShuffle_nll_bits_per_test_dyad"].std(ddof=1)),
        },
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_df["n_splits"] = int(args.n_splits)
    summary_df["m3_dyads_per_split"] = int(args.n_dyads_m3)
    summary_df["train_frac"] = float(args.train_frac)
    summary_df["distance_bins"] = int(args.n_bins)
    summary_df["mapped_core_nodes"] = int(graph.n_nodes)
    summary_df["mapped_core_edges"] = int(graph.total_edge_count)
    summary_df.to_csv(out_dir / "TableB_HeldoutDyadNLL_summary.csv", index=False)

    print(f"[OK] wrote {out_dir / 'TableB_HeldoutDyadNLL_splits.csv'}")
    print(f"[OK] wrote {out_dir / 'TableB_HeldoutDyadNLL_summary.csv'}")


if __name__ == "__main__":
    main()

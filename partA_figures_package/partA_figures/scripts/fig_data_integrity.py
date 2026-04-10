import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# If you keep your existing style module:
import utils_style as style


def _pick_retino_cols(retinotopy: pd.DataFrame):
    """
    Robustly select u,v columns.
    Priority: mean_u/mean_v, else u_rad/v_rad, else explicit 'u'/'v'.
    Hard-fail if ambiguous.
    """
    cols = list(retinotopy.columns)

    def pick_one(candidates):
        hits = [c for c in cols if c in candidates]
        return hits[0] if len(hits) == 1 else (hits if len(hits) > 1 else None)

    u = pick_one(["mean_u", "u_rad", "u"])
    v = pick_one(["mean_v", "v_rad", "v"])

    # If we got multiple hits, force you to decide (prevents silent schema bugs)
    if isinstance(u, list) or isinstance(v, list):
        raise ValueError(f"Ambiguous retinotopy columns. u={u}, v={v}. "
                         "Please choose explicitly in code.")
    if u is None or v is None:
        raise ValueError("Could not find retinotopy u/v columns. "
                         "Expected one of mean_u/u_rad/u and mean_v/v_rad/v.")
    return u, v


def make_fig_data_integrity(
    cells_path: str,
    edges_path: str,
    retino_path: str,
    out_dir: str,
    *,
    dpi: int = 600,
    log_base_10: bool = True,
    degree_kind: str = "out"  # "out" (pre_id) or "in" (post_id) or "total"
):
    os.makedirs(out_dir, exist_ok=True)
    style.set_pub_style()

    nodes = pd.read_parquet(cells_path)
    edges = pd.read_parquet(edges_path)
    ret = pd.read_parquet(retino_path)

    # ---- ID normalization
    if "neuron_id" in ret.columns and "bodyId" not in ret.columns:
        ret = ret.rename(columns={"neuron_id": "bodyId"})
    if "bodyId" not in nodes.columns and "neuron_id" in nodes.columns:
        nodes = nodes.rename(columns={"neuron_id": "bodyId"})

    if "bodyId" not in nodes.columns:
        raise ValueError("cells.parquet must contain bodyId (or neuron_id).")
    if "type" not in nodes.columns:
        raise ValueError("cells.parquet must contain 'type' column (NaN = untyped).")

    # ---- Retinotopy columns
    u_col, v_col = _pick_retino_cols(ret)

    # ---- Panel A counts
    n_total = len(nodes)
    n_typed = int(nodes["type"].notna().sum())
    n_untyped = n_total - n_typed

    # ---- Panel B: typed-only mapping coverage
    typed = nodes.loc[nodes["type"].notna(), ["bodyId", "type"]].copy()
    merged = typed.merge(ret[["bodyId", u_col, v_col]], on="bodyId", how="left")
    is_mapped = merged[u_col].notna() & merged[v_col].notna()
    n_mapped = int(is_mapped.sum())
    n_unmapped = len(merged) - n_mapped

    # ---- Panel C: synapse degree definition
    if "weight" not in edges.columns:
        raise ValueError("cell_graph.parquet must contain 'weight' (synapse count / weight).")

    if degree_kind == "out":
        if "pre_id" not in edges.columns:
            raise ValueError("cell_graph.parquet missing pre_id for out-degree.")
        deg = edges.groupby("pre_id")["weight"].sum().rename("synapses").reset_index()
        deg = deg.rename(columns={"pre_id": "bodyId"})
        degree_label = "Outgoing synapses"
    elif degree_kind == "in":
        if "post_id" not in edges.columns:
            raise ValueError("cell_graph.parquet missing post_id for in-degree.")
        deg = edges.groupby("post_id")["weight"].sum().rename("synapses").reset_index()
        deg = deg.rename(columns={"post_id": "bodyId"})
        degree_label = "Incoming synapses"
    elif degree_kind == "total":
        if "pre_id" not in edges.columns or "post_id" not in edges.columns:
            raise ValueError("cell_graph.parquet missing pre_id/post_id for total degree.")
        outd = edges.groupby("pre_id")["weight"].sum().rename("out").reset_index().rename(columns={"pre_id":"bodyId"})
        ind  = edges.groupby("post_id")["weight"].sum().rename("in").reset_index().rename(columns={"post_id":"bodyId"})
        deg = outd.merge(ind, on="bodyId", how="outer").fillna(0.0)
        deg["synapses"] = deg["out"] + deg["in"]
        deg = deg[["bodyId", "synapses"]]
        degree_label = "Total synapses"
    else:
        raise ValueError("degree_kind must be one of: out, in, total")

    merged_deg = merged[["bodyId"]].merge(deg, on="bodyId", how="left").fillna({"synapses": 0.0})

    mapped_ids = set(merged.loc[is_mapped, "bodyId"].values)
    syn_mapped = merged_deg.loc[merged_deg["bodyId"].isin(mapped_ids), "synapses"].to_numpy()
    syn_unmapped = merged_deg.loc[~merged_deg["bodyId"].isin(mapped_ids), "synapses"].to_numpy()

    # ---- log transform (and label correctly)
    if log_base_10:
        vals_mapped = np.log10(syn_mapped + 1.0)
        vals_unmapped = np.log10(syn_unmapped + 1.0)
        ylab = f"log10({degree_label} + 1)"
    else:
        vals_mapped = np.log1p(syn_mapped)
        vals_unmapped = np.log1p(syn_unmapped)
        ylab = f"ln({degree_label} + 1)"

    # ---- Figure layout
    fig = plt.figure(figsize=(10.2, 5.8))
    gs = fig.add_gridspec(1, 3, wspace=0.72)

    # Panel A
    ax0 = fig.add_subplot(gs[0])
    ax0.bar(["Total"], [n_typed], label="Typed", color=style.OKABE_ITO["blue"])
    ax0.bar(["Total"], [n_untyped], bottom=[n_typed], label="Untyped", color=style.OKABE_ITO["grey"])
    ax0.set_ylabel("Neuron count")
    ax0.set_title("Dataset composition")
    
    # Move legend outside
    style.outside_legend(ax0, loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=2, frameon=False)

    ax0.text(0, n_typed/2, f"{n_typed}\n({(n_typed/n_total)*100:.1f}%)",
             ha="center", va="center", color="white", fontsize=7, fontweight="bold")
    if n_untyped > 0:
        ax0.text(0, n_typed + n_untyped/2, f"{n_untyped}",
                 ha="center", va="center", color="white", fontsize=7)
    style.panel_label(ax0, "a")

    # Panel B
    ax1 = fig.add_subplot(gs[1])
    ax1.bar(["Typed"], [n_mapped], label="Mapped", color=style.OKABE_ITO["vermillion"])
    ax1.bar(["Typed"], [n_unmapped], bottom=[n_mapped], label="Unmapped", color=style.OKABE_ITO["grey"])
    ax1.set_ylabel("Neuron count")
    ax1.set_title("Spatial coverage")
    
    # Move legend outside
    style.outside_legend(ax1, loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=2, frameon=False)

    ax1.text(0, n_mapped/2, f"{n_mapped}\n({(n_mapped/len(merged))*100:.1f}%)",
             ha="center", va="center", color="white", fontsize=7, fontweight="bold")
    style.panel_label(ax1, "b")

    # Panel C
    ax2 = fig.add_subplot(gs[2])
    bp = ax2.boxplot(
        [vals_mapped, vals_unmapped],
        tick_labels=["Mapped", "Unmapped"],
        patch_artist=True,
        showfliers=False
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(style.OKABE_ITO["skyblue"])
    for med in bp["medians"]:
        med.set_color("black")

    ax2.set_ylabel(ylab)
    ax2.set_title("Connectivity bias check")

    # Expand y-limits to make room for text at the top
    y_min, y_max = ax2.get_ylim()
    ax2.set_ylim(y_min, y_max * 1.15)

    # N labels in axes coords (stable placement)
    ax2.text(0.25, 0.95, f"N={len(vals_mapped)}", transform=ax2.transAxes,
             ha="center", va="top", fontsize=6)
    ax2.text(0.75, 0.95, f"N={len(vals_unmapped)}", transform=ax2.transAxes,
             ha="center", va="top", fontsize=6)

    style.panel_label(ax2, "c")

    # Save as editable PDF + high DPI PNG
    fig.subplots_adjust(bottom=0.26, top=0.90, left=0.07, right=0.98, wspace=0.72)
    style.save_figure(fig, out_dir, "FigA1_DataIntegrity", dpi_png=dpi)
    style.save_figure(fig, out_dir, "FigA1_DataIntegrity_EDITED", dpi_png=dpi)
    plt.close(fig)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Data Integrity Figure (A1)")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--out_dir", type=str, default="figures_out", help="Path to output directory")
    parser.add_argument("--dpi", type=int, default=600, help="DPI for raster output")
    
    args = parser.parse_args()

    # Construct paths assuming standard directory structure inside data_dir
    cells_p = os.path.join(args.data_dir, "cells.parquet")
    edges_p = os.path.join(args.data_dir, "cell_graph.parquet")
    ret_p = os.path.join(args.data_dir, "retinotopy.parquet")

    print(f"Running fig_data_integrity...")
    print(f"  Data dir: {args.data_dir}")
    print(f"  Out dir:  {args.out_dir}")

    make_fig_data_integrity(
        cells_path=cells_p,
        edges_path=edges_p,
        retino_path=ret_p,
        out_dir=args.out_dir,
        dpi=args.dpi,
        log_base_10=True,
        degree_kind="out",
    )

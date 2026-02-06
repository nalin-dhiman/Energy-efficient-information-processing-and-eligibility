

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from scipy.stats import bootstrap

import matplotlib as mpl
import matplotlib.pyplot as plt



def configure_matplotlib() -> None:
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.linewidth": 1.2,
        "lines.linewidth": 2.0,
        "patch.linewidth": 1.0,
        "pdf.fonttype": 42,  
        "ps.fonttype": 42,
    })


def save_pub(fig: plt.Figure, out_dir: Path, name: str) -> None:
    pdf_dir = out_dir / "pdf"
    png_dir = out_dir / "png"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(pdf_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(png_dir / f"{name}.png", bbox_inches="tight", dpi=600)
    plt.close(fig)



def bootstrap_mean_ci(x: np.ndarray, n_resamples: int = 10000, confidence_level: float = 0.95, seed: int = 0) -> Tuple[float, float]:
   
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return (np.nan, np.nan)
    res = bootstrap((x,), np.mean, n_resamples=n_resamples, confidence_level=confidence_level, random_state=seed, method="BCa")
    return (float(res.confidence_interval.low), float(res.confidence_interval.high))


def load_required_csv(data_dir: Path, name: str) -> pd.DataFrame:
    path = data_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def get_default_efficiency_row(df_sens: pd.DataFrame, alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0) -> pd.DataFrame:
    mask = (df_sens["alpha"] == alpha) & (df_sens["beta"] == beta) & (df_sens["gamma"] == gamma)
    out = df_sens.loc[mask].copy()
    if out.empty:

        triples = df_sens[["alpha","beta","gamma"]].drop_duplicates().to_numpy(float)
        target = np.array([alpha,beta,gamma], float)
        idx = int(np.argmin(np.linalg.norm(triples - target[None,:], axis=1)))
        a,b,g = triples[idx]
        out = df_sens[(df_sens["alpha"]==a)&(df_sens["beta"]==b)&(df_sens["gamma"]==g)].copy()
        out.attrs["note"] = f"[WARN] (alpha,beta,gamma)=({alpha},{beta},{gamma}) not found. Using closest ({a},{b},{g})."
    return out


def parse_bin_columns(df_local: pd.DataFrame) -> pd.DataFrame:
    
    df = df_local.copy()
    if "bin_u" in df.columns and "bin_v" in df.columns:
        df["bin_u"] = df["bin_u"].astype(int)
        df["bin_v"] = df["bin_v"].astype(int)
        return df
    if "bin" not in df.columns:
        raise KeyError(f"info_local.csv must contain either bin_u/bin_v or bin. Columns={list(df.columns)}")

    def parse_one(s: str) -> Tuple[int,int]:
        s = str(s).strip().strip("()")
        parts = [p.strip() for p in s.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Could not parse bin='{s}'")
        return int(parts[0]), int(parts[1])

    uv = df["bin"].apply(parse_one)
    df["bin_u"] = uv.apply(lambda t: t[0]).astype(int)
    df["bin_v"] = uv.apply(lambda t: t[1]).astype(int)
    return df



def make_tables(energy: pd.DataFrame, global_info: pd.DataFrame, local_info: pd.DataFrame, sens: pd.DataFrame, tables_dir: Path) -> Dict[str, pd.DataFrame]:
    tables_dir.mkdir(parents=True, exist_ok=True)


    e = energy.copy()
    comp_cols = ["base_component","spike_component","syn_component","wire_component"]
    for c in comp_cols:
        if c not in e.columns:
            e[c] = 0.0
    e["energy_total"] = e[comp_cols].sum(axis=1)
    for c in comp_cols:
        e[f"share_{c.replace('_component','')}"] = e[c] / e["energy_total"]

    e_out = e[["condition"] + comp_cols + ["energy_total"] + [f"share_{c.replace('_component','')}" for c in comp_cols]]
    e_out.to_csv(tables_dir / "TableD1_EnergyComponents.csv", index=False)


    g = global_info.copy()
    if "mi_lb" not in g.columns:
        raise KeyError(f"info_global.csv missing mi_lb. Columns={list(g.columns)}")

    loc = parse_bin_columns(local_info)
    loc_sum = (loc.groupby("condition")["mi_lb"]
               .agg(local_mean="mean", local_median="median", local_max="max", local_count="count")
               .reset_index())
    ci_rows = []
    for cond, sub in loc.groupby("condition"):
        lo, hi = bootstrap_mean_ci(sub["mi_lb"].to_numpy(float))
        ci_rows.append({"condition": cond, "local_mean_ci_lo": lo, "local_mean_ci_hi": hi})
    loc_ci = pd.DataFrame(ci_rows)

    info_out = (g.merge(loc_sum, on="condition", how="left")
                 .merge(loc_ci, on="condition", how="left"))
    info_out.to_csv(tables_dir / "TableD2_InfoSummary.csv", index=False)


    s = sens.copy()
    piv = s.pivot_table(index=["alpha","beta","gamma"], columns="condition", values=["Eff_Global","Eff_Local"])
    piv.columns = [f"{a}__{b}" for a,b in piv.columns]
    piv = piv.reset_index()
    if "Eff_Global__Real" in piv.columns and "Eff_Global__Null_Conn" in piv.columns:
        piv["ratio_global_real_vs_nullconn"] = piv["Eff_Global__Real"] / piv["Eff_Global__Null_Conn"]
        piv["ratio_local_real_vs_nullconn"] = piv["Eff_Local__Real"] / piv["Eff_Local__Null_Conn"]
    piv.to_csv(tables_dir / "TableD3_EfficiencySensitivity.csv", index=False)

    return {"energy": e_out, "info": info_out, "sensitivity": piv}



def figD1_energy_components(energy: pd.DataFrame, out_dir: Path) -> None:
    comp_cols = ["base_component","spike_component","syn_component","wire_component"]
    e = energy.copy()
    for c in comp_cols:
        if c not in e.columns:
            e[c] = 0.0
    e = e.set_index("condition")[comp_cols]
    e = e.loc[[c for c in ["Real","Null_Conn","Null_Strength"] if c in e.index] + [c for c in e.index if c not in ["Real","Null_Conn","Null_Strength"]]]

    e_m = e / 1e6 

    fig, ax = plt.subplots(figsize=(7.8, 4.9), constrained_layout=True)
    bottom = np.zeros(len(e_m))
    colors = {
        "base_component": "#4C78A8",
        "spike_component": "#F58518",
        "syn_component": "#54A24B",
        "wire_component": "#B279A2",
    }
    for col in comp_cols:
        ax.bar(e_m.index, e_m[col].values, bottom=bottom,
               label=col.replace("_component","").replace("_"," ").title(),
               edgecolor="black", linewidth=0.8, color=colors.get(col))
        bottom += e_m[col].values

    ax.set_ylabel("Energy per trial (×10⁶ a.u.)")
    ax.set_title("Energy decomposition by condition")
    ax.tick_params(axis="x", rotation=0)

    totals = e_m.sum(axis=1).values
    for i, tot in enumerate(totals):
        ax.text(i, tot + 0.12, f"{tot:.1f}", ha="center", va="bottom", fontsize=11)

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True, title="Component")

    save_pub(fig, out_dir, "FigD1_EnergyComponents")


def figD2_info_and_efficiency(global_info: pd.DataFrame, local_info: pd.DataFrame, sens: pd.DataFrame, out_dir: Path) -> None:
    g = global_info.copy()
    if "mi_lb" not in g.columns:
        raise KeyError(f"info_global.csv missing mi_lb. Columns={list(g.columns)}")

    loc = parse_bin_columns(local_info)
    loc_mean = loc.groupby("condition")["mi_lb"].mean().rename("local_mean")

    ci_rows = []
    for cond, sub in loc.groupby("condition"):
        lo, hi = bootstrap_mean_ci(sub["mi_lb"].to_numpy(float))
        ci_rows.append((cond, lo, hi))
    loc_ci = pd.DataFrame(ci_rows, columns=["condition","local_ci_lo","local_ci_hi"]).set_index("condition")

    g = g.set_index("condition").join(loc_mean, how="left").join(loc_ci, how="left").reset_index()

    s_def = get_default_efficiency_row(sens, 1.0, 1.0, 1.0)
    note = s_def.attrs.get("note", None)

    order = [c for c in ["Real","Null_Conn","Null_Strength"] if c in g["condition"].tolist()]
    g["condition"] = pd.Categorical(g["condition"], categories=order, ordered=True)
    g = g.sort_values("condition")

    s_def["condition"] = pd.Categorical(s_def["condition"], categories=order, ordered=True)
    s_def = s_def.sort_values("condition")

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.9), constrained_layout=True)

    ax = axes[0]
    x = np.arange(len(g))
    w = 0.38

    ax.bar(x - w/2, g["mi_lb"].values, width=w, label="Global $I_{lb}$", edgecolor="black", linewidth=0.8)
    if {"mi_low","mi_high"}.issubset(g.columns):
        yerr = np.vstack([g["mi_lb"].values - g["mi_low"].values, g["mi_high"].values - g["mi_lb"].values])
        ax.errorbar(x - w/2, g["mi_lb"].values, yerr=yerr, fmt="none", ecolor="black", elinewidth=1.2, capsize=4)

    ax.bar(x + w/2, g["local_mean"].values, width=w, label="Mean local $I_{lb}$", edgecolor="black", linewidth=0.8)
    yerr_l = np.vstack([g["local_mean"].values - g["local_ci_lo"].values, g["local_ci_hi"].values - g["local_mean"].values])
    ax.errorbar(x + w/2, g["local_mean"].values, yerr=yerr_l, fmt="none", ecolor="black", elinewidth=1.2, capsize=4)

    ax.set_xticks(x, g["condition"].astype(str))
    ax.set_ylabel("Information lower bound (bits)")
    ax.set_title("Information (global vs local)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.18), ncol=2, frameon=False)

    ax = axes[1]
    x2 = np.arange(len(s_def))
    ax.bar(x2 - w/2, s_def["Eff_Global"].values, width=w, label="Global eff. (bits / energy)", edgecolor="black", linewidth=0.8)
    ax.bar(x2 + w/2, s_def["Eff_Local"].values, width=w, label="Local eff. (bits / energy)", edgecolor="black", linewidth=0.8)
    ax.set_xticks(x2, s_def["condition"].astype(str))
    ax.set_ylabel("Efficiency (bits per energy unit)")
    ax.set_title("Energy efficiency (α=β=γ=1)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.18), ncol=2, frameon=False)

    if note:
        fig.suptitle(note, y=1.02, fontsize=11)

    save_pub(fig, out_dir, "FigD2_InfoAndEfficiency")


def figD3_local_mi_maps(local_info: pd.DataFrame, out_dir: Path) -> None:
    loc = parse_bin_columns(local_info)
    required = {"bin_u","bin_v","mi_lb","condition"}
    missing = required - set(loc.columns)
    if missing:
        raise KeyError(f"info_local.csv missing columns: {missing}. Found: {list(loc.columns)}")

    u_vals = np.sort(loc["bin_u"].unique())
    v_vals = np.sort(loc["bin_v"].unique())
    nu, nv = len(u_vals), len(v_vals)

    cond_order = [c for c in ["Real","Null_Conn","Null_Strength"] if c in loc["condition"].unique()]
    ncond = len(cond_order)
    fig, axes = plt.subplots(1, ncond, figsize=(4.2*ncond + 1.8, 5.0), constrained_layout=True, sharex=True, sharey=True)
    if ncond == 1:
        axes = [axes]

    vmin = float(loc["mi_lb"].min())
    vmax = float(loc["mi_lb"].max())

    im = None
    u2i = {u:i for i,u in enumerate(u_vals)}
    v2i = {v:i for i,v in enumerate(v_vals)}

    for ax, cond in zip(axes, cond_order):
        sub = loc[loc["condition"] == cond]
        grid = np.full((nv, nu), np.nan, dtype=float)
        for _, r in sub.iterrows():
            grid[v2i[int(r["bin_v"])], u2i[int(r["bin_u"])]] = float(r["mi_lb"])
        im = ax.imshow(grid, origin="lower", interpolation="nearest", vmin=vmin, vmax=vmax, aspect="auto")
        im.set_rasterized(True)  # keeps PDF small + fast
        ax.set_title(cond)
        ax.set_xlabel("Retinotopy U bin")
        ax.set_xticks(np.arange(nu))
        ax.set_xticklabels([str(int(u)) for u in u_vals], rotation=0)
        ax.set_yticks(np.arange(nv))
        ax.set_yticklabels([str(int(v)) for v in v_vals], rotation=0)

    axes[0].set_ylabel("Retinotopy V bin")
    cbar = fig.colorbar(im, ax=axes, shrink=0.90, pad=0.02)
    cbar.set_label("Local $I_{lb}$ (bits)")

    save_pub(fig, out_dir, "FigD3_LocalMIMaps")


def figD4_efficiency_sensitivity(sens: pd.DataFrame, out_dir: Path) -> None:
    piv = sens.pivot_table(index=["alpha","beta","gamma"], columns="condition", values=["Eff_Global","Eff_Local"])
    piv.columns = [f"{a}__{b}" for a,b in piv.columns]
    piv = piv.reset_index()

    needed = ["Eff_Global__Real","Eff_Global__Null_Conn","Eff_Local__Real","Eff_Local__Null_Conn"]
    for c in needed:
        if c not in piv.columns:
            raise KeyError(f"sensitivity_grid.csv does not provide {c}. Columns={list(piv.columns)}")

    piv["ratio_g"] = piv["Eff_Global__Real"] / piv["Eff_Global__Null_Conn"]
    piv["ratio_l"] = piv["Eff_Local__Real"] / piv["Eff_Local__Null_Conn"]

    alpha_vals = np.sort(piv["alpha"].unique())
    beta_vals  = np.sort(piv["beta"].unique())
    gamma_vals = np.sort(piv["gamma"].unique())

    fig, axes = plt.subplots(2, len(gamma_vals), figsize=(4.2*len(gamma_vals) + 1.2, 7.2),
                             constrained_layout=True, sharex=True, sharey=True)

    vmin = 1.0
    vmax = float(max(piv["ratio_g"].max(), piv["ratio_l"].max()))
    vmax = max(vmax, 1.05)

    ims = []
    for j, gamma in enumerate(gamma_vals):
        sub = piv[piv["gamma"] == gamma]

        def to_grid(val_col: str) -> np.ndarray:
            grid = np.full((len(beta_vals), len(alpha_vals)), np.nan, dtype=float)
            a2i = {a:i for i,a in enumerate(alpha_vals)}
            b2i = {b:i for i,b in enumerate(beta_vals)}
            for _, r in sub.iterrows():
                grid[b2i[float(r["beta"])], a2i[float(r["alpha"])]] = float(r[val_col])
            return grid

        grid_g = to_grid("ratio_g")
        grid_l = to_grid("ratio_l")

        axg = axes[0, j]
        im_g = axg.imshow(grid_g, origin="lower", vmin=vmin, vmax=vmax, aspect="auto")
        im_g.set_rasterized(True)
        axg.set_title(f"γ={gamma:g}\nGlobal eff. ratio")
        axg.set_xlabel("α (spike weight)")
        if j == 0:
            axg.set_ylabel("β (syn weight)")
        axg.set_xticks(np.arange(len(alpha_vals)))
        axg.set_xticklabels([f"{a:g}" for a in alpha_vals], rotation=0)
        axg.set_yticks(np.arange(len(beta_vals)))
        axg.set_yticklabels([f"{b:g}" for b in beta_vals], rotation=0)

        axl = axes[1, j]
        im_l = axl.imshow(grid_l, origin="lower", vmin=vmin, vmax=vmax, aspect="auto")
        im_l.set_rasterized(True)
        axl.set_title(f"γ={gamma:g}\nLocal eff. ratio")
        axl.set_xlabel("α (spike weight)")
        if j == 0:
            axl.set_ylabel("β (syn weight)")
        axl.set_xticks(np.arange(len(alpha_vals)))
        axl.set_xticklabels([f"{a:g}" for a in alpha_vals], rotation=0)
        axl.set_yticks(np.arange(len(beta_vals)))
        axl.set_yticklabels([f"{b:g}" for b in beta_vals], rotation=0)

        ims.append(im_g)

    cbar = fig.colorbar(ims[0], ax=axes.ravel().tolist(), shrink=0.97, pad=0.02)
    cbar.set_label("Efficiency ratio: Real / Null_Conn")

    save_pub(fig, out_dir, "FigD4_EfficiencySensitivity")



def maybe_fig_stagewise(data_dir: Path, out_dir: Path) -> None:
    
    preferred = data_dir / "stagewise_decoder_results.csv"
    legacy = data_dir / "stagewise_decoder_results_energywiring.csv"
    path = preferred if preferred.exists() else legacy
    if not path.exists():
        return

    df = pd.read_csv(path)


    rename = {}
    if "stage" in df.columns:
        rename["stage"] = "Stage"
    if "mi_bits" in df.columns:
        rename["mi_bits"] = "MI_bits"
    df = df.rename(columns=rename)

    required = {"Stage","Model","MI_bits"}
    if not required.issubset(df.columns):
        return


    stage_candidates = list(df["Stage"].unique())
    canonical_order = ["T4", "T5", "L_Out", "Global"]
    if all(s in stage_candidates for s in canonical_order):
        stage_order = canonical_order
    else:
        stage_order = sorted(stage_candidates)

    df["Stage"] = pd.Categorical(df["Stage"], categories=stage_order, ordered=True)
    df = df.sort_values(["Stage","Model"])

    fig, ax = plt.subplots(figsize=(7.8, 4.3), constrained_layout=True)
    stages = [s for s in stage_order if s in df["Stage"].unique()]
    models = list(df["Model"].unique())

    x = np.arange(len(stages))
    width = 0.38 if len(models)==2 else min(0.22, 0.8/len(models))

    for i, m in enumerate(models):
        sub = df[df["Model"]==m].set_index("Stage").reindex(stages)
        ax.bar(x + (i-(len(models)-1)/2)*width, sub["MI_bits"].values, width=width,
               label=m, edgecolor="black", linewidth=0.8)

    ax.set_xticks(x, stages)
    ax.set_ylabel("Decoder MI lower bound (bits)")
    ax.set_title("Stage-wise information (supplementary)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02,1.0), frameon=True)

    save_pub(fig, out_dir, "FigD_S1_StagewiseMI")



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="Directory containing Part D CSVs")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory for figures")
    ap.add_argument("--tables_dir", type=str, required=True, help="Output directory for tables")
    args = ap.parse_args()

    configure_matplotlib()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    tables_dir = Path(args.tables_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    energy = load_required_csv(data_dir, "energy_components.csv")
    global_info = load_required_csv(data_dir, "info_global.csv")
    local_info = load_required_csv(data_dir, "info_local.csv")
    sens = load_required_csv(data_dir, "sensitivity_grid.csv")

    make_tables(energy, global_info, local_info, sens, tables_dir)

    figD1_energy_components(energy, out_dir)
    figD2_info_and_efficiency(global_info, local_info, sens, out_dir)
    figD3_local_mi_maps(local_info, out_dir)
    figD4_efficiency_sensitivity(sens, out_dir)
    maybe_fig_stagewise(data_dir, out_dir)

    print("ok", out_dir)
    print("ok", tables_dir)


if __name__ == "__main__":
    main()

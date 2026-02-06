import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import argparse
import json
import re
import sys 

import utils_style as style


REQUIRED_INPUTS = {
    'nodes': ['cells.parquet'],
    'edges': ['cell_graph.parquet'],
    'retinotopy': ['retinotopy.parquet'],
    'provenance': ['provenance.json']
}



def load_and_map_data(data_dir):
    """
    Loads data and maps to standardized names:
    nodes (from cells.parquet)
    edges (from cell_graph.parquet)
    retinotopy (from retinotopy.parquet)
    """
    data = {}
    

    for key, candidates in REQUIRED_INPUTS.items():
        found = False
        for c in candidates:
            p = os.path.join(data_dir, c)
            if os.path.exists(p):

                if c.endswith('.json'):
                    with open(p, 'r') as f:
                        data[key] = json.load(f)
                else:
                    data[key] = pd.read_parquet(p)
                found = True
                print(f"Loaded {key} from {c}")
                break
        if not found:
            print(f"CRITICAL ERROR: Missing required input for '{key}'. Expected one of {candidates} in {data_dir}")
            sys.exit(1)
    
    if 'nodes' in data:
        df = data['nodes']

        if 'bodyId' not in df.columns and 'neuron_id' in df.columns:
            df.rename(columns={'neuron_id': 'bodyId'}, inplace=True)
            

    if 'edges' in data:
        df = data['edges']
        
        pass 
        
    return data

def check_coverage(nodes, retinotopy):

    n_total = len(nodes)
   
    mapped_ids = retinotopy.dropna(subset=['mean_u', 'mean_v'])
    
   
    if 'bodyId' in retinotopy.columns:
        common = set(nodes['bodyId']).intersection(set(mapped_ids['bodyId']))
    elif 'neuron_id' in retinotopy.columns:
        common = set(nodes['bodyId']).intersection(set(mapped_ids['neuron_id']))
    else:

        common = []
        
    n_mapped = len(common)
    pct = (n_mapped / n_total) * 100
    
    print(f"Coverage Check: {n_mapped}/{n_total} ({pct:.2f}%)")
    
    if pct < 30:
        print("WARNING")
        return pct, True
    return pct, False



def fig_A1(data, out_dir):
    
    print("Generating Fig A1...")
    nodes = data['nodes']
    retinotopy = data['retinotopy']
    edges = data['edges']
    
    fig = plt.figure(figsize=(style.DOUBLE_COLUMN_WIDTH, 3))
    gs = gridspec.GridSpec(1, 3, wspace=0.4)
    

    ax0 = fig.add_subplot(gs[0])
    
    n_total = len(nodes) 
    n_typed = nodes['type'].notna().sum()
    n_untyped = n_total - n_typed
    
    ax0.bar(['Total'], [n_typed], label='Typed', color=style.OKABE_ITO['blue'])
    ax0.bar(['Total'], [n_untyped], bottom=[n_typed], label='Untyped', color=style.OKABE_ITO['grey'])
    
    ax0.text(0, n_typed/2, f"{n_typed}\n({n_typed/n_total*100:.1f}%)", ha='center', va='center', color='white', fontsize=7, fontweight='bold')
    if n_untyped > 0:
        ax0.text(0, n_typed + n_untyped/2, f"{n_untyped}", ha='center', va='center', color='white', fontsize=7)
        
    ax0.set_ylabel("Neuron Count")
    ax0.set_title("Datset Composition")
    ax0.legend(loc='upper right', fontsize=6)
    style.panel_label(ax0, 'a')
    

    ax1 = fig.add_subplot(gs[1])
    

    typed_nodes = nodes[nodes['type'].notna()]
   
    r_cols = [c for c in retinotopy.columns if 'mean_u' in c or 'u_rad' in c]
    if not r_cols:

        print("CRITICAL: No retinotopy coords found.")
        return
        

    r_df = retinotopy.copy()
    if 'neuron_id' in r_df.columns:
        r_df.rename(columns={'neuron_id': 'bodyId'}, inplace=True)
        
    merged = pd.merge(typed_nodes, r_df, on='bodyId', how='left')
    
    has_coords = merged[r_cols[0]].notna().sum()
    missing_coords = len(merged) - has_coords
    
    ax1.bar(['Typed'], [has_coords], label='Mapped', color=style.OKABE_ITO['vermillion'])
    ax1.bar(['Typed'], [missing_coords], bottom=[has_coords], label='Unmapped', color=style.OKABE_ITO['grey'])
    
    ax1.text(0, has_coords/2, f"{has_coords}\n({has_coords/len(merged)*100:.1f}%)", ha='center', va='center', color='white', fontsize=7, fontweight='bold')
    
    ax1.set_ylabel("Neuron Count")
    ax1.set_title("Spatial Coverage")

    ax1.text(0.5, -0.2, "Coverage is partial;\nspatial analyses restricted", transform=ax1.transAxes, ha='center', va='top', fontsize=6, style='italic')
    ax1.legend(loc='upper right', fontsize=6)
    style.panel_label(ax1, 'b')
    

    ax2 = fig.add_subplot(gs[2])
    
   
    if 'pre_id' in edges.columns:
        out_degrees = edges.groupby('pre_id')['weight'].sum().reset_index()
        out_degrees.rename(columns={'pre_id': 'bodyId', 'weight': 'degree'}, inplace=True)
    else:
        
        print("CRITICAL: Edges file missing pre_id/weight")
        return


    deg_merged = pd.merge(merged[['bodyId']], out_degrees, on='bodyId', how='left').fillna({'degree': 0})
    
    
    mapped_ids = set(merged[merged[r_cols[0]].notna()]['bodyId'])
    
    vals_mapped = np.log1p(deg_merged[deg_merged['bodyId'].isin(mapped_ids)]['degree'])
    vals_unmapped = np.log1p(deg_merged[~deg_merged['bodyId'].isin(mapped_ids)]['degree'])
    
    bp = ax2.boxplot([vals_mapped, vals_unmapped], tick_labels=['Mapped', 'Unmapped'], patch_artist=True,
                     boxprops=dict(facecolor=style.OKABE_ITO['skyblue']),
                     medianprops=dict(color='black'))
    

    ax2.text(1, vals_mapped.max(), f"N={len(vals_mapped)}", ha='center', fontsize=6)
    ax2.text(2, vals_unmapped.max(), f"N={len(vals_unmapped)}", ha='center', fontsize=6)
    
    ax2.set_ylabel("Log10(Synapses + 1)")
    ax2.set_title("Bias Check")
    style.panel_label(ax2, 'c')
    
    style.save_figure(fig, 'FigA1_DataIntegrity', out_dir)
    plt.close(fig)

def fig_A2(data, out_dir):
    """
    Fig A2 — Retinotopy Geometry (Mapped Only)
    Hexbin of U,V
    """
    print("Generating Fig A2...")
    retinotopy = data['retinotopy']
    

    cols = retinotopy.columns
    u_col = next((c for c in cols if 'mean_u' in c or 'u_rad' in c), None)
    v_col = next((c for c in cols if 'mean_v' in c or 'v_rad' in c), None)
    
    valid = retinotopy.dropna(subset=[u_col, v_col])
    n_mapped = len(valid)
    n_total = len(data['nodes'])
    pct = (n_mapped / n_total) * 100
    
    fig, ax = plt.subplots(figsize=(style.SINGLE_COLUMN_WIDTH, style.SINGLE_COLUMN_WIDTH))
    
    hb = ax.hexbin(valid[u_col], valid[v_col], gridsize=40, cmap='viridis', mincnt=1)
    cb = plt.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('Neuron Count')
    
    ax.set_aspect('equal')
    ax.set_xlabel("Retinotopy U (azimuth)")
    ax.set_ylabel("Retinotopy V (elevation)")
    

    ax.set_title(f"Mapped neurons only\n(≈{pct:.1f}% of typed population)", fontsize=8)
    
    style.panel_label(ax, 'a')
    
    style.save_figure(fig, 'FigA2_RetinotopyCoverage', out_dir)
    plt.close(fig)

def fig_A3(data, out_dir):
    
    print("Generating Fig A3...")
    nodes = data['nodes']
    retinotopy = data['retinotopy']
    edges = data['edges']
    
    fig = plt.figure(figsize=(style.DOUBLE_COLUMN_WIDTH, 2.5))
    gs = gridspec.GridSpec(1, 3, wspace=0.3)
    

    t4t5 = nodes[nodes['type'].str.match(r'^T[45][abcd]$', na=False)].copy()
    subtype_counts = t4t5['type'].value_counts().sort_index()
    

    ax0 = fig.add_subplot(gs[0])
    ax0.bar(subtype_counts.index, subtype_counts.values, color=style.OKABE_ITO['skyblue'])
    ax0.set_ylabel("Count")
    ax0.set_title("Subtype Population")
    ax0.tick_params(axis='x', rotation=45)
    style.panel_label(ax0, 'a')
    

    ax1 = fig.add_subplot(gs[1])
    

    r_df = retinotopy.copy()
    if 'neuron_id' in r_df.columns:
        r_df.rename(columns={'neuron_id': 'bodyId'}, inplace=True)
    r_cols = [c for c in r_df.columns if 'mean_u' in c]
    
    merged = pd.merge(t4t5, r_df, on='bodyId', how='left')
    
    subtypes = sorted(subtype_counts.index)
    mapped_counts = []
    unmapped_counts = []
    
    for st in subtypes:
        sub = merged[merged['type'] == st]
        m = sub[r_cols[0]].notna().sum()
        mapped_counts.append(m)
        unmapped_counts.append(len(sub) - m)
        
    ax1.bar(subtypes, mapped_counts, label='Mapped', color=style.OKABE_ITO['vermillion'])
    ax1.bar(subtypes, unmapped_counts, bottom=mapped_counts, label='Unmapped', color=style.OKABE_ITO['grey'])
    ax1.set_title("Subtype Mapping")
    ax1.tick_params(axis='x', rotation=45)
    style.panel_label(ax1, 'b')
    

    ax2 = fig.add_subplot(gs[2])
    weights = edges['weight']
    ax2.hist(weights, bins=np.logspace(np.log10(weights.min()+0.1), np.log10(weights.max()), 40), 
             color=style.OKABE_ITO['black'], alpha=0.7)
    

    med = weights.median()
    p95 = weights.quantile(0.95)
    
    ax2.axvline(med, color=style.OKABE_ITO['vermillion'], ls='--', lw=1, label='Median')
    ax2.axvline(p95, color=style.OKABE_ITO['blue'], ls=':', lw=1, label='95%')
    
    ax2.set_xscale('log')
    ax2.set_xlabel("Synapse Count (Log)")
    ax2.set_title("Weight Distribution")
    ax2.legend(fontsize=6)
    
    style.panel_label(ax2, 'c')
    
    style.save_figure(fig, 'FigA3_T4T5_Subtypes', out_dir)
    plt.close(fig)

def fig_A4(data, out_dir):
    
    print("Generating Fig A4...")
   
    
    nodes = data['nodes']
    edges = data['edges']
    

    id_to_type = nodes.set_index('bodyId')['type'].to_dict()
    

    e = edges.copy()
    e['pre_type'] = e['pre_id'].map(id_to_type)
    e['post_type'] = e['post_id'].map(id_to_type)
    
    valid_e = e.dropna(subset=['pre_type', 'post_type'])
    
    
    agg = valid_e.groupby(['pre_type', 'post_type'])['weight'].sum().reset_index()
    

    mat = agg.pivot(index='pre_type', columns='post_type', values='weight').fillna(0)
    

    all_types = sorted(list(set(mat.index) | set(mat.columns)))
    mat = mat.reindex(index=all_types, columns=all_types, fill_value=0)
    

    t4 = sorted([t for t in all_types if re.match(r'^T4', t)])
    t5 = sorted([t for t in all_types if re.match(r'^T5', t)])
    others = sorted([t for t in all_types if t not in t4 + t5], key=lambda t: mat.loc[t].sum() + mat[t].sum(), reverse=True)
    
    sorted_types = t4 + t5 + others[:15]
    
    sub_mat = mat.loc[sorted_types, sorted_types]
    
    fig, ax = plt.subplots(figsize=(style.DOUBLE_COLUMN_WIDTH, style.DOUBLE_COLUMN_WIDTH))
    
    im = ax.imshow(np.log1p(sub_mat), cmap='magma')
    plt.colorbar(im, ax=ax, label='Log10(Synapses + 1)', fraction=0.046, pad=0.04)
    

    ax.set_xticks(range(len(sorted_types)))
    ax.set_xticklabels(sorted_types, rotation=90, fontsize=6)
    ax.set_yticks(range(len(sorted_types)))
    ax.set_yticklabels(sorted_types, fontsize=6)
    
    ax.set_xlabel("Postsynaptic Type")
    ax.set_ylabel("Presynaptic Type")
    ax.set_title("Type-Type Connectivity Matrix")
    
    style.panel_label(ax, 'a')
    
    style.save_figure(fig, 'FigA4_TypeConnectivity', out_dir)
    plt.close(fig)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--out_dir', required=True)
    args = parser.parse_args()
    

    data = load_and_map_data(args.data_dir)
    

    check_coverage(data['nodes'], data['retinotopy'])
    

    fig_A1(data, args.out_dir)
    fig_A2(data, args.out_dir)
    fig_A3(data, args.out_dir)
    fig_A4(data, args.out_dir)
    

    manifest = {
        "A1": {"caption": "Dataset integrity and coverage.", "sources": ["cells.parquet", "retinotopy.parquet", "cell_graph.parquet"]},
        "A2": {"caption": "Retinotopy geometry of mapped neurons.", "sources": ["retinotopy.parquet"]},
        "A3": {"caption": "T4/T5 subsystem composition.", "sources": ["cells.parquet", "cell_graph.parquet"]},
        "A4": {"caption": "Type-to-type connectivity.", "sources": ["cells.parquet", "cell_graph.parquet"]}
    }
    with open(os.path.join(os.path.dirname(args.out_dir), 'partA_figure_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

if __name__ == '__main__':
    main()

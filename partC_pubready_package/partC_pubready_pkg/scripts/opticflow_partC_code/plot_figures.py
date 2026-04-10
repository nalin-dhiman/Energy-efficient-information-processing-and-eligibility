import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def set_style():
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['svg.fonttype'] = 'none'
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("ticks")

def plot_global_metrics(csv_path, out_path):
    print("Plotting C-Fig1: Global Metrics...")
    df = pd.read_csv(csv_path)
    
    # Conditions: Real, LabelShuffle, ConnShuffle
    # Plot I_lb
    
    plt.figure(figsize=(6, 5))
    
    # Colors
    palette = {"Real": "#1f77b4", "LabelShuffle": "#7f7f7f", "ConnShuffle": "#d62728"}
    
    sns.barplot(data=df, x="condition", y="I_lb", palette=palette, capsize=.1)
    
    # Chance line? I_lb=0 is chance.
    plt.axhline(0, color='k', linestyle='--', linewidth=1)
    
    plt.ylabel("Information (bits)")
    plt.title("Global Decoding Performance")
    sns.despine()
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=600)
    plt.close()

def plot_local_heatmap(csv_path, out_path):
    print("Plotting C-Fig2: Local Decoding Heatmap...")
    df = pd.read_csv(csv_path)
    
    # Pivot to (u, v) -> I_lb
    # u_bin, v_bin are 0..5 coordinates
    heatmap_data = df.pivot(index="v_bin", columns="u_bin", values="ILB_Real")
    
    # Fill missing with 0 or NaN
    heatmap_data = heatmap_data.fillna(0)
    
    # Ensure all bins present
    for i in range(6):
        if i not in heatmap_data.index:
            heatmap_data.loc[i] = 0
        if i not in heatmap_data.columns:
            heatmap_data[i] = 0
            
    heatmap_data = heatmap_data.sort_index(ascending=False) # standard image coords
    
    plt.figure(figsize=(7, 6))
    sns.heatmap(heatmap_data, cmap="viridis", vmin=0, square=True, cbar_kws={'label': 'Information (bits)'})
    
    plt.xlabel("Azimuth (bin)")
    plt.ylabel("Elevation (bin)")
    plt.title("Local Decoding Information")
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=600)
    plt.close()
    
def plot_local_summary(csv_path, out_path):
    print("Plotting C-Fig2b: Local vs Null...")
    df = pd.read_csv(csv_path)
    
    # Compare Mean Real vs Mean Null across valid tiles
    # Melt?
    
    valid_tiles = df[df["n_neurons"] > 10].copy()
    
    plt.figure(figsize=(5, 5))
    
    # Bar plot of means
    means = [valid_tiles["ILB_Real"].mean(), valid_tiles["ILB_LabelShuffle"].mean()]
    sems = [valid_tiles["ILB_Real"].sem(), valid_tiles["ILB_LabelShuffle"].sem()]
    labels = ["Real Tiles", "Null Tiles"]
    
    plt.bar(labels, means, yerr=sems, capsize=10, color=["#1f77b4", "#7f7f7f"])
    
    plt.ylabel("Information (bits)")
    plt.title("Average Local Information")
    sns.despine()
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=600)
    plt.close()

def main():
    DATA_DIR = "outputs/partC_canonical"
    FIG_DIR = os.path.join(DATA_DIR, "figures")
    os.makedirs(FIG_DIR, exist_ok=True)
    
    set_style()
    
    metrics_path = os.path.join(DATA_DIR, "partC_metrics.csv")
    local_path = os.path.join(DATA_DIR, "partC_local_metrics.csv")
    
    if os.path.exists(metrics_path):
        plot_global_metrics(metrics_path, os.path.join(FIG_DIR, "C_Fig1_Global.png"))
        plot_global_metrics(metrics_path, os.path.join(FIG_DIR, "C_Fig1_Global.pdf"))
    
    if os.path.exists(local_path):
        plot_local_heatmap(local_path, os.path.join(FIG_DIR, "C_Fig2_Map.png"))
        plot_local_heatmap(local_path, os.path.join(FIG_DIR, "C_Fig2_Map.pdf"))
        
        plot_local_summary(local_path, os.path.join(FIG_DIR, "C_Fig2_Summary.png"))

if __name__ == "__main__":
    main()

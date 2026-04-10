
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Constants
DATA_PATH = "tables/TableD1_EnergyComponents.csv"
OUTPUT_DIR = "figures"
FIG_NAME_BASE = "FigD1_Energy_Audit"

# Visualization Config
COLORS = {
    "base_component": "#1f77b4",  # Blue
    "spike_component": "#ff7f0e", # Orange
    "syn_component": "#2ca02c",   # Green
    "wire_component": "#d62728"   # Red
}

LABELS = {
    "base_component": "Base",
    "spike_component": "Spike",
    "syn_component": "Synaptic",
    "wire_component": "Wiring"
}

ORDER = ["base_component", "spike_component", "syn_component", "wire_component"]

def load_data():
    """Loads the energy component data."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}")
    return pd.read_csv(DATA_PATH)

def audit_data(df):
    """
    Computes fractions and flags minor components.
    Returns diagnostics info.
    """
    df = df.copy()
    
    # Ensure raw components are present
    components = [c for c in ORDER if c in df.columns]
    
    # Calculate Total if not reliable or just recompute
    df["calculated_total"] = df[components].sum(axis=1)
    
    # Calculate Fractions
    fractions = pd.DataFrame()
    fractions["condition"] = df["condition"]
    for c in components:
        fractions[f"frac_{c}"] = df[c] / df["calculated_total"]
    
    # Check 1% rule
    hidden_components = set()
    warnings = []
    
    print("\n--- AUDITOR DIAGNOSTICS ---\n")
    
    print("1. Component Values:")
    print(df[["condition"] + components].to_string(index=False))
    print("\n")
    
    print("2. Fractional Contributions:")
    print(fractions.to_string(index=False, float_format="%.4f"))
    print("\n")
    
    print("3. Scale Dominance Check (< 1%):")
    for c in components:
        # Check if max fraction across conditions is < 0.01
        max_frac = fractions[f"frac_{c}"].max()
        if max_frac < 0.01:
            hidden_components.add(c)
            msg = f"[WARNING] Component '{c}' contributes < 1% (Max: {max_frac:.2%}). Visually suppressed at linear scale."
            warnings.append(msg)
            print(msg)
    
    if not warnings:
        print("All components contribute > 1% in at least one condition.")
        
    return df, fractions, hidden_components

def plot_figure(df, hidden_components):
    """
    Generates the Main + Inset plot.
    """
    conditions = df["condition"]
    x = np.arange(len(conditions))
    width = 0.6
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # --- MAIN PLOT (Dominant Terms) ---
    # We stack ALL components, but the small ones will be invisible
    bottom = np.zeros(len(conditions))
    
    # Plot in order
    for c in ORDER:
        ax.bar(x, df[c], bottom=bottom, width=width, label=LABELS[c], color=COLORS[c])
        bottom += df[c]
        
    ax.set_ylabel("Energy (Arbitrary Units)")
    ax.set_title("Energy Decomposition (Main Scale)")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    
    # Legend - Inverse order to match stack
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='upper left', bbox_to_anchor=(1, 1), title="Components")
    
    # --- INSET PLOT (Minor Terms) ---
    # If we have hidden components, we MUST show them in an inset or second panel.
    # We will put an inset on the top right or "best" location.
    
    if hidden_components:
        minor_cols = [c for c in ORDER if c in hidden_components]
        # Inset location: upper right, zooming in on the bottom of the bars??
        # Actually, since base/spike are at the BOTTOM of the stack (first in ORDER), 
        # a zoom-in on y=[0, small_limit] would work.
        
        # Calculate max height needed for minor components
        # sum of max of each minor component? No, sum of minor components per condition
        minor_totals = df[minor_cols].sum(axis=1)
        y_max_inset = minor_totals.max() * 1.5
        
        # Inset axes - Relocated to Side
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        # Position: bbox_to_anchor=(x, y, width, height) in normalized axes coordinates
        # Moved up (0.2) and right (1.25) to avoid overlap and ensure visibility
        ax_ins = inset_axes(ax, width="100%", height="100%", 
                            bbox_to_anchor=(1.25, 0.2, 0.45, 0.45),
                            bbox_transform=ax.transAxes,
                            loc='lower left')
        
        bottom_ins = np.zeros(len(conditions))
        for c in ORDER:
            # We plot ALL of them again, but we limit Y axis
            ax_ins.bar(x, df[c], bottom=bottom_ins, width=width, color=COLORS[c])
            bottom_ins += df[c]
            
        ax_ins.set_ylim(0, y_max_inset)
        ax_ins.set_xticks(x)
        # Add labels to inset since it's separate now
        ax_ins.set_xticklabels(conditions, rotation=45, ha='right', fontsize=8) 
        
        ax_ins.set_title("Minor Components\n(Zoom)", fontsize=10)
        ax_ins.grid(axis='y', linestyle='--', alpha=0.5)
        
        # Justification text
        txt = "Minor components (<1%) shown in inset"
        plt.figtext(0.99, 0.01, txt, horizontalalignment='right', fontsize=8, style='italic')

    # Formatting
    # Reserve even more space on the right (35%) to accommodate the shifted inset
    plt.tight_layout(rect=[0, 0, 0.65, 1]) 
    
    # Save
    out_png = os.path.join(OUTPUT_DIR, f"{FIG_NAME_BASE}.png")
    out_pdf = os.path.join(OUTPUT_DIR, f"{FIG_NAME_BASE}.pdf")
    
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    print(f"\nFigure saved to:\n  {out_png}\n  {out_pdf}")

if __name__ == "__main__":
    try:
        df = load_data()
        df, fractions, hidden = audit_data(df)
        plot_figure(df, hidden)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

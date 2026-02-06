import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import numpy as np

def generate_plots(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    print("Figures...")
    
    
    try:
        nodes = pd.read_parquet(os.path.join(data_dir, "outputs", "full_opticlobe_dataset", "nodes.parquet"))
        ret = pd.read_parquet(os.path.join(data_dir, "outputs", "audit", "retinotopy_typed.parquet"))
        
        typed_nodes = nodes[nodes["type"] != "UNKNOWN"].copy()
        mapped_ids = set(ret["body_id"])
        typed_nodes["Group"] = typed_nodes["body_id"].isin(mapped_ids).map({True: "Mapped", False: "Unmapped"})
        
        plt.figure(figsize=(6, 4))

        sns.histplot(data=typed_nodes, x="post_count", hue="Group", log_scale=True, element="step", common_norm=False)
        plt.title("Fig 1: Selection Bias (Degree Distribution)")
        plt.xlabel("Synapses (Post)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "fig1_bias_control.png"))
        plt.close()
        print("Generated Fig 1.")
    except Exception as e:
        print(f"Skipped Fig 1: {e}")


    try:
        df_mdl = pd.read_csv(os.path.join(data_dir, "outputs", "hardening_v2", "spatial_mdl_table.csv"))

        plt.figure(figsize=(6, 4))

        sns.barplot(data=df_mdl, x="model", y="mdl_bits", hue="scope")
        plt.title("Fig 2: Spatial Structure Compression")
        plt.ylabel("Description Length (Bits)")
        plt.yscale("log")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "fig2_spatial_mdl.png"))
        plt.close()
        print("Generated Fig 2.")
    except Exception as e:
        print(f"Skipped Fig 2: {e}")


    try:
        df_energy = pd.read_csv(os.path.join(data_dir, "outputs", "hardening_v2", "energy_stress_test.csv"))
        plt.figure(figsize=(8, 5))
        sns.barplot(data=df_energy, x="metric", y="energy", hue="null")
        plt.title("Fig 3: Efficiency Robustness (Stress Test)")
        plt.ylabel("Wiring Energy (a.u.)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "fig3_energy_robustness.png"))
        plt.close()
        print("Generated Fig 3.")
    except Exception as e:
        print(f"Skipped Fig 3: {e}")


    try:
        df_func = pd.read_csv(os.path.join(data_dir, "outputs", "final_controls", "functional_task_extension_results.csv"))
        plt.figure(figsize=(6, 4))
        sns.barplot(data=df_func, x="Stage", y="MI", hue="Task")
        plt.title("Fig 4: Functional Generalization")
        plt.ylabel("Mutual Information (Bits)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "fig4_functional_task.png"))
        plt.close()
        print("Generated Fig 4.")
    except Exception as e:
        print(f"Skipped Fig 4: {e}")
        

    try:
        pattern = os.path.join(data_dir, "outputs", "partE_runs", "learning_curves_*.csv")
        files = glob.glob(pattern)
        dfs = []
        for f in files:
            d = pd.read_csv(f)

            name = os.path.basename(f).replace("learning_curves_", "").replace(".csv", "")
            d["Run"] = name
            dfs.append(d)
            
        if dfs:
            df_curve = pd.concat(dfs)
            plt.figure(figsize=(8, 4))
            sns.lineplot(data=df_curve, x="epoch", y="J", hue="Run")
            plt.title("Fig 5: Plasticity Optimization (Objective J)")
            plt.ylabel("Objective J (Info - Energy)")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "fig5_plasticity_curves.png"))
            plt.close()
            print("Generated Fig 5.")
    except Exception as e:
        print(f"Skipped Fig 5: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    
    generate_plots(args.data, args.out)

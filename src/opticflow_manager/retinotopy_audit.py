import pandas as pd
import numpy as np
import os
import glob
from scipy.stats import entropy

def run_retinotopy_audit(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    report = []
    report.append("# Retinotopy Quality & Bias Report")
    

    ret_typed = pd.read_parquet(os.path.join(data_dir, "outputs", "audit", "retinotopy_typed.parquet"))
    nodes = pd.read_parquet(os.path.join(data_dir, "outputs", "full_opticlobe_dataset", "nodes.parquet"))
    
 
    
    avg_pins = ret_typed["pin_count"].mean()
    multi_pin_fraction = (ret_typed["pin_count"] > 1).mean()
    
    report.append("## 1. Mapping Quality")
    report.append(f"- **Mean Pins per Neuron**: {avg_pins:.2f}")
    report.append(f"- **Multi-Pin Fraction**: {multi_pin_fraction:.2%}")
    report.append(f"- **Mean Spatial StdDev**: {ret_typed['std_u'].mean():.2f} units")
    

    if "primary_region" in ret_typed.columns:
        counts = ret_typed["primary_region"].value_counts()
        report.append("\n### Primary Neuropil Breakdown")
        report.append(counts.to_markdown())
        
   
    typed_nodes = nodes[nodes["type"] != "UNKNOWN"].copy()
    
    mapped_ids = set(ret_typed["body_id"])
    typed_nodes["is_mapped"] = typed_nodes["body_id"].isin(mapped_ids)
    
    mapped = typed_nodes[typed_nodes["is_mapped"]]
    unmapped = typed_nodes[~typed_nodes["is_mapped"]]
    
    report.append("\n## 2. Selection Bias (Mapped vs Unmapped)")
    report.append(f"- **Mapped N**: {len(mapped)}")
    report.append(f"- **Unmapped N**: {len(unmapped)}")
    

    def get_stats(df, col):
        return f"{df[col].mean():.1f} ± {df[col].std():.1f}"
        
    report.append("\n| Metric | Mapped | Unmapped | Ratio (M/U) |")
    report.append("| --- | --- | --- | --- |")
    
    metrics = ["pre_count", "post_count", "size"]
    for m in metrics:
        if m in typed_nodes.columns:
            m_val = mapped[m].mean()
            u_val = unmapped[m].mean()
            ratio = m_val / u_val if u_val > 0 else 0
            report.append(f"| {m} | {get_stats(mapped, m)} | {get_stats(unmapped, m)} | {ratio:.2f} |")
            
    
    with open(os.path.join(output_dir, "retinotopy_quality_report.md"), "w") as f:
        f.write("\n".join(report))
        
   
    with open(os.path.join(output_dir, "bias_check_mapped_vs_unmapped.md"), "w") as f:
        f.write("# Bias Check: Mapped vs Unmapped Typed Neurons\n\n")
        f.write("We compared the subpopulation with valid retinotopy against those without.\n")
        f.write("Significant differences in degree or size indicate selection bias.\n\n")
        for m in metrics:
             if m in typed_nodes.columns:
                m_val = mapped[m].mean()
                u_val = unmapped[m].mean()
                report_line = f"- **{m}**: Mapped ({m_val:.1f}) vs Unmapped ({u_val:.1f})."
                f.write(report_line + "\n")
                if m_val > 2 * u_val:
                    f.write("  - **WARNING**: Mapped neurons are significantly larger/more connected.\n")
    
    print("Retinotopy Audit Complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    
    run_retinotopy_audit(args.data, args.out)

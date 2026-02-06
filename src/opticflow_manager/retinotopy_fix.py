import pandas as pd
import numpy as np
import os
import glob
from . import builder # reuse? No, standalone for audit safety

def fix_retinotopy_coverage(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print("Fixing Retinotopy Coverage...")
    
    
    n_path = os.path.join(data_dir, "Neuprint_Neurons.feather")
    neurons = pd.read_feather(n_path)
    

    type_col = "type" if "type" in neurons.columns else "type:string"
    pk_col = "bodyId" if "bodyId" in neurons.columns else ":ID(Body-ID)"
    
    typed_df = neurons[neurons[type_col].notna()].copy()
    typed_ids = set(typed_df[pk_col].unique())
    n_typed = len(typed_ids)
    
    print(f"Total Typed Neurons (Denominator): {n_typed}")
    
   
    ret_path = os.path.join(data_dir, "outputs", "full_opticlobe_dataset", "retinotopy.parquet")
    if os.path.exists(ret_path):
        ret = pd.read_parquet(ret_path)
    else:
       
        raise FileNotFoundError("Retinotopy parquet missing. Please run Phase 2 (Manager) first.")
        
    
    
    ret_ids = set(ret["body_id"].unique())
    

    mapped_typed_ids = typed_ids.intersection(ret_ids)
    n_mapped_typed = len(mapped_typed_ids)
    

    cov_typed = n_mapped_typed / n_typed if n_typed > 0 else 0
    
    print(f"Mapped Typed Neurons: {n_mapped_typed}")
    print(f"Coverage (Typed): {cov_typed:.2%}")
    

    

    ret_typed = ret[ret["body_id"].isin(mapped_typed_ids)].copy()
    out_pq = os.path.join(output_dir, "retinotopy_typed.parquet")
    ret_typed.to_parquet(out_pq)
    print(f"Saved {out_pq}")
    

    with open(os.path.join(output_dir, "corrected_retinotopy_coverage.md"), "w") as f:
        f.write("# Corrected Retinotopy Coverage\n\n")
        f.write(f"**Denominator (Typed Neurons)**: {n_typed}\n")
        f.write(f"**Numerator (Mapped Typed)**: {n_mapped_typed}\n")
        f.write(f"**Coverage**: {cov_typed:.2%}\n\n")
        f.write("## Notes\n")
        f.write("- Only counting neurons with assigned Types in Neuprint_Neurons.\n")
        f.write("- Mapping requires valid link in `retinotopy.parquet` (Medulla/Lobula/LP pins).\n")
        
    print("Retinotopy Fix Complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    
    fix_retinotopy_coverage(args.data, args.out)

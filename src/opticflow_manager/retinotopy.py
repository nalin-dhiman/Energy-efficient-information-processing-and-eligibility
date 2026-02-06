import pandas as pd
import numpy as np
import glob
import os
import time
from scipy.stats import entropy

def parse_pin_folder(folder_path, region_label):

    files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not files:
        return pd.DataFrame()
        
    dfs = []
   
    use_cols = [":ID(Element-ID)", "olHex1:int", "olHex2:int", "depth:int", "bodyId:long"]
    
    for f in files:
        try:
            df = pd.read_csv(f, usecols=lambda c: c in use_cols)
            df["region"] = region_label
            dfs.append(df)
        except Exception:
            continue
            
    if not dfs:
        return pd.DataFrame()
        
    full = pd.concat(dfs, ignore_index=True)

    rename = {
        ":ID(Element-ID)": "element_id",
        "olHex1:int": "u",
        "olHex2:int": "v",
        "depth:int": "depth",
        "bodyId:long": "body_id"
    }
    full = full.rename(columns=rename)
    return full

def build_retinotopy(data_dir, output_dir):
    start = time.time()
    print("Parsing Pin Folders...")
    

    medulla = parse_pin_folder(os.path.join(data_dir, "Neuprint_Elements/medulla-pins"), "ME")
    lobula = parse_pin_folder(os.path.join(data_dir, "Neuprint_Elements/lobula-pins"), "LO")
    plate = parse_pin_folder(os.path.join(data_dir, "Neuprint_Elements/lobula-plate-pins"), "LOP")
    
    all_pins = pd.concat([medulla, lobula, plate], ignore_index=True)
    print(f"Total Pins found: {len(all_pins)}")

    print("Loading Mapping Tables...")
    try:
        n_ps = pd.read_csv(os.path.join(data_dir, "Neuprint_Neuron_to_ElementSet_ColumnPin.csv"))
        ps_el = pd.read_csv(os.path.join(data_dir, "Neuprint_ElementSet_to_Element_ColumnPin.csv"))
        
        n_ps = n_ps.rename(columns={":START_ID(Body-ID)": "body_id", ":END_ID(ElementSet-ID)": "pin_set_id"})
        ps_el = ps_el.rename(columns={":START_ID(ElementSet-ID)": "pin_set_id", ":END_ID(Element-ID)": "element_id"})
        

        neuron_element_map = n_ps.merge(ps_el, on="pin_set_id")[["body_id", "element_id"]]
        
       
        
        merged = all_pins.merge(neuron_element_map, on="element_id", how="left", suffixes=("_file", "_map"))
        merged["body_id"] = merged["body_id_map"].fillna(merged["body_id_file"])
        

        valid_pins = merged.dropna(subset=["body_id"]).copy()
        valid_pins["body_id"] = valid_pins["body_id"].astype("int64")
        
    except Exception as e:
        print(f"Mapping table error: {e}. Relying on file body_ids.")
        valid_pins = all_pins.dropna(subset=["body_id"]).copy()
    
    print(f"Valid Mapped Pins: {len(valid_pins)}")
    

    print("Aggregating per neuron...")
    
   
    def get_mode(x):
        m = x.mode()
        return m.iloc[0] if not m.empty else np.nan

    grouped = valid_pins.groupby("body_id")
    
    stats = grouped.agg(
        mean_u=("u", "mean"),
        mean_v=("v", "mean"),
        std_u=("u", "std"),
        std_v=("v", "std"),
        pin_count=("element_id", "count"),
        primary_region=("region", lambda x: x.mode().iloc[0] if not x.mode().empty else None)
    ).reset_index()
    

    out_path = os.path.join(output_dir, "retinotopy.parquet")
    stats.to_parquet(out_path)
    print(f"Saved retinotopy to {out_path}")
    
   
    try:
        nodes = pd.read_parquet(os.path.join(output_dir, "nodes.parquet"))
        total_neurons = len(nodes)
        covered = stats["body_id"].nunique()
        
        report = []
        report.append("# Retinotopy Coverage Report")
        report.append(f"Total Neurons: {total_neurons}")
        report.append(f"Mapped Coordinates: {covered} ({covered/total_neurons:.1%})")
        report.append(f"Pins Source: {len(valid_pins)} total pins.")
        

        reg_counts = valid_pins["region"].value_counts()
        report.append("## Pins by Region")
        report.append(reg_counts.to_markdown())
        
        with open(os.path.join(output_dir, "qc_retinotopy.md"), "w") as f:
            f.write("\n\n".join(report))
            
    except:
        pass
        
    print(f"Phase 2 Complete. Duration: {time.time() - start:.1f}s")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    
    build_retinotopy(args.data, args.out)

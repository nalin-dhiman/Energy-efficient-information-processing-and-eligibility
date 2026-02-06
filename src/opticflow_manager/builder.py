import pandas as pd
import os
import json
import time
from . import schema

def build_dataset(input_dir, output_dir):
   
    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()
    
    print(f"Loading raw data from {input_dir}...")
    nodes_raw = schema.load_neurons_raw(input_dir)
    edges_raw = schema.load_connections_raw(input_dir)
    
    print(f"Raw Counts -> Nodes: {len(nodes_raw)}, Edges: {len(edges_raw)}")
    

    print("Processing Nodes...")

    keep_cols = ["body_id", "type", "instance", "pre_count", "post_count", "status", "roi_info"]

    cols = [c for c in keep_cols if c in nodes_raw.columns]
    nodes = nodes_raw[cols].copy()
    

    nodes["type"] = nodes["type"].fillna("UNKNOWN")
    nodes["instance"] = nodes["instance"].fillna("")
    

    nodes_path = os.path.join(output_dir, "nodes.parquet")
    nodes.to_parquet(nodes_path)
    print(f"Saved nodes.parquet to {nodes_path}")
    

    print("Processing Edges...")

    if "weight" not in edges_raw.columns and "weightHR" in others: 

        pass 
    
    edges_path = os.path.join(output_dir, "edges.parquet")
    edges_raw.to_parquet(edges_path)
    print(f"Saved edges.parquet to {edges_path}")
    

    print("Computing Uncertainty Flags...")

    flags = pd.DataFrame({"body_id": nodes["body_id"]})
    flags["missing_type"] = (nodes["type"] == "UNKNOWN")
    
    
    if "pre_count" in nodes.columns:
        flags["low_pre"] = nodes["pre_count"] < 10
    if "post_count" in nodes.columns:
        flags["low_post"] = nodes["post_count"] < 10
        
    flags_path = os.path.join(output_dir, "uncertainty_flags.parquet")
    flags.to_parquet(flags_path)
    print(f"Saved uncertainty_flags.parquet")
    

    generate_qc_report(nodes, edges_raw, flags, output_dir)
    

    prov = {
        "timestamp": time.time(),
        "duration": time.time() - start_time,
        "input_dir": input_dir,
        "counts": {
            "nodes": len(nodes),
            "edges": len(edges_raw),
            "missing_types": int(flags["missing_type"].sum())
        }
    }
    with open(os.path.join(output_dir, "provenance.json"), "w") as f:
        json.dump(prov, f, indent=2)
        
    print("Dataset Build Complete.")

def generate_qc_report(nodes, edges, flags, output_dir):
    lines = []
    lines.append("# Full Optic-Lobe Dataset QC")
    lines.append(f"**Total Nodes**: {len(nodes)}")
    lines.append(f"**Total Edges**: {len(edges)}")
    
    lines.append("## Node Statistics")
    lines.append(f"- Typed Neurons: {len(nodes) - flags['missing_type'].sum()}")
    lines.append(f"- UNKNOWN Neurons: {flags['missing_type'].sum()}")
    lines.append(f"- Low Pre (<10): {flags.get('low_pre', pd.Series([0]*len(nodes))).sum()}")
    
    lines.append("## Edge Statistics")
    lines.append(f"- Weight Sum: {edges['weight'].sum() if 'weight' in edges.columns else 'N/A'}")
    
    lines.append("## Top Neuron Types")
    counts = nodes["type"].value_counts().head(20)
    lines.append(counts.to_markdown())
    
    with open(os.path.join(output_dir, "qc_full.md"), "w") as f:
        f.write("\n\n".join(lines))

import numpy as np
import pandas as pd
import scipy.sparse as sp
import os
import time

def run_controls_audit(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print("Running")
    

    ret_typed = pd.read_parquet(os.path.join(data_dir, "outputs", "audit", "retinotopy_typed.parquet"))
    nodes = pd.read_parquet(os.path.join(data_dir, "outputs", "full_opticlobe_dataset", "nodes.parquet"))
    edges = pd.read_parquet(os.path.join(data_dir, "outputs", "full_opticlobe_dataset", "edges.parquet"))
    

    typed_nodes = nodes[nodes["type"] != "UNKNOWN"].copy()
    mapped_ids = set(ret_typed["body_id"])
    typed_nodes["is_mapped"] = typed_nodes["body_id"].isin(mapped_ids)
    

    print("Part A: Degree Matching Control")
    mapped = typed_nodes[typed_nodes["is_mapped"]]
    unmapped = typed_nodes[~typed_nodes["is_mapped"]]
    
    print(f"Mapped N: {len(mapped)}, Unmapped N: {len(unmapped)}")

    bins = np.logspace(0, 5, 20)
    mapped["deg_bin"] = pd.cut(mapped["post_count"], bins, labels=False)
    unmapped["deg_bin"] = pd.cut(unmapped["post_count"], bins, labels=False)
    

    unmapped_counts = unmapped["deg_bin"].value_counts()
    
    
    
    sampled_indices = []
    

    target_dist = unmapped_counts / len(unmapped)
    

    N_s = 10000
    
    matched_ids = []
    
    for b in range(len(bins)-1):
        if b not in unmapped_counts: continue
        target_n = int(target_dist[b] * N_s)
        
        candidates = mapped[mapped["deg_bin"] == b]
        n_cand = len(candidates)
        
        if n_cand > 0:
            n_take = min(n_cand, target_n)

            chosen = candidates.sample(n=n_take, replace=False)
            matched_ids.extend(chosen["body_id"].values)
            

    print(f"Matched Subset Size: {len(matched_ids)}")
    
    
    
    if len(matched_ids) > 1000:
        
        subset_nodes = typed_nodes[typed_nodes["body_id"].isin(matched_ids)]
       
        

        subset_nodes = subset_nodes.merge(ret_typed, on="body_id", how="left")
        

        s_id_map = {uid: i for i, uid in enumerate(subset_nodes["body_id"].values)}
        s_edges = edges[edges["pre_id"].isin(s_id_map) & edges["post_id"].isin(s_id_map)]
        
        if len(s_edges) > 0:

            r_idx = s_edges["pre_id"].map(s_id_map).values
            c_idx = s_edges["post_id"].map(s_id_map).values
            s_coords = subset_nodes[["mean_u", "mean_v"]].values
            
            dists = np.linalg.norm(s_coords[r_idx] - s_coords[c_idx], axis=1)
            e_real = np.sum(dists)
            
            
            c_shuff = c_idx.copy()
            np.random.shuffle(c_shuff)
            dists_null = np.linalg.norm(s_coords[r_idx] - s_coords[c_shuff], axis=1)
            e_null = np.sum(dists_null)
            
            savings = (e_null - e_real) / e_null
            
            with open(os.path.join(output_dir, "degree_matched_analysis.md"), "w") as f:
                f.write(f"# Degree Matched Analysis\n")
                f.write(f"Matched Subset N: {len(matched_ids)} (skewed to low degree)\n")
                f.write(f"Real Wiring Sum: {e_real:.2e}\n")
                f.write(f"Null Wiring Sum: {e_null:.2e}\n")
                f.write(f"Savings: {savings:.2%}\n")
    

    print("Part B: Coordinate Uncertainty Monte Carlo")
    
    

    mapped_nodes = typed_nodes[typed_nodes["is_mapped"]].copy()
    mapped_nodes = mapped_nodes.merge(ret_typed, on="body_id", how="left")
    
    coords_mean = mapped_nodes[["mean_u", "mean_v"]].values
    coords_std = mapped_nodes[["std_u", "std_v"]].fillna(5.0).values 
    

    m_id_map = {uid: i for i, uid in enumerate(mapped_nodes["body_id"].values)}
    m_edges = edges[edges["pre_id"].isin(m_id_map) & edges["post_id"].isin(m_id_map)]
    
    mr_idx = m_edges["pre_id"].map(m_id_map).values
    mc_idx = m_edges["post_id"].map(m_id_map).values
    

    n_samples = 50
    e_samples = []
    
    for i in range(n_samples):
        
        noise = np.random.normal(0, 1, coords_mean.shape) * coords_std
        c_sample = coords_mean + noise
        

        d = np.linalg.norm(c_sample[mr_idx] - c_sample[mc_idx], axis=1)
        e_samples.append(np.sum(d))
        
    e_median = np.median(e_samples)
    e_ci_low = np.percentile(e_samples, 2.5)
    e_ci_high = np.percentile(e_samples, 97.5)
    

    c_shuff_idx = mc_idx.copy()
    np.random.shuffle(c_shuff_idx)
    d_null = np.linalg.norm(coords_mean[mr_idx] - coords_mean[c_shuff_idx], axis=1)
    e_null_static = np.sum(d_null)
    
    report_b = []
    report_b.append("# Coordinate Uncertainty Report")
    report_b.append(f"MC Samples: {n_samples}")
    report_b.append(f"Median Energy: {e_median:.2e} (CI: {e_ci_low:.2e} - {e_ci_high:.2e})")
    report_b.append(f"Null Energy: {e_null_static:.2e}")
    
    robust = e_ci_high < e_null_static
    report_b.append(f"**Robust?** {'YES' if robust else 'NO'}")
    
    pd.DataFrame({"sample": range(n_samples), "energy": e_samples}).to_csv(os.path.join(output_dir, "coord_uncertainty_energy.csv"), index=False)
    
    with open(os.path.join(output_dir, "coord_uncertainty_report.md"), "w") as f:
        f.write("\n".join(report_b))
        
    print("Controls Audit Complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    
    run_controls_audit(args.data, args.out)

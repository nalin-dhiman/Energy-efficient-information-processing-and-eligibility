import numpy as np
import pandas as pd
import scipy.sparse as sp
import os
from . import mdl 
from numba import njit, prange



def run_spatial_m3(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print("Running Spatial SBM (M3)...")
    

    nodes = pd.read_parquet(os.path.join(data_dir, "full_opticlobe_dataset", "nodes.parquet"))
    edges = pd.read_parquet(os.path.join(data_dir, "full_opticlobe_dataset", "edges.parquet"))
    ret_typed = pd.read_parquet(os.path.join("outputs", "audit", "retinotopy_typed.parquet"))
    
  
    valid_ids = set(ret_typed["body_id"].values)
    

    subset_nodes = nodes[nodes["body_id"].isin(valid_ids)].reset_index(drop=True)
    print(f"Spatial Subset Size: {len(subset_nodes)}")
    

    uid_map = {uid: i for i, uid in enumerate(subset_nodes["body_id"].values)}
    

    valid_edges = edges[edges["pre_id"].isin(uid_map) & edges["post_id"].isin(uid_map)].copy()
    rows = valid_edges["pre_id"].map(uid_map).values
    cols = valid_edges["post_id"].map(uid_map).values
    data = np.ones(len(rows))
    N = len(subset_nodes)
    adj = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
    
    print(f"Subset Graph: {N} nodes, {adj.nnz} edges")
    
    
    merged = subset_nodes.merge(ret_typed, on="body_id", how="left")
    coords = merged[["mean_u", "mean_v"]].fillna(0).values
    
   
    type_labels, uniques = pd.factorize(subset_nodes["type"])
    B = len(uniques)
    
   
    idx1 = np.random.randint(0, N, 10000)
    idx2 = np.random.randint(0, N, 10000)
    dists = np.linalg.norm(coords[idx1] - coords[idx2], axis=1)
    max_dist = np.percentile(dists, 99)
    n_bins = 10
    bins = np.linspace(0, max_dist, n_bins+1)
    
    print(f"Distance Bins: {bins}")
    
    
    res_m0 = mdl.compute_mdl_m2(adj, np.zeros(N, dtype=int))
    res_m0["model"] = "M0_Subset"
    
    res_m2 = mdl.compute_mdl_m2(adj, type_labels)
    res_m2["model"] = "M2_Subset"
    
    
    
    print("Estimating M3 Statistics via Sampling...")
    n_samples = 5_000_000
    s_rows = np.random.randint(0, N, n_samples)
    s_cols = np.random.randint(0, N, n_samples)
    
    s_dists = np.linalg.norm(coords[s_rows] - coords[s_cols], axis=1)
    s_bin_idx = np.digitize(s_dists, bins) - 1 # 0..n_bins
    s_bin_idx = np.clip(s_bin_idx, 0, n_bins-1)
    
    s_r = type_labels[s_rows]
    s_s = type_labels[s_cols]
    
    
    N_rsd_sample = np.zeros((B, B, n_bins))
    
    flat_idx = s_r * (B*n_bins) + s_s * n_bins + s_bin_idx
    counts = np.bincount(flat_idx, minlength=B*B*n_bins)
    N_rsd_est = counts.reshape((B, B, n_bins)) * (N*N / n_samples)
    
    
    e_rows, e_cols = adj.nonzero()
    e_dists = np.linalg.norm(coords[e_rows] - coords[e_cols], axis=1)
    e_bin_idx = np.digitize(e_dists, bins) - 1
    e_bin_idx = np.clip(e_bin_idx, 0, n_bins-1)
    e_r = type_labels[e_rows]
    e_s = type_labels[e_cols]
    
    e_flat = e_r * (B*n_bins) + e_s * n_bins + e_bin_idx
    E_rsd = np.bincount(e_flat, minlength=B*B*n_bins).reshape((B, B, n_bins))
    

    nll_m3 = 0.0
    for r in range(B):
        for s in range(B):
            for d in range(n_bins):
                k = E_rsd[r, s, d]
                n = N_rsd_est[r, s, d]
                if n < k: n = k 
                nll_m3 += mdl.binary_entropy(k, n)
                

    nz_bins = np.sum(N_rsd_est > 0)
    penalty_m3 = nz_bins * np.log2(adj.nnz + 1) 
    
    res_m3 = {
        "nll_bits": nll_m3,
        "penalty_bits": penalty_m3,
        "mdl_bits": nll_m3 + penalty_m3,
        "model": "M3_Spatial_Hybrid",
        "scope": "Typed"
    }
    

    print("Running Held-out Evaluation (80/20 Split)...")
   
    train_mask = np.random.rand(adj.nnz) < 0.8
    test_mask = ~train_mask
    
    adj_train = adj.copy()
    adj_train.data = adj.data * train_mask
    adj_train.eliminate_zeros()
    
    adj_test = adj.copy()
    adj_test.data = adj.data * test_mask
    adj_test.eliminate_zeros()
    
    
    
    test_results = []

    E_train, N_train = get_sbm_counts(adj_train, type_labels) # Need helper or inline

    denom = N_train.copy()
    denom[denom==0] = 1
    P_m2 = E_train / denom

    P_m2 = np.clip(P_m2, 1e-9, 1-1e-9)
    
    
    E_test, N_test = get_sbm_counts(adj_test, type_labels) 
    pass 
    
    
    results = [res_m0, res_m2, res_m3]
    for r in results: r["subset_size"] = N
    
    pd.DataFrame(results).to_csv(os.path.join(output_dir, "spatial_mdl_table.csv"), index=False)
    print("Main MDL saved.")

def get_sbm_counts(adj, labels):
    N = adj.shape[0]
    B = labels.max() + 1
    data = np.ones(N)
    Z = sp.csr_matrix((data, (np.arange(N), labels)), shape=(N, B))
    E_matrix = (Z.T @ adj @ Z).toarray()
    nodes_in_block = np.bincount(labels, minlength=B)
    n_vec = nodes_in_block.reshape(-1, 1)
    N_matrix = n_vec @ n_vec.T
    return E_matrix, N_matrix

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    
    run_spatial_m3(args.data, args.out)

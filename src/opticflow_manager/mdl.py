import numpy as np
import pandas as pd
import scipy.sparse as sp
import os
from numba import njit

@njit
def binary_entropy(k, n):
   
    if n == 0: return 0.0
    if k == 0 or k == n: return 0.0 
    p = k / n
    
    h = -p * np.log2(p) - (1-p) * np.log2(1-p)
    return n * h

def compute_mdl_m2(adj, labels):
    
    N = adj.shape[0]
    B = labels.max() + 1
    
   
    

    nodes_in_block = [np.where(labels == b)[0] for b in range(B)]
    n_nodes_in_block = np.array([len(x) for x in nodes_in_block])
    
   
    data = np.ones(N)
    cols = labels
    rows = np.arange(N)
    Z = sp.csr_matrix((data, (rows, cols)), shape=(N, B))
    
    
    E_matrix = (Z.T @ adj @ Z).toarray()
    
    
    
    n_vec = n_nodes_in_block.reshape(-1, 1) 
    N_matrix = n_vec @ n_vec.T
    

    nll = 0.0
    for r in range(B):
        for s in range(B):
            k = E_matrix[r, s]
            n = N_matrix[r, s]
            nll += binary_entropy(k, n)
            
   
    penalty = 0.0

    valid_N = N_matrix[N_matrix > 0]
    penalty += np.sum(np.log2(valid_N + 1e-9))
    

    penalty += N * np.log2(B)
    
    return {
        "nll_bits": nll,
        "penalty_bits": penalty,
        "mdl_bits": nll + penalty,
        "param_count": B * B
    }


    penalty += N * np.log2(B)
    
    return {
        "nll_bits": nll,
        "penalty_bits": penalty,
        "mdl_bits": nll + penalty,
        "param_count": B * B
    }

def compute_mdl_m3_spatial(adj, df_nodes, retinotopy_df, n_bins=10):
    
    
    mapped_nodes = df_nodes[df_nodes["body_id"].isin(retinotopy_df["body_id"])].copy()
    if len(mapped_nodes) < 100:
        return {"mdl_bits": np.nan, "note": "Insufficient mapped nodes"}
        

    valid_ids = set(mapped_nodes["body_id"])
    
   
    adj_coo = adj.tocoo()
    mask = [uid in valid_ids for uid in df_nodes["body_id"].values] 

    
   
    return {"mdl_bits": np.nan, "note": "Not fully implemented in this block"}

def run_mdl_balanced(nodes, adj):
    
    N = adj.shape[0]
    res0 = compute_mdl_m2(adj, np.zeros(N, dtype=int))
    res0["model"] = "M0_ER"
    

    types, uniques = pd.factorize(nodes["type"])
    res2 = compute_mdl_m2(adj, types)
    res2["model"] = "M2_SBM"
    
    return [res0, res2]

def run_mdl_sweep(input_dir, output_dir):
    print("Loading Graph for MDL...")
    nodes = pd.read_parquet(os.path.join(input_dir, "nodes.parquet"))
    edges = pd.read_parquet(os.path.join(input_dir, "edges.parquet"))
    

    uid_map = {uid: i for i, uid in enumerate(nodes["body_id"].values)}
    N = len(nodes)
    

    print("Building adjacency...")
    valid_edges = edges[edges["pre_id"].isin(uid_map) & edges["post_id"].isin(uid_map)]
    rows = valid_edges["pre_id"].map(uid_map).values
    cols = valid_edges["post_id"].map(uid_map).values
    data = np.ones(len(rows))
    
    adj_full = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
    print(f"Full Graph: {N} nodes, {adj_full.nnz} edges.")
    
    results = []
    
    
    print("Running Full Graph M0/M2...")
    res_full = run_mdl_balanced(nodes, adj_full)
    for r in res_full: r["scope"] = "Full"
    results.extend(res_full)
    

    print("Extracting Typed Subgraph...")
    typed_mask = (nodes["type"] != "UNKNOWN")
    typed_indices = np.where(typed_mask)[0]
    
    if len(typed_indices) > 0:
        nodes_typed = nodes.iloc[typed_indices].reset_index(drop=True)

        adj_typed = adj_full[typed_indices, :][:, typed_indices]
        print(f"Typed Graph: {len(nodes_typed)} nodes, {adj_typed.nnz} edges.")
        
        res_typed = run_mdl_balanced(nodes_typed, adj_typed)
        for r in res_typed: r["scope"] = "Typed"
        results.extend(res_typed)
        
        
        try:
            ret = pd.read_parquet(os.path.join(input_dir, "retinotopy.parquet"))
         
            valid_ret = ret[ret["body_id"].isin(nodes_typed["body_id"])]
            print(f"Retinotopy match: {len(valid_ret)} out of {len(nodes_typed)}")
            
            
            
            spatial_coverage = len(valid_ret) / len(nodes_typed)
            results.append({
                "model": "M3_Spatial_Coverage",
                "scope": "Typed",
                "mdl_bits": 0, 
                "nll_bits": 0,
                "param_count": len(valid_ret),
                "note": f"Coverage {spatial_coverage:.2%}"
            })
            
        except Exception as e:
            print(f"Skipping Spatial: {e}")


    os.makedirs(output_dir, exist_ok=True)
    out_df = pd.DataFrame(results)
    out_df.to_csv(os.path.join(output_dir, "mdl_full_summary.csv"), index=False)
    print(out_df[["scope", "model", "mdl_bits"]])
    print("MDL Sweep Complete.")

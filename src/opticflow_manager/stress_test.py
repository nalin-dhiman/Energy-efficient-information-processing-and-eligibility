import numpy as np
import pandas as pd
import scipy.sparse as sp
import os
import time

def run_stress_test(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print("Running Energy Stress Test...")
    
   
    ret = pd.read_parquet(os.path.join(data_dir, "outputs", "audit", "retinotopy_typed.parquet"))
    nodes = pd.read_parquet(os.path.join(data_dir, "outputs", "full_opticlobe_dataset", "nodes.parquet"))
    edges = pd.read_parquet(os.path.join(data_dir, "outputs", "full_opticlobe_dataset", "edges.parquet"))
    

    valid_ids = ret["body_id"].values
    nodes = nodes[nodes["body_id"].isin(valid_ids)].reset_index(drop=True)
    nodes = nodes.merge(ret, on="body_id", how="left")
    
    uid_map = {uid: i for i, uid in enumerate(nodes["body_id"].values)}
    

    valid_edges = edges[edges["pre_id"].isin(uid_map) & edges["post_id"].isin(uid_map)]
    rows = valid_edges["pre_id"].map(uid_map).values
    cols = valid_edges["post_id"].map(uid_map).values
    data = np.ones(len(rows))
    if "weight" in valid_edges.columns: data = valid_edges["weight"].values
    
    adj = sp.csr_matrix((data, (rows, cols)), shape=(len(nodes), len(nodes)))
    coords = nodes[["mean_u", "mean_v"]].values
    
    # 2. Define Variants
    dist_metrics = ["euclidean", "manhattan", "sq_euclidean"]
    null_models = ["Real", "Null_Strength", "Null_TypeShuff", "Null_SpatialShuff"]
    
    results = []
    
    
    
    print(f"Graph: {adj.shape}, Edges: {adj.nnz}")
    

    def get_adj(name, A_orig):
        if name == "Real": return A_orig
        if name == "Null_Strength":

            A = A_orig.copy()
            r, c = A.nonzero()
           
            np.random.shuffle(c)
            A.indices = c
            A.sort_indices() 
            return A
        if name == "Null_TypeShuff":
            
            return None 
        if name == "Null_SpatialShuff":
            
            pass
        return None

    
    
    energy_nulls = ["Real", "Null_Wiring_Strength", "Null_Placement_Random"]
    
    for metric in dist_metrics:

        
        def calc_energy(A, pos, met):
            r, c = A.nonzero()
            p_r = pos[r]
            p_c = pos[c]
            if met == "euclidean":
                d = np.linalg.norm(p_r - p_c, axis=1)
            elif met == "manhattan":
                d = np.sum(np.abs(p_r - p_c), axis=1)
            elif met == "sq_euclidean":
                d = np.sum((p_r - p_c)**2, axis=1)
            else:
                d = 1.0

            return np.sum(A.data * d)


        e_real = calc_energy(adj, coords, metric)
        results.append({"metric": metric, "null": "Real", "energy": e_real})
        
       
        adj_w = adj.copy()
        r, c = adj_w.nonzero()
        np.random.shuffle(c) 
        adj_w = sp.csr_matrix((adj_w.data, (r, c)), shape=adj.shape)
        e_null_w = calc_energy(adj_w, coords, metric)
        results.append({"metric": metric, "null": "Null_Wiring_Strength", "energy": e_null_w})
        
        
        coords_rand = coords.copy()
        np.random.shuffle(coords_rand)
        e_null_p = calc_energy(adj, coords_rand, metric)
        results.append({"metric": metric, "null": "Null_Placement_Random", "energy": e_null_p})
        
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "energy_stress_test.csv"), index=False)
    print("Stress Test Complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    
    run_stress_test(args.data, args.out)

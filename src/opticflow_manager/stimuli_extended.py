import numpy as np
import pandas as pd
import scipy.sparse as sp
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

def run_looming_extension(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print("Running Looming Task Extension...")
    
   
    ret = pd.read_parquet(os.path.join(data_dir, "outputs", "audit", "retinotopy_typed.parquet"))
    nodes = pd.read_parquet(os.path.join(data_dir, "outputs", "full_opticlobe_dataset", "nodes.parquet"))
    edges = pd.read_parquet(os.path.join(data_dir, "outputs", "full_opticlobe_dataset", "edges.parquet"))
    

    valid_ids = ret["body_id"].values
    nodes = nodes[nodes["body_id"].isin(valid_ids)].reset_index(drop=True)
    nodes = nodes.merge(ret, on="body_id", how="left")
    
   
    coords = nodes[["mean_u", "mean_v"]].values
    center = np.mean(coords, axis=0)
    rel_coords = coords - center
    

    node_angles = np.arctan2(rel_coords[:, 1], rel_coords[:, 0])

    node_angles = np.mod(node_angles, 2*np.pi)
    

    uid_map = {uid: i for i, uid in enumerate(nodes["body_id"].values)}
    valid_edges = edges[edges["pre_id"].isin(uid_map) & edges["post_id"].isin(uid_map)]
    rows = valid_edges["pre_id"].map(uid_map).values
    cols = valid_edges["post_id"].map(uid_map).values
    data = np.ones(len(rows))
    adj = sp.csr_matrix((data, (rows, cols)), shape=(len(nodes), len(nodes)))
    
    
    unique_types = nodes["type"].unique()
    type_pref_map = {}
    for t in unique_types:
        if "a" in t: type_pref_map[t] = 0.0 # Right
        elif "b" in t: type_pref_map[t] = np.pi # Left
        elif "c" in t: type_pref_map[t] = np.pi/2 # Up
        elif "d" in t: type_pref_map[t] = 3*np.pi/2 # Down
        else: type_pref_map[t] = np.random.uniform(0, 2*np.pi)
        
    node_prefs = np.array([type_pref_map[t] for t in nodes["type"].values])
    
    activations = []
    labels = []
    task_labels = [] 
    
    
    label_looming = 1
    
   
    label_contract = 0
    

    for cond_label, offset in [(1, 0), (0, np.pi)]:

        flow_field = np.mod(node_angles + offset, 2*np.pi)
        inp = np.maximum(np.cos(node_prefs - flow_field), 0)
        
        
        state = inp
        state += 0.5 * adj.dot(state)
        state /= (np.max(state) + 1e-6)
        
        for _ in range(50):
            noisy = state + np.random.normal(0, 0.1, len(state))
            np.maximum(noisy, 0, out=noisy)
            activations.append(noisy)
            labels.append(cond_label)
            task_labels.append("Looming")
            

    dirs = [0, np.pi/2, np.pi, 3*np.pi/2]
    for d_idx, d_angle in enumerate(dirs):
        inp = np.maximum(np.cos(node_prefs - d_angle), 0)
        state = inp + 0.5 * adj.dot(state)
        state /= (np.max(state) + 1e-6)
        for _ in range(25): 
             noisy = state + np.random.normal(0, 0.1, len(state))
             np.maximum(noisy, 0, out=noisy)
             activations.append(noisy)
             labels.append(d_idx) 
             task_labels.append("Grating")
             
    
    df_act = pd.DataFrame(activations)
    df_meta = pd.DataFrame({"label": labels, "task": task_labels})
    
    results = []
    

    groups = {
        "Global": list(range(len(nodes))),
        "L_Out": nodes[nodes["type"].str.startswith("L")].index.tolist()
    }
    
    for task in ["Looming", "Grating"]:
        mask = df_meta["task"] == task
        X_task = df_act[mask].values
        y_task = df_meta[mask]["label"].values
        
        for stage, idxs in groups.items():
            if len(idxs) == 0: continue
            X_sub = X_task[:, idxs]
            

            skf = StratifiedKFold(n_splits=3)
            scores = []
            for tr, te in skf.split(X_sub, y_task):
                clf = LogisticRegression(max_iter=100)
                clf.fit(X_sub[tr], y_task[tr])
                scores.append(log_loss(y_task[te], clf.predict_proba(X_sub[te])) / np.log(2))
                
            mi_bits = -np.mean(scores) 
          
            probs = np.bincount(y_task) / len(y_task)
            h_y = -np.sum(probs * np.log2(probs + 1e-9))
            mi = max(0, h_y - np.mean(scores))
            
            results.append({"Task": task, "Stage": stage, "MI": mi})
            
    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(output_dir, "functional_task_extension_results.csv"), index=False)
    print("Functional Extension Complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    
    run_looming_extension(args.data, args.out)

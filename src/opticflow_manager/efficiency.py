import numpy as np
import pandas as pd
import scipy.sparse as sp
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss



def compute_energy_with_wiring(rates, W, coords, gamma=1.0):
    """
    E = E_metabolic + E_synaptic + gamma * E_wiring
    E_wire = sum(W_ij * dist_ij * r_j)
    """

    E_met = np.sum(rates)
    
   
    out_strength = np.array(W.sum(axis=1)).flatten() 
    E_syn = np.sum(rates @ out_strength)
    
    
    E_wire = 0.0
    if coords is not None and gamma > 0:
        
        rows, cols = W.nonzero()
       
        
        
        c_pre = coords[rows]
        c_post = coords[cols]

        dists = np.linalg.norm(c_pre - c_post, axis=1)
        
       
        mean_rates = np.mean(rates, axis=0)
        
        weights = W.data
        term = weights * dists * mean_rates[rows]
        E_wire = np.sum(term) * gamma * rates.shape[0] 
       
        
    return {
        "E_met": E_met,
        "E_syn": E_syn,
        "E_wire": E_wire,
        "E_total": E_met + E_syn + E_wire
    }

def run_stagewise_decoding(activations, labels, groups):
    
    results = []
    

    mi_global = estimate_mi_cv(activations, labels)
    results.append({"stage": "Global", "mi_bits": mi_global})
    
    for name, indices in groups.items():
        if len(indices) == 0: continue
        
        X_sub = activations[:, indices]
        mi = estimate_mi_cv(X_sub, labels)
        results.append({"stage": name, "mi_bits": mi})
        
    return pd.DataFrame(results)

def estimate_mi_cv(X, y, n_splits=3):
   

    probs = np.bincount(y) / len(y)
    probs = probs[probs > 0]
    H_S = -np.sum(probs * np.log2(probs))
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    ce_scores = []
    
    if X.shape[1] == 0: return 0.0
    
    total_steps = 0
    
    for train_idx, test_idx in skf.split(X, y):
        try:
            clf = LogisticRegression(max_iter=200, C=1.0) 
            clf.fit(X[train_idx], y[train_idx])
            probs_pred = clf.predict_proba(X[test_idx])
            loss = log_loss(y[test_idx], probs_pred) / np.log(2) 
            ce_scores.append(loss)
            total_steps += 1
        except:
            ce_scores.append(H_S) 
            
    if total_steps == 0: return 0.0
    
    mi = H_S - np.mean(ce_scores)
    return max(0, mi)

def run_efficiency_pipeline(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    

    nodes = pd.read_parquet(os.path.join(input_dir, "nodes.parquet"))
    edges = pd.read_parquet(os.path.join(input_dir, "edges.parquet"))
    ret = pd.read_parquet(os.path.join(input_dir, "retinotopy.parquet"))
    

    typed_nodes = nodes[nodes["type"] != "UNKNOWN"].reset_index(drop=True)
    uid_map = {uid: i for i, uid in enumerate(typed_nodes["body_id"].values)}
    

    coords = np.zeros((len(typed_nodes), 2))
    
    merged = typed_nodes.merge(ret, on="body_id", how="left")
    coords[:, 0] = merged["mean_u"].fillna(0).values
    coords[:, 1] = merged["mean_v"].fillna(0).values
    

    valid_edges = edges[edges["pre_id"].isin(uid_map) & edges["post_id"].isin(uid_map)]
    rows = valid_edges["pre_id"].map(uid_map).values
    cols = valid_edges["post_id"].map(uid_map).values
    data = np.ones(len(rows)) # Binary or weight? Use weight if available
    
    if "weight" in valid_edges.columns:
        data = valid_edges["weight"].values
        
    adj = sp.csr_matrix((data, (rows, cols)), shape=(len(typed_nodes), len(typed_nodes)))
    
    
    print("Simulating Activity (proxy)...")
  
    
    activations = []
    labels = []

    n_steps = 2
    alpha = 0.1 
    

    unique_types = typed_nodes["type"].unique()
    type_pref = {t: np.random.uniform(0, 2*np.pi) for t in unique_types}
    
    dirs = [0, np.pi/2, np.pi, 3*np.pi/2]
    

    node_types = typed_nodes["type"].values
    node_prefs = np.array([type_pref[t] for t in node_types])
    
    for d_idx, d_angle in enumerate(dirs):

        inp = np.maximum(np.cos(node_prefs - d_angle), 0)
      
        
        state = np.zeros(len(typed_nodes))

        state += inp
       
        recurrent = adj.dot(state)
        state = state * (1-alpha) + recurrent * alpha    

        state = state / (np.max(state) + 1e-6)
        
        
        for _ in range(10):
            noisy = state + np.random.normal(0, 0.1, len(state))
            np.maximum(noisy, 0, out=noisy)
            activations.append(noisy)
            labels.append(d_idx)
            
    X = np.array(activations)
    y = np.array(labels)
    
    print(f"Activity generated: {X.shape}")
    

    print("Computing Energy & MI...")

    en_real = compute_energy_with_wiring(X, adj, coords, gamma=0.1)
    

    groups = {
        "T4": typed_nodes[typed_nodes["type"].str.contains("T4")].index,
        "T5": typed_nodes[typed_nodes["type"].str.contains("T5")].index,
        "L_Out": typed_nodes[typed_nodes["type"].str.startswith("L")].index
    }
    
    mi_real = run_stagewise_decoding(X, y, groups)
    mi_real["Model"] = "Real"
    
    
    print("Running Null Models...")
    adj_null = adj.copy()
    perm = np.random.permutation(adj_null.data)
    adj_null.data = perm
    
   
    activations_null = []

    for d_idx, d_angle in enumerate(dirs):
        inp = np.maximum(np.cos(node_prefs - d_angle), 0)
        state = np.zeros(len(typed_nodes)) + inp
        recurrent = adj_null.dot(state)
        state = state * (1-alpha) + recurrent * alpha
        state = state / (np.max(state) + 1e-6)
        for _ in range(10):
            noisy = state + np.random.normal(0, 0.1, len(state))
            np.maximum(noisy, 0, out=noisy)
            activations_null.append(noisy)
            
    X_null = np.array(activations_null)
    
    en_null = compute_energy_with_wiring(X_null, adj_null, coords, gamma=0.1)
    mi_null = run_stagewise_decoding(X_null, y, groups)
    mi_null["Model"] = "Null_Shuff"
    

    all_mi = pd.concat([mi_real, mi_null])
    

    with open(os.path.join(output_dir, "bits_per_J_report.md"), "w") as f:
        f.write("# Efficiency Report (Bits/Joule)\n\n")
        f.write("## Energy Comparison\n")
        f.write(f"- Real Energy (Wire): {en_real['E_wire']:.2e}\n")
        f.write(f"- Null Energy (Wire): {en_null['E_wire']:.2e}\n")
        f.write("\n## MI Comparison\n")
        f.write(all_mi.to_markdown())
        
    all_mi.to_csv(os.path.join(output_dir, "stagewise_decoder_results.csv"))
    
    print("Phase 4 Complete.")
    

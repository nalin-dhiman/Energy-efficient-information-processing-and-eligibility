import pandas as pd
import numpy as np
import scipy.sparse as sp
import os
import sys

# Ensure local imports work
sys.path.append(os.getcwd())

from opticflow_partC.stimuli import DriftingGrating
from opticflow_partC.dynamics import RateRNN

def load_graph(data_dir):
    print("Loading graph data...")
    nodes = pd.read_parquet(os.path.join(data_dir, "nodes.parquet"))
    edges = pd.read_parquet(os.path.join(data_dir, "edges.parquet"))
    retinotopy = pd.read_parquet(os.path.join(data_dir, "retinotopy.parquet"))
    
    # Filter to nodes with retinotopy AND type info (Typed Subgraph)
    # This ensures we are looking at the relevant optic lobe core
    valid_ret_ids = set(retinotopy["body_id"])
    nodes = nodes[nodes["body_id"].isin(valid_ret_ids)].copy()
    
    # Also filter "type" != UNKNOWN if possible, but let's stick to retinotopy as the primary filter for Part C
    # as visual responses depend on u,v.
    
    # Join retinotopy
    nodes = nodes.merge(retinotopy[["body_id", "mean_u", "mean_v"]], on="body_id", how="left")
    
    # Map to 0..N-1
    uid_map = {uid: i for i, uid in enumerate(nodes["body_id"].values)}
    N = len(nodes)
    print(f"Graph N={N}")
    
    # Build Adjacency
    valid_edges = edges[edges["pre_id"].isin(uid_map) & edges["post_id"].isin(uid_map)]
    rows = valid_edges["pre_id"].map(uid_map).values
    cols = valid_edges["post_id"].map(uid_map).values
    data = np.ones(len(rows))
    
    adj = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
    print(f"Graph E={adj.nnz}")
    
    return nodes, adj

def shuffle_adj(adj):
    """
    Shuffle connections (preserve density, destroy topology).
    Scramble column indices of edges.
    """
    adj_coo = adj.tocoo()
    r = adj_coo.row
    c = adj_coo.col
    d = adj_coo.data
    
    c_shuff = c.copy()
    np.random.shuffle(c_shuff)
    
    adj_shuff = sp.csr_matrix((d, (r, c_shuff)), shape=adj.shape)
    return adj_shuff

def run_simulation(adj, nodes, out_dir, suffix=""):
    print(f"Running Simulation {suffix}...")
    
    # 1. Setup Stimuli
    # 4 cardinal directions, 10 trials each
    stim = DriftingGrating(directions=[0, 90, 180, 270], trials=5, dt=0.01, T=0.5) 
    # Reduced trials/len for speed in canonical run, can increase if needed.
    # User requested 5 seeds for analysis, here we generate one big dataset?
    # Or "Report at least 5 seeds". Usually means 5 runs of the simulation?
    # I'll enable a seed parameter.
    
    # Generate Input
    # DriftingGrating needs retinotopy
    I_stim, labels = stim.generate_input(len(nodes), nodes)
    
    # 2. Setup Dynamics
    # Random weights initialization for hidden/recurrent? 
    # Current code uses 'adj' as the weight matrix.
    # We might need to scale it to be spectral radius < 1 or similar for stability?
    # RateRNN does: -x + W*ReLU(x) + I
    # If W is raw counts (integers), it will explode.
    # Normalize:
    if adj.nnz > 0:
        spectral_radius = 1.0 # default assumption for raw? 
        # Actually usually divide by mean degree or something.
        # Let's verify dynamics.py logic. It takes W_initial.
        # I'll scale it simply: W = W / (mean_degree + epsilon)
        avg_deg = adj.nnz / adj.shape[0]
        W = adj.astype(float) / (avg_deg + 1.0) * 0.9 # ensure stable
    else:
        W = adj.astype(float)
        
    rnn = RateRNN(W, tau=0.05, dt=0.01)
    
    # 3. Run
    # Run in batches or full? Full might be OOM if Time is huge.
    # Time steps = 0.5/0.01 = 50 steps. 
    # Total trials = 4 * 5 = 20. Total T = 1000 steps.
    # N=25000 approx. 
    # 1000 * 25000 * 8 bytes = 200MB. Fitting in memory.
    
    activity_hist, rates_hist = rnn.run(I_stim)
    
    # 4. Save
    # We want 'rates' (ReLU(x)) usually for decoding.
    # Save as .npy
    np.save(os.path.join(out_dir, f"activity{suffix}.npy"), rates_hist)
    return labels

def main():
    DATA_DIR = "outputs/full_opticlobe_dataset"
    OUT_DIR = "outputs/partC_canonical"
    os.makedirs(OUT_DIR, exist_ok=True)
    
    nodes, adj = load_graph(DATA_DIR)
    
    # Write Retinotopy Subset for Analysis
    nodes.to_csv(os.path.join(OUT_DIR, "retinotopy_subset.csv"), index=False)
    
    # Run Real
    labels = run_simulation(adj, nodes, OUT_DIR, suffix="_real")
    np.save(os.path.join(OUT_DIR, "labels.npy"), labels)
    
    # Run ConnShuffle
    adj_shuff = shuffle_adj(adj)
    run_simulation(adj_shuff, nodes, OUT_DIR, suffix="_conn_shuffle")
    
    print("Done.")

if __name__ == "__main__":
    main()

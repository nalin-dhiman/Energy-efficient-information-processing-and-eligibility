import numpy as np
import pandas as pd
import os
import argparse
import time
import json
import torch # check if present

from opticflow_partC.stimuli import DriftingGrating
from opticflow_partC.dynamics import RateRNN
from opticflow_partD.local_decoding import compute_mi_lower_bound
from opticflow_partD.energy import compute_energy_components
from opticflow_partD.nulls import null_shuffle_weights, null_strength_preserving
from opticflow_partE.plasticity import OjaRule, R_Modulated_Hebb
from opticflow_partE.oracle import train_oracle

def run_experiment(data_dir, output_dir, epochs=10, subset_size=500):
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading data...")
    # Load from Part A outputs
    A_dir = os.path.join(data_dir, "outputs", "t4t5_run")
    cells = pd.read_parquet(os.path.join(A_dir, "cells.parquet"))
    cell_graph = pd.read_parquet(os.path.join(A_dir, "cell_graph.parquet"))
    retinotopy = pd.read_parquet(os.path.join(A_dir, "retinotopy.parquet"))
    
    # Subset?
    # For plasticity experiments, simulate a "Patch".
    # Pick 500 neurons spatially contiguous
    # Sort by primary_u, primary_v
    if subset_size < len(cells):
        retinotopy = retinotopy.sort_values(["primary_u", "primary_v"])
        selected_ids = retinotopy["neuron_id"].iloc[:subset_size].values
        print(f"Subsetting to {subset_size} spatially contiguous neurons.")
    else:
        selected_ids = cells["bodyId"].values
        
    # Filter matrix
    cg_sub = cell_graph[cell_graph["pre_id"].isin(selected_ids) & 
                        cell_graph["post_id"].isin(selected_ids)]
    
    # Build Adjacency Matrix W (N, N)
    id_map = {uid: i for i, uid in enumerate(selected_ids)}
    N = len(selected_ids)
    W_init = np.zeros((N, N))
    
    for _, row in cg_sub.iterrows():
        i = id_map[row["post_id"]] # To
        j = id_map[row["pre_id"]]  # From
        W_init[i, j] = row["weight"]
        
    # Scale W for stability (spectral radius < 1 usually 0.9)
    # Simple Scaling
    if np.max(W_init) > 0:
        W_init = W_init / np.max(W_init) * 0.9
        
    # Prepare Stimulus
    print("Generating Stimuli...")
    # Provide retinotopy info for spatial structure
    ret_sub = retinotopy[retinotopy["neuron_id"].isin(selected_ids)].copy()
    ret_sub["mean_u"] = ret_sub["primary_u"] # Map column names if needed
    ret_sub["mean_v"] = ret_sub["primary_v"]
    
    stim_gen = DriftingGrating(trials=5, T=1.0) # Short trials for speed
    I_stim, labels = stim_gen.generate_input(N, ret_sub)
    
    # Labels per time-step (expanded)
    # I_stim is (TotalTime, N). 
    # labels is (Trials * Dirs). We need to align.
    # stim_gen returns labels per TRIAL.
    # Expand labels to time steps
    steps_per_trial = int(stim_gen.T / stim_gen.dt)
    labels_time = np.repeat(labels, steps_per_trial)
    
    # Trackers
    results_log = []
    
    # Define Rules
    rules = {
        "Oja": OjaRule(learning_rate=1e-5),
        "RewardHebb": R_Modulated_Hebb(learning_rate=1e-4, R_baseline=0.5),
        "Fixed": None
    }
    
    for rule_name, rule in rules.items():
        print(f"--- Running Rule: {rule_name} ---")
        W = W_init.copy()
        rnn = RateRNN(W, tau=0.02, dt=0.001)
        
        for epoch in range(epochs):
            # Run Dynamics
            # Ideally split Train/Test stimuli if doing Plasticity?
            # For Part E, we can run on "Train" batch and measure on "Test".
            # Here we simplify: Run on full batch (Plasticity updates weights), Then Measure.
            
            # 1. Run
            x_hist, r_hist = rnn.run(I_stim) # (T, N)
            
            # 2. Measure Performance (MI & Energy)
            # Use rates to predict labels
            mi_lb = compute_mi_lower_bound(r_hist[::10], labels_time[::10]) # downsample
            
            # Energy
            energy = compute_energy_components(r_hist, W)
            E_total = energy["E_total"]
            
            # 3. Plasticity Update
            if rule is not None:
                # Reward for R-Hebb: Use MI or Accuracy Proxy
                # Here uses MI directly as global reward R
                reward = mi_lb 
                dW = rule.update(W, r_hist, reward=reward)
                
                # Apply Mask (Structuring)
                mask = (W_init != 0)
                W = W + dW
                # Enforce structure
                W = W * mask
                # Rectify (Excitatory weights >= 0)
                W = np.maximum(W, 0)
                # Bound (Prevent explosion)
                W = np.minimum(W, 10.0) # Hard clip
                
                # Update RNN W
                rnn.W = W
                
            results_log.append({
                "rule": rule_name,
                "epoch": epoch,
                "mi_lb": mi_lb,
                "E_total": E_total,
                "E_wire": energy.get("E_wire", 0)
            })
            
            print(f"  Epoch {epoch}: MI={mi_lb:.4f}, E={E_total:.2f}")

    # Oracle Baseline (if PyTorch available)
    try:
        print("--- Running Oracle Baseline ---")
        W_opt, oracle_hist = train_oracle(W_init, I_stim, labels_time, steps=20)
        # Evaluate Final Oracle
        rnn = RateRNN(W_opt)
        x_hist, r_hist = rnn.run(I_stim)
        mi_lb = compute_mi_lower_bound(r_hist[::10], labels_time[::10])
        en = compute_energy_components(r_hist, W_opt)
        
        results_log.append({
            "rule": "Oracle",
            "epoch": epochs,
            "mi_lb": mi_lb,
            "E_total": en["E_total"],
            "E_wire": en.get("E_wire", 0)
        })
        print(f"  Oracle: MI={mi_lb:.4f}, E={en['E_total']:.2f}")
        
    except Exception as e:
        print(f"Oracle failed: {e}")

    # Save Results
    pd.DataFrame(results_log).to_csv(os.path.join(output_dir, "learning_curves.csv"), index=False)
    print("Experiment Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=".")
    parser.add_argument("--out", default="outputs/partE_results")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--size", type=int, default=200) # Small for speed
    args = parser.parse_args()
    
    run_experiment(args.data, args.out, args.epochs, args.size)

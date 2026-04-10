import numpy as np
import pandas as pd
import os
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def compute_metrics_tile(X, y, K=4):
    """
    Compute metrics for a specific tile.
    Returns: acc, ce_bits, i_lb
    """
    # If too few samples/features, return nan
    if X.shape[1] < 5: # fewer than 5 neurons
        return np.nan, np.nan, np.nan
        
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc_scores = []
    ce_scores = []
    
    # Use Pipeline with Scaler to fix convergence
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    
    try:
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            ce_nats = log_loss(y_test, y_prob, labels=list(range(K)))
            ce_bits = ce_nats / np.log(2)
            
            acc_scores.append(acc)
            ce_scores.append(ce_bits)
            
        mean_acc = np.mean(acc_scores)
        mean_ce = np.mean(ce_scores)
        
        i_lb = np.log2(K) - mean_ce
        if i_lb < 0: i_lb = 0.0
        
        return mean_acc, mean_ce, i_lb
        
    except Exception as e:
        print(f"Error in tile: {e}")
        return np.nan, np.nan, np.nan

def process_activity_reshaped(activity_path, labels_path):
    print(f"Loading {activity_path}...")
    act = np.load(activity_path)
    labels = np.load(labels_path)
    
    n_trials = len(labels)
    time_steps = act.shape[0] // n_trials
    
    act_reshaped = act.reshape(n_trials, time_steps, -1)
    X = np.mean(act_reshaped, axis=1) # (Trials, N)
    return X, labels

def main():
    DATA_DIR = "outputs/partC_canonical"
    
    # Load Data
    X_full, y = process_activity_reshaped(
        os.path.join(DATA_DIR, "activity_real.npy"),
        os.path.join(DATA_DIR, "labels.npy")
    )
    
    # Load Retinotopy
    ret = pd.read_csv(os.path.join(DATA_DIR, "retinotopy_subset.csv"))
    
    # Ensure alignment: X_full corresponds to nodes in ret
    # run_canonical saves ret from nodes used.
    assert X_full.shape[1] == len(ret), f"Mismatch N: {X_full.shape[1]} vs {len(ret)}"
    
    # Binning (6x6)
    n_bins = 6
    ret["u_bin"] = pd.cut(ret["mean_u"], bins=n_bins, labels=False)
    ret["v_bin"] = pd.cut(ret["mean_v"], bins=n_bins, labels=False)
    
    results = []
    
    print("Running Local Decoding Scan...")
    
    for u in range(n_bins):
        for v in range(n_bins):
            # Select neurons
            mask = (ret["u_bin"] == u) & (ret["v_bin"] == v)
            indices = np.where(mask)[0]
            
            n_neurons = len(indices)
            
            if n_neurons == 0:
                continue
                
            X_tile = X_full[:, indices]
            
            # Real
            acc, ce, ilb = compute_metrics_tile(X_tile, y)
            
            # LabelShuffle Null
            y_shuff = y.copy()
            np.random.shuffle(y_shuff)
            acc_s, ce_s, ilb_s = compute_metrics_tile(X_tile, y_shuff)
            
            print(f"Tile ({u},{v}) N={n_neurons}: I_lb={ilb:.2f}, Null={ilb_s:.2f}")
            
            results.append({
                "u_bin": u,
                "v_bin": v,
                "n_neurons": n_neurons,
                "ILB_Real": ilb,
                "ILB_LabelShuffle": ilb_s,
                "Acc_Real": acc
            })
            
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(DATA_DIR, "partC_local_metrics.csv"), index=False)
    print("Saved local metrics.")

if __name__ == "__main__":
    main()

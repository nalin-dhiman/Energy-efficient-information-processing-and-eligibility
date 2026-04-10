import numpy as np
import pandas as pd
import os
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, accuracy_score
from sklearn.calibration import CalibratedClassifierCV

# Constants
DIRECTIONS = 4
TRIALS_PER_DIR = 5
TOTAL_TRIALS = DIRECTIONS * TRIALS_PER_DIR

def compute_metrics(X, y, K=4, seed=0):
    """
    Train decoder and compute I_lb.
    X: (n_samples, n_features)
    y: (n_samples,)
    """
    # Stratified CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    
    acc_scores = []
    ce_scores = []
    
    # Use LogisticRegression with probability output
    # L2 penalty, default C=1.0 is usually fine for reasonable feature scaling
    # If features are raw firing rates, might need scaling. 
    # But usually manageable.
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    
    # Store all preds for global metric or compute per fold?
    # Better to aggregate probs and compute CE once? 
    # Or mean CE across folds. Mean CE is standard.
    
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        
        # log_loss returns Nats by default (natural log) or bits? 
        # SKLearn log_loss uses ln.
        # We want bits.
        ce_nats = log_loss(y_test, y_prob, labels=list(range(K)))
        ce_bits = ce_nats / np.log(2)
        
        acc_scores.append(acc)
        ce_scores.append(ce_bits)
        
    mean_acc = np.mean(acc_scores)
    mean_ce = np.mean(ce_scores)
    
    # I_lb
    # H(S) = log2(K)
    h_s = np.log2(K)
    i_lb = h_s - mean_ce
    
    # Clip and Warn
    if i_lb < 0:
        print(f"Warning: Negative I_lb ({i_lb:.4f}) clipped to 0.")
        i_lb = 0.0
        
    return mean_acc, mean_ce, i_lb

def process_activity(activity_path, labels_path):
    print(f"Loading {activity_path}...")
    act = np.load(activity_path)
    labels = np.load(labels_path)
    
    # Reshape: (TotalTime, N) -> (Trials, TimeSteps, N)
    n_trials = len(labels)
    time_steps = act.shape[0] // n_trials
    
    # Ensure divisible
    assert act.shape[0] % n_trials == 0, "Activity time dim not divisible by trials"
    
    act_reshaped = act.reshape(n_trials, time_steps, -1)
    
    # Feature Extraction: Mean Rate over trial
    X = np.mean(act_reshaped, axis=1) # (Trials, N)
    
    print(f"Feature Matrix: {X.shape}")
    return X, labels

def main():
    DATA_DIR = "outputs/partC_canonical"
    
    results = []
    
    # 1. Real
    print("--- REAL ---")
    X_real, y_real = process_activity(
        os.path.join(DATA_DIR, "activity_real.npy"),
        os.path.join(DATA_DIR, "labels.npy")
    )
    acc, ce, ilb = compute_metrics(X_real, y_real)
    results.append({
        "condition": "Real",
        "acc": acc,
        "ce_bits": ce,
        "I_lb": ilb
    })
    print(f"Real: Acc={acc:.2f}, I_lb={ilb:.2f} bits")
    
    # 2. Label Shuffle
    print("--- LABEL SHUFFLE ---")
    y_shuff = y_real.copy()
    np.random.shuffle(y_shuff)
    acc_s, ce_s, ilb_s = compute_metrics(X_real, y_shuff)
    results.append({
        "condition": "LabelShuffle",
        "acc": acc_s,
        "ce_bits": ce_s,
        "I_lb": ilb_s
    })
    print(f"LabelShuffle: Acc={acc_s:.2f}, I_lb={ilb_s:.2f} bits")
    
    # 3. Conn Shuffle
    print("--- CONN SHUFFLE ---")
    try:
        X_conn, _ = process_activity(
            os.path.join(DATA_DIR, "activity_conn_shuffle.npy"),
            os.path.join(DATA_DIR, "labels.npy")
        )
        # Use real labels for ConnShuffle (labels correspond to stimulus direction, which is same order)
        acc_c, ce_c, ilb_c = compute_metrics(X_conn, y_real)
        results.append({
            "condition": "ConnShuffle",
            "acc": acc_c,
            "ce_bits": ce_c,
            "I_lb": ilb_c
        })
        print(f"ConnShuffle: Acc={acc_c:.2f}, I_lb={ilb_c:.2f} bits")
        
        # Check ConnShuffle > Real Logic
        if ilb_c > ilb:
            print("Notice: ConnShuffle > Real. Topology might hurt simple decoding or random is good enough reservoir.")
            
    except FileNotFoundError:
        print("ConnShuffle activity not found, skipping.")

    # Save
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(DATA_DIR, "partC_metrics.csv"), index=False)
    print("Saved metrics.")
    
    # Validation
    assert all(df["I_lb"] >= 0), "Found negative I_lb in output!"

if __name__ == "__main__":
    main()

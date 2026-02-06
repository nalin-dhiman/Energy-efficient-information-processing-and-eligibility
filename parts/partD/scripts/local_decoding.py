import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

class Decoder:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000, multi_class='multinomial')
        
    def fit(self, X, y):
        self.model.fit(X, y)
        
    def predict_proba(self, X):
        return self.model.predict_proba(X)
        
    def score(self, X, y):
        return self.model.score(X, y)

def compute_mi_lower_bound(X, y, n_splits=5):
    
    probs = np.bincount(y) / len(y)
    probs = probs[probs > 0]
    H_S = -np.sum(probs * np.log2(probs))
    

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    ce_scores = []
    
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        clf = LogisticRegression(max_iter=1000, multi_class='multinomial')
        try:
            clf.fit(X_train, y_train)
            y_pred = clf.predict_proba(X_test)
            
            ce = log_loss(y_test, y_pred) / np.log(2)
            ce_scores.append(ce)
        except:
            ce_scores.append(H_S) 
            
    mean_ce = np.mean(ce_scores)
    I_lb = H_S - mean_ce
    
    return max(0, I_lb)

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
    """
    Compute MI lower bound I_lb(S; R) using cross-validation Cross Entropy.
    I(S; R) = H(S) - H(S|R)
    H(S|R) <= CE(y, y_pred)
    I_lb = H(S) - CE
    
    X: (Samples, Features) - population rates
    y: (Samples,) - stimulus labels
    """
    # Entropy of labels H(S)
    # Assuming balanced classes for simplicity or compute empirical
    probs = np.bincount(y) / len(y)
    probs = probs[probs > 0]
    H_S = -np.sum(probs * np.log2(probs))
    
    # Cross Entropy estimation via CV
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    ce_scores = []
    
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        clf = LogisticRegression(max_iter=1000, multi_class='multinomial')
        try:
            clf.fit(X_train, y_train)
            y_pred = clf.predict_proba(X_test)
            # log_loss returns nat based mean CE usually, convert to bits
            # sklearn log_loss is -mean(log(p)). 
            # log base e. Convert to base 2: / ln(2)
            ce = log_loss(y_test, y_pred) / np.log(2)
            ce_scores.append(ce)
        except:
            ce_scores.append(H_S) # Fail safe
            
    mean_ce = np.mean(ce_scores)
    I_lb = H_S - mean_ce
    
    return max(0, I_lb)

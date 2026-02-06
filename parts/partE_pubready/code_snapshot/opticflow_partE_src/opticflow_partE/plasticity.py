import numpy as np

class PlasticityRule:
    def __init__(self, learning_rate=1e-4):
        self.lr = learning_rate
        
    def update(self, W, rates_hist, reward=None):
        
        raise NotImplementedError

class OjaRule(PlasticityRule):
    
    def __init__(self, learning_rate=1e-4, alpha=1.0):
        super().__init__(learning_rate)
        self.alpha = alpha
        
    def update(self, W, rates_hist, reward=None):
        
        
        T = rates_hist.shape[0]
        r = rates_hist
        

        hebbian = (r.T @ r) / T
        
        
        r_sq = np.mean(r**2, axis=0)
        

        decay = self.alpha * r_sq[:, None] * W
        
        dW = self.lr * (hebbian - decay)
        return dW

class R_Modulated_Hebb(PlasticityRule):
    
    def __init__(self, learning_rate=1e-4, R_baseline=0.0):
        super().__init__(learning_rate)
        self.R_base = R_baseline
        
    def update(self, W, rates_hist, reward=None):
        if reward is None:
            return np.zeros_like(W)
            
        
        
        T = rates_hist.shape[0]
        r = rates_hist
        

        correlation = (r.T @ r) / T
        

        dW = self.lr * (reward - self.R_base) * correlation
        
        return dW

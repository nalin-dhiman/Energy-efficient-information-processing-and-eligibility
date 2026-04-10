import numpy as np

class PlasticityRule:
    def __init__(self, learning_rate=1e-4):
        self.lr = learning_rate
        
    def update(self, W, rates_hist, reward=None):
        """
        Compute delta_W.
        W: (N, N)
        rates_hist: (Time, N)
        reward: scalar or (Time,)
        """
        raise NotImplementedError

class OjaRule(PlasticityRule):
    """
    Delta W_ij = eta * (r_i * r_j - alpha * r_i^2 * W_ij)
    Standard Oja: y(x - y*w). Pre=x, Post=y.
    Here W_ij is Input(j)->Output(i).
    Delta W_ij = eta * (r_i * r_j - r_i^2 * W_ij)
    """
    def __init__(self, learning_rate=1e-4, alpha=1.0):
        super().__init__(learning_rate)
        self.alpha = alpha
        
    def update(self, W, rates_hist, reward=None):
        # Batch update: sum over time
        # r_post: (T, N) -> r_i
        # r_pre: (T, N) -> r_j
        
        # dW_unconstrained = r_post.T @ r_pre  (N, N)
        # But Oja has a decay term: - alpha * r_post^2 * W
        
        T = rates_hist.shape[0]
        r = rates_hist
        
        # Hebbian term: Mean over time
        hebbian = (r.T @ r) / T
        
        # Decay term
        # r_post_sq = mean(r^2, axis=0) -> (N,)
        r_sq = np.mean(r**2, axis=0)
        
        # Oja: dW_ij = <r_i r_j> - alpha * <r_i^2> W_ij
        decay = self.alpha * r_sq[:, None] * W
        
        dW = self.lr * (hebbian - decay)
        return dW

class R_Modulated_Hebb(PlasticityRule):
    """
    Three-factor rule:
    dW_ij = eta * (R - baseline) * eligibility_ij
    eligibility_ij = r_i * r_j (filtered or instantaneous)
    """
    def __init__(self, learning_rate=1e-4, R_baseline=0.0):
        super().__init__(learning_rate)
        self.R_base = R_baseline
        
    def update(self, W, rates_hist, reward=None):
        if reward is None:
            return np.zeros_like(W)
            
        # Reward is scalar per trial (or time series?)
        # Implementation Plan says "R is a scalar reward per trial derived from MI proxy"
        # So reward is a scalar R.
        # r = rates_hist (Time, N)
        
        T = rates_hist.shape[0]
        r = rates_hist
        
        # Average correlation
        correlation = (r.T @ r) / T
        
        # dW = lr * (R - b) * correl
        dW = self.lr * (reward - self.R_base) * correlation
        
        return dW

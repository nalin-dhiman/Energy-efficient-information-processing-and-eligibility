import numpy as np

class RateRNN:
    """
    Rate-based RNN: tau * dx/dt = -x + W * ReLU(x) + I
    """
    def __init__(self, W_initial, tau=0.02, dt=0.001):
        """
        W_initial: Adjacency matrix (N, N)
        tau: Time constant (s)
        dt: Simulation step (s)
        """
        self.W = W_initial
        self.tau = tau
        self.dt = dt
        self.n_neurons = W_initial.shape[0]
        
    def step(self, x, I_ext):
        """
        Euler integration step.
        x: (Batch, N) or (N,)
        I_ext: (Batch, N) or (N,)
        """
        x = np.atleast_1d(x)
        I_ext = np.atleast_1d(I_ext)
        
        # Activation f(x) = ReLU(x)
        r = np.maximum(x, 0)
        
        # dxdt
        # W dot r -> (N, N) dot (N,) -> (N,)
        # For batch: (Batch, N) dot (N, N)^T -> (Batch, N)
        recurrent_input = r @ self.W.T
        
        dxdt = (-x + recurrent_input + I_ext) / self.tau
        
        x_new = x + dxdt * self.dt
        return x_new
        
    def run(self, I_stim, x0=None):
        """
        Run simulation for entire stimulus trace.
        I_stim: (Time, N)
        """
        T_steps = I_stim.shape[0]
        if x0 is None:
            x0 = np.zeros(self.n_neurons)
            
        x_hist = np.zeros((T_steps, self.n_neurons))
        r_hist = np.zeros((T_steps, self.n_neurons))
        
        x = x0
        
        for t in range(T_steps):
            x = self.step(x, I_stim[t])
            x_hist[t] = x
            r_hist[t] = np.maximum(x, 0)
            
        return x_hist, r_hist

import numpy as np

class DriftingGrating:
    """
    Generates time-dependent input currents for T4/T5 neurons
    simulating drifting gratings in 4 cardinal directions.
    """
    def __init__(self, directions=[0, 90, 180, 270], trials=10, dt=1e-3, T=1.0):
        self.directions = directions
        self.trials = trials
        self.dt = dt
        self.T = T
        self.time_steps = int(T / dt)
        
    def generate_input(self, n_neurons, retinotopy_df):
        """
        Generate input current matrix I_stim (Time x Neurons).
        
        Args:
           n_neurons: Total neurons
           retinotopy_df: DataFrame with ['neuron_id', 'u', 'v', 'mean_u', 'mean_v']
        
        Returns:
           I_stim: (Directions * Trials * Time, N_Neurons)
           labels: (Directions * Trials) - direction index per trial
        """
        # For simplicity in this reconstruction, we'll assume a simplified 
        # spatial receptive field where neurons prefer specific directions 
        # based on their type (T4a/b/c/d, T5a/b/c/d) or just random if unknown.
        # But crucially, they must respond to spatial frequency.
        
        # However, without detailed RFs, we can simulate "tuned" inputs
        # by assigning a preferred direction to each neuron (e.g. based on Type).
        
        # Types:
        # a: front-to-back (approx 0 deg)
        # b: back-to-front (approx 180 deg)
        # c: up (90)
        # d: down (270)
        
        # We need neuron types to assign tuning. 
        # If retinotopy_df has types, great. If not, we rely on broad tuning.
        
        # Let's assume we pass in a mapping or just generate generic "tuned" noise 
        # plus a spatial wave component if (u,v) are present.
        
        # 1. Assign preferred direction based on type if available, else random.
        # We'll just generate inputs that have spatial structure.
        
        # Spatial wave parameters
        k = 2.0 * np.pi / 20.0 # spatial freq (20 units wavelength)
        omega = 2.0 * np.pi * 2.0 # temporal freq (2 Hz)
        
        # Coordinates
        if 'mean_u' in retinotopy_df.columns and 'mean_v' in retinotopy_df.columns:
            u = retinotopy_df['mean_u'].fillna(0).values
            v = retinotopy_df['mean_v'].fillna(0).values
        else:
            u = np.zeros(n_neurons)
            v = np.zeros(n_neurons)
            
        inputs = []
        labels = []
        
        for i, direction in enumerate(self.directions):
            theta = np.radians(direction)
            kx = k * np.cos(theta)
            ky = k * np.sin(theta)
            
            for tr in range(self.trials):
                t = np.arange(self.time_steps) * self.dt
                # Wave: cos(kx*u + ky*v - omega*t)
                # Shape: (Time, Neurons)
                # Outer sum: (Time, 1) + (1, Neurons) -> broadcasting
                phase = kx * u + ky * v
                wave = np.cos(phase[None, :] - omega * t[:, None])
                
                # Rectify (firing rate input > 0)
                wave = np.maximum(wave, 0)
                
                # Add some noise
                noise = np.random.normal(0, 0.1, wave.shape)
                
                I_tr = wave + noise
                inputs.append(I_tr)
                labels.append(i)
                
        return np.concatenate(inputs, axis=0), np.array(labels)

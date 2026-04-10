import torch
import torch.nn as nn
import numpy as np

class RateRNNTorch(nn.Module):
    def __init__(self, n_neurons, mask, tau=0.02, dt=0.001):
        super().__init__()
        self.n = n_neurons
        self.tau = tau
        self.dt = dt
        
        # W -> Parameter (masked)
        # We store full W but multiply by mask in forward
        self.W = nn.Parameter(torch.randn(n_neurons, n_neurons) * 0.01)
        self.mask = torch.tensor(mask, dtype=torch.float32)
        self.elu = nn.ReLU() # Rate model usually ReLU
        
    def forward(self, inputs, x0=None):
        # inputs: (Time, N)
        T = inputs.shape[0]
        if x0 is None:
            x = torch.zeros(self.n, device=inputs.device)
        else:
            x = x0
            
        rate_hist = []
        
        # Apply mask
        W_eff = self.W * self.mask
        
        for t in range(T):
            I_t = inputs[t]
            # dx/dt = (-x + W r + I) / tau
            r = self.elu(x)
            
            # W r: (N, N) @ (N,) -> (N,)
            rec = W_eff @ r
            
            dxdt = (-x + rec + I_t) / self.tau
            x = x + dxdt * self.dt
            rate_hist.append(self.elu(x))
            
        return torch.stack(rate_hist)

def train_oracle(W_init_np, inputs_np, labels_np, wiring_coords_np=None, 
                 steps=100, lr=0.01, lambda_energy=0.001, lambda_mi=1.0):
    """
    Run gradient descent to maximize Objective J = MI - Energy.
    Because MI is hard to differentiate directly (decoder based), 
    we use a proxy Loss: CrossEntropy of a linear readout trained *alongside* or 
    assume simple readout.
    
    Actually, to differentiate through MI, we usually need the system to optimize "Discriminability".
    Standard approach: Train RNN to minimize CrossEntropy directly on the labels.
    J = -CE - lambda * Energy.
    
    W_init_np: (N, N)
    inputs_np: (Time, N)
    labels_np: (Time,) or per trial? 
               Assuming inputs_np is concatenated trials. labels aligned.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    N = W_init_np.shape[0]
    mask = (W_init_np != 0).astype(np.float32)
    
    model = RateRNNTorch(N, mask).to(device)
    model.W.data = torch.tensor(W_init_np, dtype=torch.float32).to(device)
    
    # Readout head (linear)
    # Mapping rates -> logits for Directions
    n_classes = len(np.unique(labels_np))
    readout = nn.Linear(N, n_classes).to(device)
    
    optimizer = torch.optim.Adam(list(model.parameters()) + list(readout.parameters()), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    
    inputs_t = torch.tensor(inputs_np, dtype=torch.float32).to(device)
    labels_t = torch.tensor(labels_np, dtype=torch.long).to(device)
    
    coords = None
    if wiring_coords_np is not None:
        coords = torch.tensor(wiring_coords_np, dtype=torch.float32).to(device)
    
    history = []
    
    # For efficiency we might need batching if Time is huge
    # Here assuming simplified "Short" traces
    
    for i in range(steps):
        optimizer.zero_grad()
        
        rates = model(inputs_t)
        
        # Readout on last step of each trial? 
        # Or average rate over trial? 
        # Stimuli inputs_np is continuous.
        # Let's assume we predict label at every step or mean pool?
        # Standard: Predict label from mean rate of trial.
        
        # But here inputs is (TotalTime, N). And labels (TotalTime) (broadcasted)
        # Let's do frame-by-frame decoding for dense gradient
        logits = readout(rates)
        
        ce_loss = loss_fn(logits, labels_t)
        
        # Energy
        # 1. Metabolic: sum(rates)
        e_met = torch.sum(rates)
        # 2. Wire: sum(W * dist * pre_rate)
        e_wire = torch.tensor(0.0).to(device)
        
        # Calc e_wire differentiable
        if coords is not None:
            # pairwise diff
            # This is heavy for N>2000 in loop.
            # dists pre-calc?
            pass # Skip complex wiring grad for Oracle baseline prompt simplicity if N large
            
        loss = ce_loss + lambda_energy * e_met 
        # (Wiring term omitted in Oracle grad for speed unless critical)
        
        loss.backward()
        optimizer.step()
        
        # Clamp W to be positive? (If excitatory only)
        # model.W.data.clamp_(min=0)
        
        history.append(loss.item())
        
        if i % 10 == 0:
            pass # print(f"Oracle Step {i}: Loss {loss.item()}")
            
    return (model.W.data * model.mask.to(device)).cpu().numpy(), history

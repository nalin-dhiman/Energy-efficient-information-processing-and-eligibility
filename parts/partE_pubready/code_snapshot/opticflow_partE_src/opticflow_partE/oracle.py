import torch
import torch.nn as nn
import numpy as np

class RateRNNTorch(nn.Module):
    def __init__(self, n_neurons, mask, tau=0.02, dt=0.001):
        super().__init__()
        self.n = n_neurons
        self.tau = tau
        self.dt = dt
        
       
        self.W = nn.Parameter(torch.randn(n_neurons, n_neurons) * 0.01)
        self.mask = torch.tensor(mask, dtype=torch.float32)
        self.elu = nn.ReLU() 
    def forward(self, inputs, x0=None):

        T = inputs.shape[0]
        if x0 is None:
            x = torch.zeros(self.n, device=inputs.device)
        else:
            x = x0
            
        rate_hist = []
        

        W_eff = self.W * self.mask
        
        for t in range(T):
            I_t = inputs[t]

            r = self.elu(x)
            

            rec = W_eff @ r
            
            dxdt = (-x + rec + I_t) / self.tau
            x = x + dxdt * self.dt
            rate_hist.append(self.elu(x))
            
        return torch.stack(rate_hist)

def train_oracle(W_init_np, inputs_np, labels_np, wiring_coords_np=None, 
                 steps=100, lr=0.01, lambda_energy=0.001, lambda_mi=1.0):
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    N = W_init_np.shape[0]
    mask = (W_init_np != 0).astype(np.float32)
    
    model = RateRNNTorch(N, mask).to(device)
    model.W.data = torch.tensor(W_init_np, dtype=torch.float32).to(device)
    
   
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
    
    
    
    for i in range(steps):
        optimizer.zero_grad()
        
        rates = model(inputs_t)
        
        
        logits = readout(rates)
        
        ce_loss = loss_fn(logits, labels_t)
        
        
        e_met = torch.sum(rates)

        e_wire = torch.tensor(0.0).to(device)
        

        if coords is not None:
            
            pass 
            
        loss = ce_loss + lambda_energy * e_met 

        
        loss.backward()
        optimizer.step()
        
        
        
        history.append(loss.item())
        
        if i % 10 == 0:
            pass 
    return (model.W.data * model.mask.to(device)).cpu().numpy(), history

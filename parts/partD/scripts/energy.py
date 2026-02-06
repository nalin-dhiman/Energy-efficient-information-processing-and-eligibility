import numpy as np
import scipy.sparse as sp

def compute_wiring_distance_matrix(retinotopy_df, method="euclidean"):
   
    coords = retinotopy_df[['mean_u', 'mean_v']].fillna(0).values
   
    return coords

def compute_energy_components(rates, W, dist_coords=None, wiring_penalty_alpha=0.01):
    
    E_met = np.sum(rates)
    
    
    
    out_degrees = np.sum(np.abs(W), axis=0) # Sum over i (post) for each j (pre)
    E_syn = np.sum(rates @ out_degrees) 
    
    
    
    E_wire = 0.0
    if dist_coords is not None:
        
        rows, cols = W.nonzero()
        weights = W[rows, cols]
        

        dists = np.linalg.norm(dist_coords[rows] - dist_coords[cols], axis=1)
        
        
        total_activity = np.sum(rates, axis=0)
        
        E_wire = np.sum(weights * dists * total_activity[cols]) * wiring_penalty_alpha

    return {
        "E_met": E_met,
        "E_syn": E_syn,
        "E_wire": E_wire,
        "E_total": E_met + E_syn + E_wire
    }

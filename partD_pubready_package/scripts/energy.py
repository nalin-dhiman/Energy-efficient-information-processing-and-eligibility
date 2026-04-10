import numpy as np
import scipy.sparse as sp

def compute_wiring_distance_matrix(retinotopy_df, method="euclidean"):
    """
    Compute distance matrix between all pairs of neurons based on retinotopy.
    retinotopy_df: ['neuron_id', 'mean_u', 'mean_v']
    """
    coords = retinotopy_df[['mean_u', 'mean_v']].fillna(0).values
    # N x N distance
    # For large N, this is huge. Prefer working with sparse adjacency usually.
    # Here, we assume W is sparse, so we only need dists for edges in W.
    return coords

def compute_energy_components(rates, W, dist_coords=None, wiring_penalty_alpha=0.01):
    """
    Compute energy components.
    rates: (Time, N)
    W: (N, N) adjacency (synaptic weights)
    dist_coords: (N, 2) array of u,v coordinates for wiring cost
    
    Returns dict of energy terms (summed over time and population).
    """
    # 1. Metabolic / Baseline (sum of rates)
    # E_met = sum(r)
    E_met = np.sum(rates)
    
    # 2. Synaptic (sum of weights * pre_rate)
    # E_syn = sum_j (sum_i W_ij * r_i) = sum_i r_i * (sum_j W_ij)
    # W is (Post, Pre) in math usually, but here stored as (Pre, Post) often?
    # In dynamics.py: r @ W.T => W is (Post, Pre) ? 
    # Let's assume W is (N, N) where W[i,j] is weight from j to i (standard RNN)
    # Or W[i,j] from i to j (adjacency)?
    # dynamics.py says: recurrent_input = r @ W.T 
    # If r is (1, N), output is (1, N). 
    # r @ W.T = [r1...rn] @ [col1...coln]
    # This implies W[i,j] is weight TO i FROM j.
    
    out_degrees = np.sum(np.abs(W), axis=0) # Sum over i (post) for each j (pre)
    E_syn = np.sum(rates @ out_degrees) 
    
    # 3. Wiring Cost (D-FIX1)
    # E_wire = sum_i sum_j (W_ij * dist(i,j) * r_j)
    # This is "active wiring cost" - cost active synapses
    # Or static wiring cost? "J = ... - lambda * E" usually implies active dynamic cost?
    # Or often wiring cost is structural: sum |W_ij| * d_ij.
    # The prompt says: E_wire = gamma * Sum(A_ij * dist * r_j) integrated over time.
    # So it IS activity dependent.
    
    E_wire = 0.0
    if dist_coords is not None:
        # We need distances for non-zero weights
        # Iterate sparse W or approximate
        rows, cols = W.nonzero()
        weights = W[rows, cols]
        
        # dist(i,j)
        dists = np.linalg.norm(dist_coords[rows] - dist_coords[cols], axis=1)
        
        # active wiring cost: weight * dist * r_pre (cols)
        # Sum over time: total activity of pre_neuron
        total_activity = np.sum(rates, axis=0) # (N,)
        
        E_wire = np.sum(weights * dists * total_activity[cols]) * wiring_penalty_alpha

    return {
        "E_met": E_met,
        "E_syn": E_syn,
        "E_wire": E_wire,
        "E_total": E_met + E_syn + E_wire
    }

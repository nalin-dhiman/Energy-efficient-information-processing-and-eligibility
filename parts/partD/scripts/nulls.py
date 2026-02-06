import numpy as np

def null_shuffle_weights(W):
    
    W_null = W.copy()
    rows, cols = W_null.nonzero()
    weights = W_null[rows, cols]
    np.random.shuffle(weights)
    W_null[rows, cols] = weights
    return W_null

def null_strength_preserving(W):
    
    W_null = W.copy()
   
    
    rows, cols = W.nonzero()
    vals = W[rows, cols]
    
    new_cols = cols.copy()
    
    n_rows = W.shape[0]
    
    
    if isinstance(W, np.ndarray):
        W_null = np.zeros_like(W)
        for i in range(n_rows):
            row_vals = W[i, W[i] != 0]
            if len(row_vals) > 0:

                rand_cols = np.random.choice(n_rows, size=len(row_vals), replace=False)
                W_null[i, rand_cols] = row_vals
        return W_null
    else:

        return null_shuffle_weights(W) 

def null_spatial_preserving_type_shuffle(W, nodes_df):
   
    
    n_nodes = W.shape[0]
    perm = np.arange(n_nodes)
    
   
    pass 
    

    return W.copy()

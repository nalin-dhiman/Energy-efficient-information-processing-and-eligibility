import pandas as pd
import os


NEURON_FILE_FEATHER = "Neuprint_Neurons.feather"
NEURON_CONN_FILE_FEATHER = "Neuprint_Neuron_Connections.feather"
NEURON_CONN_FILE_CSV = "Neuprint_Neuron_Connections.csv"

NEURON_COL_MAP = {
    "bodyId:long": "body_id",
    "bodyId": "body_id",
    ":ID(Body-ID)": "body_id",
    
    "type:string": "type",
    "type": "type",
    
    "instance:string": "instance",
    "instance": "instance",
    
    "size:long": "size",
    "size": "size",
    
    "pre:int": "pre_count",
    "pre": "pre_count",
    
    "post:int": "post_count",
    "post": "post_count",
    
    "status:string": "status",
    "status": "status",
    
    "roiInfo:string": "roi_info",
    "roiInfo": "roi_info",
    
    "statusLabel:string": "status_label",
    "statusLabel": "status_label"
}

CONN_COL_MAP = {
    ":START_ID(Body-ID)": "pre_id",
    ":END_ID(Body-ID)": "post_id",
    "weight:int": "weight",
    "weight": "weight",
    "roi:string": "roi",
    "roiInfo:string": "roi_info",
    "roi": "roi"
}

def load_neurons_raw(data_dir):
    """
    Load raw neurons table and rename columns to canonical snake_case.
    """
    path = os.path.join(data_dir, NEURON_FILE_FEATHER)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Neurons file not found: {path}")
        
    df = pd.read_feather(path)
    
    
    canonical_targets = {
        "body_id": ["bodyId:long", "bodyId", ":ID(Body-ID)"],
        "type": ["type:string", "type"],
        "instance": ["instance:string", "instance"],
        "size": ["size:long", "size"],
        "pre_count": ["pre:int", "pre"],
        "post_count": ["post:int", "post"],
        "status": ["status:string", "status"],
        "roi_info": ["roiInfo:string", "roiInfo"],
        "status_label": ["statusLabel:string", "statusLabel"]
    }
    
    final_cols = {}
    
    for target, sources in canonical_targets.items():
        found = False
        for src in sources:
            if src in df.columns:
                final_cols[src] = target
                found = True
                break 
    df = df[list(final_cols.keys())].rename(columns=final_cols)
    

    if "body_id" in df.columns:
        df["body_id"] = df["body_id"].astype("int64")
        
    return df

def load_connections_raw(data_dir):
   
    fpath = os.path.join(data_dir, NEURON_CONN_FILE_FEATHER)
    if os.path.exists(fpath):
        df = pd.read_feather(fpath)
    else:
        cpath = os.path.join(data_dir, NEURON_CONN_FILE_CSV)
        if not os.path.exists(cpath):
            raise FileNotFoundError("no file")
        df = pd.read_csv(cpath)
        
    actual_map = {col: CONN_COL_MAP[col] for col in df.columns if col in CONN_COL_MAP}
    df = df.rename(columns=actual_map)
    
    return df

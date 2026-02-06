import pandas as pd
import os
import json

def audit_dataset_semantics(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    report = []
    report.append("# Data Contract & Semantics Audit")
    
    csv_stats = []
    

    print("Auditing Neurons Table...")

    n_path = os.path.join(data_dir, "Neuprint_Neurons.feather")
    if os.path.exists(n_path):
        neurons = pd.read_feather(n_path)
        
        if "bodyId" in neurons.columns:
            pk = "bodyId"
        elif ":ID(Body-ID)" in neurons.columns:
            pk = ":ID(Body-ID)"
        else:
            pk = neurons.columns[0]
            
        n_total = len(neurons)
        n_unique = neurons[pk].nunique()
        
        report.append("## Neurons Table")
        report.append(f"- **Total Rows**: {n_total}")
        report.append(f"- **Unique {pk}**: {n_unique}")
        if n_total == n_unique:
            report.append("- **Condition**: Primary Key Valid (Unique)")
        else:
            report.append(f"- **Condition**: DUPLICATES FOUND ({n_total - n_unique})")
            
        
        type_col = "type" if "type" in neurons.columns else "type:string"
        if type_col in neurons.columns:
            n_typed = neurons[type_col].notna().sum()
            n_unknown = n_total - n_typed
            report.append(f"- **Typed Neurons**: {n_typed}")
            report.append(f"- **Untyped/Fragments**: {n_unknown}")
            

            typed_count = n_typed
        else:
            report.append("- **Type Column**: MISSING")
            typed_count = 0
            

    print("Auditing Elements (Pins)...")

    pin_path = os.path.join(data_dir, "Neuprint_Elements/medulla-pins/000000.csv")
    if os.path.exists(pin_path):
        pins = pd.read_csv(pin_path)
        
        pk_el = ":ID(Element-ID)"
        if pk_el in pins.columns:
            
            report.append("## Elements (Pins)")
            report.append(f"- **Primary Key Candidate**: {pk_el}")
            report.append("- **Semantics**: Represents a synaptic/columnar location.")
            pass
            

    print("Auditing Mappings...")

    map_n_es = os.path.join(data_dir, "Neuprint_Neuron_to_ElementSet_ColumnPin.csv")
    if os.path.exists(map_n_es):
        df = pd.read_csv(map_n_es)

        n_unique_bodies = df[":START_ID(Body-ID)"].nunique()
        report.append("## Mappings")
        report.append("### Neuron -> PinSet")
        report.append(f"- **Mapped Bodies**: {n_unique_bodies}")
        

    report.append("## Reconciliation")
    report.append(f"The '10.3M nodes' figure refers to Total Rows in Neurons table.")
    report.append(f"However, only ~{typed_count} have assigned Types.")
    report.append("Most nodes are likely untraced fragments (Status != Traced).")
    

    report.append("## Data Contract Definition")
    report.append("1. **Neuron**: Enitity in `Neuprint_Neurons` with `status='Traced'` (approx). Ideally has `type`.")
    report.append("2. **Fragment**: Entity in `Neuprint_Neurons` with `status!='Traced'` or missing type.")
    report.append("3. **Pin/Element**: Spatial marker in `Neuprint_Elements`.")
    report.append("4. **Coverage Metric**: We must use `Typed Neurons` as the denominator for scientific claims.")
    
    with open(os.path.join(output_dir, "data_contract.md"), "w") as f:
        f.write("\n".join(report))
        
    print("Audit Complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    
    audit_dataset_semantics(args.data, args.out)

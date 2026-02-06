import pandas as pd
import json
import os
import glob
import re

def analyze_missingness(df, columns):

    report = {}
    for col in columns:
        if col in df.columns:
            n_missing = int(df[col].isna().sum())
            n_total = len(df)
            report[col] = {
                'n_missing': n_missing,
                'pct_missing': round((n_missing / n_total) * 100, 2) if n_total > 0 else 0
            }
        else:
            report[col] = 'COLUMN_MISSING'
    return report

def check_consistency(provenance_path, data_report):

    warnings = []
    
    if not os.path.exists(provenance_path):
        warnings.append("Provenance file missing.")
        return warnings

    with open(provenance_path, 'r') as f:
        prov = json.load(f)
    
   
    
    return warnings

def generate_schema_report(data_dir, out_dir):

    
    report = {
        'files': {},
        'warnings': [],
        'summary': {}
    }
    
    parquet_files = glob.glob(os.path.join(data_dir, "*.parquet"))
    
    for p_file in parquet_files:
        basename = os.path.basename(p_file)
        try:
            df = pd.read_parquet(p_file)
            file_info = {
                'columns': list(df.columns),
                'n_rows': len(df),
                'missingness': {}
            }
            

            if 'cells' in basename:
                file_info['missingness'] = analyze_missingness(df, ['type', 'instance'])
            
            if 'retinotopy' in basename:

                cols = df.columns
                coord_cols = [c for c in cols if 'mean' in c or 'u' in c or 'v' in c]
                file_info['missingness'] = analyze_missingness(df, coord_cols)
            
            report['files'][basename] = file_info
            
        except Exception as e:
            report['warnings'].append(f"Failed to read {basename}: {str(e)}")
            

    prov_path = os.path.join(data_dir, "provenance.json")
    consistency_warnings = check_consistency(prov_path, report)
    report['warnings'].extend(consistency_warnings)
    

    os.makedirs(out_dir, exist_ok=True)
    
    out_path = os.path.join(out_dir, "schema_summary.json")
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)
        
    print(f"Schema report written to {out_path}")
    return report

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    
    generate_schema_report(args.data_dir, args.out_dir)

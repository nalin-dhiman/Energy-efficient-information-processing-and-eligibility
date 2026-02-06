import argparse
from . import mdl

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    
    mdl.run_mdl_sweep(args.data, args.out)

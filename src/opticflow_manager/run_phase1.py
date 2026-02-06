import argparse
from . import builder

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw neuprint tables")
    parser.add_argument("--out", required=True, help="Output directory for full dataset")
    args = parser.parse_args()
    
    builder.build_dataset(args.input, args.out)

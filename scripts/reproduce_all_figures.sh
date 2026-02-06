#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT_DIR="$ROOT_DIR/reproduced_outputs"
mkdir -p "$OUT_DIR"

# Part A
python "$ROOT_DIR/parts/partA_figures/scripts/make_figures_partA.py" \
  --data_dir "$ROOT_DIR/parts/partA/data" \
  --out_dir "$OUT_DIR/partA" \
  --dpi 600

# Part B
python "$ROOT_DIR/parts/partB_pubready/scripts/plot_partB_pubready.py" \
  --data_dir "$ROOT_DIR/parts/partB/data" \
  --out_dir "$OUT_DIR/partB" \
  --dpi 600

# Part C
python "$ROOT_DIR/parts/partC_pubready/scripts/plot_partC_pubready.py" \
  --data_dir "$ROOT_DIR/parts/partC/data" \
  --out_dir "$OUT_DIR/partC"

python "$ROOT_DIR/parts/partC/scripts/make_partC_report.py" \
  --data_dir "$ROOT_DIR/parts/partC/data" \
  --out_md "$OUT_DIR/partC/partC_report.md"

# Part D
python "$ROOT_DIR/parts/partD/scripts/plot_partD_pubready.py" \
  --data_dir "$ROOT_DIR/parts/partD/data" \
  --out_dir "$OUT_DIR/partD" \
  --tables_dir "$OUT_DIR/partD_tables"

# Part E
python "$ROOT_DIR/parts/partE/scripts/make_partE_figures.py" \
  --data_dir "$ROOT_DIR/parts/partE/data" \
  --out_dir "$OUT_DIR/partE"

echo "Done. Outputs in: $OUT_DIR"

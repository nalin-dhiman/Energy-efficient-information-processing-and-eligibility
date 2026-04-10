# Part D — Energy & wiring efficiency (publication-ready package)

This folder contains:
- `data/` : canonical CSVs used to generate Part D figures and tables
- `scripts/plot_partD_pubready.py` : generates publication-quality figures (PDF + PNG) and summary tables
- `figures/` : rendered outputs (PDF vector + PNG high-DPI)
- `tables/` : CSV/TSV tables used in the manuscript
- `provenance/` : notes and upstream reports

## Quickstart

From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r scripts/requirements_partD.txt

python scripts/plot_partD_pubready.py \
  --data_dir data \
  --out_dir figures \
  --tables_dir tables
```

## Notes
- Figures are saved as both vector PDF and 600-dpi PNG.
- Legends are placed outside axes where possible to avoid covering data.

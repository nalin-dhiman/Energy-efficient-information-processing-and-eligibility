# Part A figure build (publication-ready)

This folder builds **Part A** figures (dataset QC + coverage + type graph) from the processed parquets.

## Figures produced
- `FigA1_PartA_QC` — basic QC panels:
  - T4/T5 subtype counts
  - type + retinotopy coverage
  - synaptic weight distribution (log-x)
- `FigA2_Retinotopy_Coverage` — 2D density of mapped retinotopy coordinates
- `FigA3_TypeGraph_Heatmap` — type→type aggregated synaptic weight matrix (log-scaled)

Outputs are written to:
- `.../pdf/*.pdf` (vector)
- `.../png/*.png` (raster, high DPI)

## Inputs
The script expects these files in `--data_dir`:
- `cells.parquet`
- `retinotopy.parquet`
- `cell_graph.parquet`
- `type_graph.parquet`

## Reproduction (recommended)
From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r REQUIREMENTS.txt

python scripts/make_figures_partA.py --data_dir data --out_dir figures_out --dpi 600
```

If your system uses `python3`:

```bash
python3 scripts/make_figures_partA.py --data_dir data --out_dir figures_out --dpi 600
```



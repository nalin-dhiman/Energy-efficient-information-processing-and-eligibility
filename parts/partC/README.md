# Part C (Functional information) — Canonical

This folder is a **clean, reproducible** plotting + reporting layer for Part C.
It intentionally avoids requiring large intermediate tensors (e.g. activity .npy files).

## What this package does

- Generates **publication-ready** figures from `partC_metrics.csv` and `partC_local_metrics.csv`.
- Generates an up-to-date `partC_report.md` directly from the CSVs (no hand-edited numbers).
- Adds a **retinotopy tile count map** (`FigC3_TileCounts`) to document coverage/valid tiles.

## What this package does *not* do

- It does **not** rerun the full dynamical simulation (that requires the connectome graph + large activity arrays).
- It does **not** reconstruct the per-tile information heatmap unless you provide `local_tile_ilb.csv` (optional input; see below).

## Inputs

Expected files in `data/`:

- `partC_metrics.csv` — per-seed global decoding results
- `partC_local_metrics.csv` — per-seed local decoding summary
- `retinotopy_subset.csv` — neuron retinotopy coordinates (used to compute tile coverage)

Optional:

- `local_tile_ilb.csv` — per-tile information values for Real vs LabelShuffle.


## How to run

From the package root:

```bash
python scripts/plot_partC_pubready.py --data_dir data --out_dir figures_pubready
python scripts/make_partC_report.py --data_dir data --out_md reports/partC_report.md
```

Outputs:

- `figures_pubready/FigC1_GlobalDecoding.{png,pdf}`
- `figures_pubready/FigC2_LocalDecoding.{png,pdf}`
- `figures_pubready/FigC3_TileCounts.{png,pdf}`
- `reports/partC_report.md`



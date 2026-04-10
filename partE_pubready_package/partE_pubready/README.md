# Part E — Publication-ready package (Nat Comms v8.6.1 locked)

This folder contains the **Part E** (plasticity) deliverables needed to regenerate the *publication-quality* figures and summary tables used in the manuscript.

## What’s inside

- `data/` — canonical CSVs used for plotting (patch generalization, compute-fairness, scale & failure-policy sensitivity, cost-model robustness).
- `scripts/make_partE_figures.py` — **single entry point** that generates all figures/tables.
- `scripts/make_partE_trajectory_supp.py` — builds the new longitudinal learning-trajectory supplement (`FigE7`) from archived `partE_v6_3` logs.
- `scripts/run_partE_motif_supp.py` — reruns a representative Part E patch, saves final learned weights, and builds the motif supplement (`FigE8`).
- `scripts/make_partE_patch_motif_generalization.py` — reruns multiple spatially distinct patches, saves final learned weights, and builds the multi-patch motif supplement (`FigE9`).
- `figures/` — output figures (PNG + PDF). *(Created by the script.)*
- `tables/` — derived summary tables (CSV). *(Created by the script.)*
- `reports/` — a short report describing each figure. *(Created by the script.)*
- `code_snapshot/` — a snapshot of the Part E code (`opticflow_partE.zip`) as provided.

## Environment

Python 3.10+ recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r scripts/requirements.txt
```

## Generate everything

Run from the **root of this folder**:

```bash
python scripts/make_partE_figures.py --data_dir data --out_dir .
```

Outputs:
- `figures/png/*.png` (600 dpi)
- `figures/pdf/*.pdf` (vector)
- `tables/*.csv`
- `reports/partE_pubready_report.md`

Additional supplementary analyses:

```bash
python scripts/make_partE_trajectory_supp.py \
  --log_dir ../../../../optic-lobe-v1.1-neuprint-tables/outputs/partE_v6_3/logs \
  --fig_dir figures --table_dir tables --report_dir reports

python scripts/run_partE_motif_supp.py \
  --raw_root ../../../../optic-lobe-v1.1-neuprint-tables \
  --fig_dir figures --table_dir tables --report_dir reports \
  --n_seeds 2 --n_nulls 10

python scripts/make_partE_patch_motif_generalization.py \
  --raw_root ../../../../optic-lobe-v1.1-neuprint-tables \
  --fig_dir figures --table_dir tables --report_dir reports \
  --n_patches 5 --n_nulls 10
```

These commands generate:
- `figures/pdf/FigE7_TrajectoryTradeoffs.pdf`
- `figures/pdf/FigE8_MotifAlignment.pdf`
- `figures/pdf/FigE9_MotifGeneralization.pdf`
- `tables/trajectory_rule_summary.csv`
- `tables/motif_rule_summary.csv`
- `tables/patch_motif_generalization_summary.csv`
- `reports/partE_trajectory_report.md`
- `reports/partE_motif_report.md`
- `reports/partE_patch_motif_generalization_report.md`

## Notes (critical)

- This package is **figure-centric**: it assumes the expensive network simulations have already been run and saved as CSV summaries.
- Comparisons are designed to be *reviewer-proof*:
  - Patch generalization across spatially diverse patches.
  - Compute fairness by **matched forward passes** (runtime is secondary).
  - Scale robustness across N ∈ {500, 1000, 2000}.
  - Failure-policy sensitivity (worst-case vs exclude vs impute).

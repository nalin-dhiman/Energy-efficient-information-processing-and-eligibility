# Part B (MDL / compression) — publication-ready figures

This folder contains a **self-contained** plotting pipeline for Part B.

## What was corrected (critical)
The earlier plots had issues that make reviewers cranky:
- Legends and annotations overlapped the data.
- The spatial null control looked blank because the null distribution is extremely tight compared to the real-vs-null gap.
- Mixed units (total bits vs bits/neuron) made comparisons hard to read.

This revision:
- Uses **bits per neuron** everywhere (clean scale, comparable across subsets).
- Uses `layout="constrained"` and puts legends **outside** axes.
- Replaces the null histogram with a **null vs real** categorical plot (box+points), which shows the effect clearly.

## Expected inputs
In `data/`:
- `partB_mdl_derived.csv` (preferred)
- `partB_mdl_canonical.csv` (fallback)
- `partB_null_spatial.csv` (optional but recommended)

## How to run

From this directory:

```bash
python scripts/plot_partB_pubready.py --data_dir data --out_dir figures_pub --dpi 600
```

Outputs are written to `figures_pub/` as both PNG (high-DPI) and PDF (vector).

### Running inside your repo
You said your data live here:

`/home/ub/Downloads/natcomm_fig/partB_canonical/scripts/data`

You can run using absolute paths:

```bash
python plot_partB_pubready.py \
  --data_dir /home/ub/Downloads/natcomm_fig/partB_canonical/scripts/data \
  --out_dir  /home/ub/Downloads/natcomm_fig/partB_canonical/figures_pub \
  --dpi 600
```

(or copy `plot_partB_pubready.py` into your `scripts/` folder and run it there.)

## Figures produced
- **FigB1_ModelSelection** — MDL decomposition (fit + BIC penalty) per neuron.
- **FigB2_ResidualGeometry_Waterfall** — waterfall explaining why M3 is penalized (fit gain vs complexity).
- **FigB3_BitsPerNeuron** — clean summary bars.
- **FigB4_NullControl** — spatial null vs real coordinates (box+points).


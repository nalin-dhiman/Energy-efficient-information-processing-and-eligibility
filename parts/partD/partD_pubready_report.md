# Part D — Energy & wiring efficiency (critical summary)

This package focuses on **energy–information efficiency** in a connectome-constrained dynamical model, using *decoder-based* information lower bounds (in bits) and explicit energy accounting.

## What the current results support (and what they don't)

### 1) Random rewiring is energetically expensive and less efficient
Compared to **Real**, the **Null_Conn** control shows substantially higher wiring energy:

- Real wire energy: 9.908e+06
- Null_Conn wire energy: 1.494e+07

Total energy per trial (base + spike + synapse + wire):

- Real total: 1.985e+07
- Null_Conn total: 2.488e+07

Even when raw information differs only modestly, **efficiency (bits / energy)** is consistently higher for Real across the entire tested energy-weight grid (α,β,γ):

- Global efficiency ratio Real/Null_Conn: median 1.144, minimum 1.094
- Local efficiency ratio Real/Null_Conn: median 1.109, minimum 1.060

**Interpretation**: the strongest Part D claim is about *energy efficiency*, not dramatic jumps in raw MI.

### 2) Information is spatially structured
Local decoding is reported per retinotopic tile (FigD3). This guards against a common failure mode: global readouts can dilute spatially localized signals.

## Reviewer-facing weaknesses / suggested upgrades

### A) The story must emphasize efficiency
Global MI lower bound (bits) currently:
- Real: 1.085 (acc 0.833)
- Null_Conn: 0.992 (acc 0.767)

These are not orders-of-magnitude different. The big, clean effect is wiring cost → efficiency.

### B) Null_Strength is not a strong falsification control
Null_Strength has similar wiring cost to Real (because geometry is largely preserved). Treat it as a “weight perturbation” control, not a primary baseline.

### C) Energy is in arbitrary units
That is fine, but you must clearly state that the conclusions are **comparative** (ratios/differences), not absolute Joule estimates.

### D) Add one more hard control (optional but strong)
If feasible, add a *spatial coordinate shuffle* (keeping adjacency but breaking retinotopy) and verify it degrades local information and/or increases wiring cost.

## Contents
- `scripts/plot_partD_pubready.py` — regenerates all pub-ready figures and tables from CSVs
- `figures/` — PDF (vector) + PNG (600 dpi)
- `tables/` — manuscript-ready CSV tables


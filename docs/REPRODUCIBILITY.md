# Reproducibility notes 

This repository supports two levels of reproducibility.

## Level 1 — Figure reproduction
**Goal:** regenerate all paper figures from the included canonical tables.

- Inputs: the CSV/parquet files under `parts/*/data/`
- Runtime: minutes
- Hardware: laptop OK

Commands:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r env/requirements.txt

bash scripts/reproduce_all_figures.sh
bash scripts/verify_checksums.sh
```

## Level 2 — End-to-end pipeline reproduction (research-grade, expensive)
**Goal:** rebuild the canonical tables from the raw connectome release.

- Requires downloading FlyEM optic-lobe connectome v1.1 from:
  - `gs://flyem-optic-lobe/v1.1/`
- Compute: can be substantial (RAM/CPU), depending on which upstream stages you rerun
- Not every upstream stage is packaged here; the journal-facing intent is figure reproducibility.



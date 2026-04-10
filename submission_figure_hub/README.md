# Submission Figure Hub

This directory centralizes the regeneration and collection of every figure that
the revised manuscript and supplement currently use.

## What this hub does

- Runs the source figure generators in `partA_figures_package`, `partB_pubready_package`,
  `partC_pubready_package`, `partD_pubready_package`, and `partE_pubready_package`.
- Collects the manuscript-facing PDF and PNG assets into one place.
- Records the mapping from manuscript figure names to source scripts in `manifest.csv`.

## Main entrypoint

```bash
python regenerate_submission_figures.py
```

Outputs are written to:

- `collected/pdf/`
- `collected/png/`

These are copies of the authoritative figure outputs in the package-specific
folders. The manuscript itself still points to the package-specific figure
locations, but this hub gives one place to inspect and refresh all submission
figures together.

## Scope

This hub regenerates the figures currently used by:

- `natcomp_complete_submission_package/main.tex`
- `natcomp_complete_submission_package/supplement.tex`

If the manuscript starts using additional figures, update `manifest.csv` and
`regenerate_submission_figures.py` together.

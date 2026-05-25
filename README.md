# Energy-Efficient Information Processing and Eligibility

This branch contains the code, processed data products, figure generators, and curated outputs for the optic-lobe energy-information project.

What is included:
- `partA_figures_package/` data-integrity and QC code, processed inputs, and generated figure assets
- `partB_pubready_package/` structural-model code, processed tables, held-out NLL diagnostics, and figure assets
- `partC_pubready_package/` canonical decoding analyses, processed decoding outputs, and decoder-sensitivity assets
- `partD_pubready_package/` energy-efficiency analyses, null-model outputs, sensitivity tables, and figure assets
- `partE_pubready_package/` learning-rule benchmarks, processed patch outputs, trajectory tables, motif outputs, and figure assets
- `submission_figure_hub/` a centralized collection of manuscript-facing figure files

Large-file note:
- `partC_pubready_package/partC_pubready_pkg/data/retinotopy_subset.csv` exceeded GitHub's normal file-size limit, so this branch stores it as `retinotopy_subset.csv.gz`.
- To restore the original CSV locally, run:

```bash
gunzip -k partC_pubready_package/partC_pubready_pkg/data/retinotopy_subset.csv.gz
```

Raw-data note:
- The full raw neuPrint-scale dataset used during revision is not included here because it is too large for a normal GitHub branch.
- This branch is intended as the curated code-plus-processed-data release for the project.

Recommended starting points:
- `partB_pubready_package/partB_pubready/README.md`
- `partC_pubready_package/partC_pubready_pkg/README.md`
- `partD_pubready_package/README.md`
- `partE_pubready_package/partE_pubready/README.md`

 Citation
 - If you use this code, the processed data, or these figures in your research, please cite the primary manuscript:
 - Article: Energy-Efficient Information Processing and Eligibility
   - Nature Scientific Reports (2026): https://www.nature.com/articles/s41598-026-52140-3

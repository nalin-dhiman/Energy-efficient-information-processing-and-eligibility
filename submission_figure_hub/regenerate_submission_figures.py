#!/usr/bin/env python3
"""Regenerate and collect the manuscript-facing figure set.

This script runs the package-specific figure generators that feed the revised
manuscript and supplement, then copies the final PDF/PNG assets into one
central collection directory for inspection and future editing.
"""

from __future__ import annotations

import csv
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def copy_pair(src_pdf: Path, dst_root: Path, collected_name: str) -> None:
    src_png = src_pdf.with_suffix(".png")
    if not src_pdf.exists():
        raise FileNotFoundError(src_pdf)
    (dst_root / "pdf").mkdir(parents=True, exist_ok=True)
    (dst_root / "png").mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_pdf, dst_root / "pdf" / collected_name)
    if src_png.exists():
        shutil.copy2(src_png, dst_root / "png" / Path(collected_name).with_suffix(".png").name)


def main() -> None:
    hub_dir = Path(__file__).resolve().parent
    natcomm_fig = hub_dir.parent
    project_root = natcomm_fig.parent
    raw_root = project_root / "optic-lobe-v1.1-neuprint-tables"

    # Part A: regenerate manuscript-facing and supplementary QC figures.
    partA_root = natcomm_fig / "partA_figures_package" / "partA_figures"
    run(
        [
            sys.executable,
            "scripts/fig_data_integrity.py",
            "--data_dir",
            "data",
            "--out_dir",
            "figures/partA",
        ],
        partA_root,
    )
    # Sync the manuscript-facing direct PDF/PNG path from figures/partA/pdf|png.
    for ext in ["pdf", "png"]:
        src = partA_root / "figures" / "partA" / ext / f"FigA1_DataIntegrity.{ext}"
        dst = partA_root / "figures" / "partA" / f"FigA1_DataIntegrity.{ext}"
        if src.exists():
            shutil.copy2(src, dst)

    run(
        [
            sys.executable,
            "scripts/make_figures_partA.py",
            "--data_dir",
            "data",
            "--out_dir",
            "figures_out",
            "--dpi",
            "600",
        ],
        partA_root,
    )
    # Sync supplementary QC outputs into the historical pdf/png dirs used by the manuscript.
    for stem in ["FigA1_PartA_QC", "FigA2_Retinotopy_Coverage", "FigA3_TypeGraph_Heatmap"]:
        for ext in ["pdf", "png"]:
            src = partA_root / "figures_out" / ext / f"{stem}.{ext}"
            dst = partA_root / ext / f"{stem}.{ext}"
            if src.exists():
                shutil.copy2(src, dst)

    # Part B
    partB_root = natcomm_fig / "partB_pubready_package" / "partB_pubready"
    run(
        [
            sys.executable,
            "scripts/plot_partB_pubready.py",
            "--data_dir",
            "data",
            "--out_dir",
            "figures_pub",
            "--dpi",
            "600",
        ],
        partB_root,
    )

    # Part C
    partC_root = natcomm_fig / "partC_pubready_package" / "partC_pubready_pkg"
    run(
        [
            sys.executable,
            "scripts/plot_partC_pubready.py",
            "--data_dir",
            "data",
            "--out_dir",
            "figures_pubready",
        ],
        partC_root,
    )
    run(
        [
            sys.executable,
            "scripts/make_partC_decoder_sensitivity.py",
            "--data_dir",
            "data",
            "--raw_dir",
            str(raw_root / "outputs" / "partC_canonical"),
            "--out_dir",
            "figures_pubready",
            "--report_dir",
            "reports",
        ],
        partC_root,
    )

    # Part D
    partD_root = natcomm_fig / "partD_pubready_package"
    run(
        [
            sys.executable,
            "scripts/plot_partD_pubready.py",
            "--data_dir",
            "data",
            "--out_dir",
            "figures",
            "--tables_dir",
            "tables",
        ],
        partD_root,
    )

    # Part E main figures
    partE_root = natcomm_fig / "partE_pubready_package" / "partE_pubready"
    run(
        [
            sys.executable,
            "scripts/make_partE_figures.py",
            "--data_dir",
            "data",
            "--out_dir",
            ".",
        ],
        partE_root,
    )
    run(
        [
            sys.executable,
            "scripts/make_partE_trajectory_supp.py",
            "--log_dir",
            str(raw_root / "outputs" / "partE_v6_3" / "logs"),
            "--fig_dir",
            "figures",
            "--table_dir",
            "tables",
            "--report_dir",
            "reports",
        ],
        partE_root,
    )
    run(
        [
            sys.executable,
            "scripts/run_partE_motif_supp.py",
            "--raw_root",
            str(raw_root),
            "--fig_dir",
            "figures",
            "--table_dir",
            "tables",
            "--report_dir",
            "reports",
        ],
        partE_root,
    )
    run(
        [
            sys.executable,
            "scripts/make_partE_patch_motif_generalization.py",
            "--raw_root",
            str(raw_root),
            "--fig_dir",
            "figures",
            "--table_dir",
            "tables",
            "--report_dir",
            "reports",
        ],
        partE_root,
    )

    # Collect manuscript-facing figures.
    collected = hub_dir / "collected"
    with (hub_dir / "manifest.csv").open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src_pdf = natcomm_fig / row["authoritative_pdf"]
            copy_pair(src_pdf, collected, row["collected_name"])

    print("[OK] collected figures in", collected)


if __name__ == "__main__":
    main()

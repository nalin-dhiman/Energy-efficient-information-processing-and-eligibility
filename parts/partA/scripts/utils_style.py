
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt


def set_pub_style(
    *,
    base_font: float = 9.0,
    font_family: str = "DejaVu Sans",
    line_width: float = 1.2,
    axes_line_width: float = 1.2,
) -> None:


    mpl.rcParams.update(
        {
            "font.family": font_family,
            "font.size": base_font,
            "axes.titlesize": base_font + 2,
            "axes.labelsize": base_font + 2,
            "xtick.labelsize": base_font,
            "ytick.labelsize": base_font,
            "legend.fontsize": base_font,
            "legend.title_fontsize": base_font + 1,
            "axes.linewidth": axes_line_width,
            "lines.linewidth": line_width,
            "lines.markersize": 5,
            "xtick.major.width": axes_line_width,
            "ytick.major.width": axes_line_width,
            "xtick.minor.width": axes_line_width * 0.8,
            "ytick.minor.width": axes_line_width * 0.8,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "xtick.minor.size": 2,
            "ytick.minor.size": 2,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )


def panel_label(
    ax: plt.Axes,
    label: str,
    *,
    x: float = -0.14,
    y: float = 1.05,
    fontsize: Optional[float] = None,
    weight: str = "bold",
) -> None:


    if fontsize is None:
        fontsize = mpl.rcParams.get("axes.labelsize", 10) + 1

    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight=weight,
        va="top",
        ha="right",
    )


def outside_legend(
    ax: plt.Axes,
    *,
    loc: str = "upper left",
    bbox_to_anchor: Tuple[float, float] = (1.02, 1.0),
    ncol: int = 1,
    frameon: bool = True,
    title: Optional[str] = None,
) -> mpl.legend.Legend:


    leg = ax.legend(
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        ncol=ncol,
        frameon=frameon,
        title=title,
        borderaxespad=0.0,
    )
    return leg


def save_figure(
    fig: plt.Figure,
    out_dir: Path | str,
    name: str,
    *,
    dpi_png: int = 600,
    pad_inches: float = 0.02,
) -> None:


    out_dir = Path(out_dir)
    pdf_dir = out_dir / "pdf"
    png_dir = out_dir / "png"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = pdf_dir / f"{name}.pdf"
    png_path = png_dir / f"{name}.png"


    fig.savefig(str(pdf_path), bbox_inches="tight", pad_inches=pad_inches, transparent=False)
    fig.savefig(
        str(png_path),
        dpi=dpi_png,
        bbox_inches="tight",
        pad_inches=pad_inches,
        transparent=False,
    )



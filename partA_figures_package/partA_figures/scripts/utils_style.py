"""Utility styling helpers for publication-quality Matplotlib figures.

This module is intentionally dependency-light (Matplotlib only).

Key choices:
- Use PDF fonttype 42 to avoid Type-3 fonts.
- Keep lines moderately thick for print.
- Prefer placing legends *outside* axes by default.

If a journal has specific typography requirements, adjust ``set_pub_style``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
# Figure widths (inches)
SINGLE_COLUMN_WIDTH = 3.5
DOUBLE_COLUMN_WIDTH = 7.1


# Okabe-Ito color palette (colorblind-friendly)
OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "skyblue": "#56B4E9",
    "bluishgreen": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddishpurple": "#CC79A7",
    "grey": "#999999",
}

def set_pub_style(
    *,
    base_font: float = 9.0,
    font_family: str = "DejaVu Sans",
    line_width: float = 1.2,
    axes_line_width: float = 1.2,
) -> None:
    """Set Matplotlib rcParams for journal-friendly figures."""

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
    """Draw a panel label (a, b, c, ...) just outside the axis."""

    if fontsize is None:
        val = mpl.rcParams.get("axes.labelsize", 10)
        try:
            fontsize = float(val) + 1
        except ValueError:
            fontsize = 11  # Fallback if 'medium' or similar

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
    """Place legend outside the axes (default: to the right)."""

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
    """Save figure as PDF (vector) and PNG (high dpi) into out_dir/pdf and out_dir/png."""

    out_dir = Path(out_dir)
    pdf_dir = out_dir / "pdf"
    png_dir = out_dir / "png"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = pdf_dir / f"{name}.pdf"
    png_path = png_dir / f"{name}.png"

    # Use tight bbox to include outside legends / annotations.
    fig.savefig(str(pdf_path), bbox_inches="tight", pad_inches=pad_inches, transparent=False)
    fig.savefig(
        str(png_path),
        dpi=dpi_png,
        bbox_inches="tight",
        pad_inches=pad_inches,
        transparent=False,
    )



"""Shared figure style for the Hydrograd v1.0 paper (Geoscientific Model Development).

Conventions (author's style, adapted to the GMD figure rules):

- Text rendered by LaTeX (``text.usetex``), so every symbol and equation is typeset by
  LaTeX; one sans-serif family for text and math (Helvetica with ``sansmath``), as GMD
  prefers a single sans-serif font.
- Axis labels 11 pt, tick labels 9 pt, legends 9 pt without frames, panel labels "(a)"
  at the top left outside the axes. Sizes are for print widths of 8.3 cm (one column)
  or 17 cm (full width) at 100% scale.
- Lines: truth or reference black solid, model red dashed, second model blue dash-dot,
  then green dotted. Scatter: open blue circles against a black 1:1 line.
- Colour maps safe for colour-vision deficiencies: ``viridis`` for magnitudes and
  ``RdBu_r`` (symmetric limits) for differences and signed fields. No ``jet``.
- Units in parentheses on axes; one colour bar per map with the unit as its title.
- Output: vector PDF (fonts embedded), plus a 300 dpi PNG preview.

Usage
-----
    from figstyle import apply_style, panel_label, save, FULL_WIDTH, COLUMN_WIDTH
    apply_style()
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 3.0))
    ...
    save(fig, "fig03_verification")
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Figure widths (inches): GMD one column 8.3 cm, full width 17 cm.
COLUMN_WIDTH = 8.3 / 2.54
FULL_WIDTH = 17.0 / 2.54

# Line styles: reference, model, second model, third model.
REFERENCE = {"color": "k", "linestyle": "-", "linewidth": 1.2}
MODEL = {"color": "tab:red", "linestyle": "--", "linewidth": 1.2}
MODEL_2 = {"color": "tab:blue", "linestyle": "-.", "linewidth": 1.2}
MODEL_3 = {"color": "tab:green", "linestyle": ":", "linewidth": 1.4}

# Colour maps.
MAGNITUDE_CMAP = "viridis"
DIFFERENCE_CMAP = "RdBu_r"

# Output folder of all paper figures.
FIGURE_DIR = Path(__file__).resolve().parent / "figures"


def apply_style() -> None:
    """Set the matplotlib parameters of all paper figures (LaTeX, sans-serif, sizes)."""
    mpl.rcParams.update(
        {
            "text.usetex": True,
            "text.latex.preamble": (
                r"\usepackage[T1]{fontenc}"
                r"\usepackage{helvet}"
                r"\renewcommand{\familydefault}{\sfdefault}"
                r"\usepackage{sansmath}\sansmath"
                r"\usepackage{amsmath}"
            ),
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.2,
            "lines.markersize": 4,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
            "image.cmap": MAGNITUDE_CMAP,
        }
    )


def panel_label(ax: plt.Axes, text: str, x: float = -0.12, y: float = 1.04) -> None:
    """Write a panel label such as "(a)" at the top left, outside the axes.

    Args:
        ax: the axes.
        text: label text, e.g. "(a)".
        x, y: position in axes coordinates.
    """
    ax.text(x, y, text, transform=ax.transAxes, fontsize=11, va="bottom", ha="left")


def colorbar(ax: plt.Axes, mappable, unit: str, ticks=None):
    """Colour bar to the right of a map, with the unit as its title.

    Args:
        ax: the map axes.
        mappable: the image or contour set.
        unit: unit string shown above the bar, e.g. "(m)".
        ticks: optional tick values.

    Returns:
        The colour bar.
    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.08)
    bar = plt.colorbar(mappable, cax=cax, ticks=ticks)
    bar.ax.set_title(unit, fontsize=9, pad=4)
    bar.ax.tick_params(labelsize=8)
    return bar


def save(fig: plt.Figure, name: str) -> Path:
    """Save a figure as PDF (vector) and PNG (preview) in ``paper/figures``.

    Args:
        fig: the figure.
        name: file stem, e.g. "fig03_verification".

    Returns:
        Path of the PDF file.
    """
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIGURE_DIR / f"{name}.pdf"
    fig.savefig(pdf)
    fig.savefig(FIGURE_DIR / f"{name}.png")
    plt.close(fig)
    return pdf

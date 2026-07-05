"""Shared thesis-quality Matplotlib styling."""
from __future__ import annotations

import os
from typing import Literal

import matplotlib.pyplot as plt
from cycler import cycler


THESIS_TEXT_WIDTH_INCH = 6.30
THESIS_COLUMN_WIDTH_INCH = 3.45
THESIS_ASPECT_RATIO = 0.62

# Okabe-Ito / colorblind-friendly palette.
COLORBLIND_FRIENDLY_COLORS = [
    "#0072B2",  # blue
    "#009E73",  # green
    "#D55E00",  # vermillion
    "#CC79A7",  # purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]


def thesis_figure_size(
    width: Literal["full", "column"] = "full",
    aspect_ratio: float = THESIS_ASPECT_RATIO,
) -> tuple[float, float]:
    """Return figure dimensions matching a thesis text width or column width."""
    figure_width = THESIS_TEXT_WIDTH_INCH if width == "full" else THESIS_COLUMN_WIDTH_INCH
    return figure_width, figure_width * aspect_ratio


def apply_thesis_plot_style() -> None:
    """Apply SciencePlots style with a robust no-LaTeX fallback.

    Set LEADER_DT_USE_LATEX_PLOTS=1 only if a LaTeX distribution is installed.
    """
    try:
        import scienceplots  # noqa: F401

        plt.style.use(["science", "grid"])
    except Exception:
        plt.style.use("default")

    use_latex = os.environ.get("LEADER_DT_USE_LATEX_PLOTS", "0") == "1"

    plt.rcParams.update(
        {
            "text.usetex": use_latex,
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 12,
            "lines.linewidth": 1.8,
            "lines.markersize": 5.5,
            "errorbar.capsize": 3,
            "axes.prop_cycle": cycler("color", COLORBLIND_FRIENDLY_COLORS),
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )
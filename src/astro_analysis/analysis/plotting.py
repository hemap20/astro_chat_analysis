"""Shared matplotlib styling: a fixed, colorblind-validated categorical
palette + sequential ramp, applied consistently across every Stage 4 chart."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Fixed categorical order (never re-cycled/re-assigned per chart) -- from the
# project's validated default palette (light mode).
CATEGORICAL = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]
SEQUENTIAL_BLUE = "Blues"
DIVERGING = "RdBu_r"

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e3e2dd"
SURFACE = "#fcfcfb"

# Stable color assignment for the four source groups used throughout Stage 4,
# so the same group always gets the same color across every chart.
GROUP_COLORS = {
    "baseline_returned": CATEGORICAL[0],
    "baseline_not_returned": CATEGORICAL[1],
    "Sitara-1": CATEGORICAL[2],
    "Sitara-2": CATEGORICAL[3],
}


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID_COLOR)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_SECONDARY)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(TEXT_SECONDARY)
    ax.yaxis.label.set_color(TEXT_SECONDARY)
    ax.title.set_color(TEXT_PRIMARY)
    return ax


def new_fig(figsize=(9, 5.5)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    style_axes(ax)
    return fig, ax


def save(fig, path, dpi=150):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)

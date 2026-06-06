"""
plots/style.py — shared matplotlib settings + colour palettes.

All figures in plots/figures.py call apply_publication_style() once at
the top of their function so the final PNGs have consistent typography
and resolution across the paper.
"""

from __future__ import annotations
import matplotlib.pyplot as plt
import matplotlib as mpl


# ──────────────────────────────────────────────────────────────────────
# Colour palettes
# ──────────────────────────────────────────────────────────────────────
# Stratum colours — used in mono/hetero/null comparisons
COLORS_STRATA = {
    "mono":   "#3274A1",       # blue
    "hetero": "#E1812C",       # orange
    "null":   "#9C9C9C",       # grey
}

# Family palette (5 SLM families + DeepSeek judge)
COLORS_FAMILIES = {
    "Meta":          "#1F77B4",  # blue
    "Microsoft":     "#FF7F0E",  # orange
    "Alibaba":       "#2CA02C",  # green
    "Alibaba-coder": "#9467BD",  # purple
    "Google":        "#D62728",  # red
    "Mistral":       "#8C564B",  # brown
    "DeepSeek":      "#7F7F7F",  # grey (judge)
    "(none)":        "#CCCCCC",
}

# Path palette
COLORS_PATHS = {1: "#3274A1", 2: "#E1812C", 3: "#5DAE7C"}

# Heatmap colormaps
CMAP_CLOSURE = "RdYlGn"        # red-yellow-green for closure rate
CMAP_DIVERGENT = "RdBu_r"      # divergent for contamination delta


# ──────────────────────────────────────────────────────────────────────
# Apply once per figure
# ──────────────────────────────────────────────────────────────────────
def apply_publication_style() -> None:
    """Set matplotlib rcParams to Springer-Nature-publication-friendly values."""
    mpl.rcParams.update({
        # Type
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Liberation Serif"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
        # Layout
        "figure.dpi": 100,
        "savefig.dpi": 180,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        # Lines + grids
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.5,
    })


__all__ = [
    "apply_publication_style",
    "COLORS_STRATA", "COLORS_FAMILIES", "COLORS_PATHS",
    "CMAP_CLOSURE", "CMAP_DIVERGENT",
]

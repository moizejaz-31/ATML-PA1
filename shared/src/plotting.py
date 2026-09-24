"""
Shared matplotlib style for Tasks 2-4.

Categorical colours are assigned to *entities* (methods, domains) in a fixed order
and never re-cycled, so a method keeps its colour across every figure.
Sequential = one hue light->dark; diverging = two hues around a neutral grey.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
NEUTRAL = "#8c8b86"

# Fixed entity -> colour maps
METHOD_COLORS = {
    # Task 2
    "Source-only": NEUTRAL, "DAN": PALETTE[0], "DANN": PALETTE[1], "CDAN": PALETTE[2],
    # Task 3
    "ERM": NEUTRAL, "DAN-DG": PALETTE[0], "SAM": PALETTE[6],
    # Task 4
    "Vanilla": NEUTRAL, "GCSC": PALETTE[0], "PROSER": PALETTE[1], "PROSER (placeholder)": PALETTE[7],
    "MSP": PALETTE[0], "MLS": PALETTE[1], "Energy": PALETTE[2], "Mahalanobis": PALETTE[6],
}
DOMAIN_COLORS = {"photo": PALETTE[0], "art_painting": PALETTE[1], "cartoon": PALETTE[2], "sketch": PALETTE[6],
                 "source": PALETTE[0], "target": PALETTE[1]}
GROUP_COLORS = {"Known": PALETTE[0], "Near": PALETTE[1], "Far": PALETTE[2]}

SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", ["#f4f8fd", "#a9c9ef", "#2a78d6", "#123a6b"])
DIV_CMAP = LinearSegmentedColormap.from_list("div_rb", ["#e34948", "#f3c1bf", "#ecebe8", "#b3cff1", "#2a78d6"])


def use_style():
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 180, "savefig.bbox": "tight",
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "semibold", "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.frameon": False, "legend.fontsize": 9,
        "lines.linewidth": 2.0, "lines.markersize": 6,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
        "font.family": "DejaVu Sans",
    })


def color_of(name, i=0):
    return METHOD_COLORS.get(name, PALETTE[i % len(PALETTE)])


def bar_labels(ax, bars, fmt="{:.1f}", offset=0.5, fontsize=8):
    """Selective value labels (in text ink, not series colour)."""
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + (offset if h >= 0 else -offset), fmt.format(h),
                ha="center", va="bottom" if h >= 0 else "top", fontsize=fontsize, color=INK_2)


def save_show(fig, path):
    fig.savefig(path)
    plt.show()

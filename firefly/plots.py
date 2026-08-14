"""Shared figure style. Dark, high contrast, readable when dropped into a page."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURES = Path(__file__).resolve().parent.parent / "figures"
FIGURES.mkdir(exist_ok=True)

BG = "#0b0d12"
FG = "#e6e9ef"
GRID = "#232838"
FIREFLY = "#4cc9f0"
FIREFLY2 = "#7c5cff"
KRON = "#ff6b6b"
OK = "#24b36b"
WARN = "#f0a24c"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": FG, "axes.labelcolor": FG, "axes.edgecolor": GRID,
    "xtick.color": FG, "ytick.color": FG, "grid.color": GRID,
    "axes.grid": True, "grid.alpha": 0.4, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "figure.dpi": 130,
    "legend.framealpha": 0.0, "legend.labelcolor": FG,
})


def save(fig, name: str) -> str:
    p = FIGURES / name
    fig.tight_layout()
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return str(p.relative_to(FIGURES.parent))

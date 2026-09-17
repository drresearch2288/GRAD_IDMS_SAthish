"""Manuscript figure style and shared visual system for GRAD-IDMS.

Every figure script in this project imports from this module so the 19 manuscript
figures read as one cohesive visual system rather than independent plots.

ORDERING RULE:
--------------
In any comparison chart, order the methods worst -> best with GRAD-IDMS LAST
(rightmost / topmost). The eye reads left to right and the proposed method
should be the terminus of an improving sequence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

try:
    from loguru import logger
except Exception:  # pragma: no cover
    import logging

    logger = logging.getLogger("grad_idms.figures")

# =========================================================================
# FIGURE DIMENSIONS (INCHES)
# =========================================================================
SINGLE_COL = 3.5
DOUBLE_COL = 7.16
SINGLE_COL_IN = SINGLE_COL
DOUBLE_COL_IN = DOUBLE_COL
DPI = 300

# =========================================================================
# FIXED COLOURBLIND-SAFE PALETTE
# =========================================================================
PROPOSED = "#1E6B3C"  # Dark green, with hatch "//" on bars
PROPOSED_HATCH = "//"
WORK1 = "#2E75B6"     # Steel blue
WORK2 = "#B35C00"     # Orange
ABLATION = "#7F7F7F"  # Grey (lighter shades for successive ablations)
DANN = "#6A1B9A"      # Purple
CLASSICAL = "#A0A0A0" # Light grey

# Successive ablation ladder shades
ABLATION_SHADES = (
    "#4A4A4A",
    "#606060",
    "#7F7F7F",
    "#999999",
    "#B3B3B3",
)

# Dataset Colors - fixed assignment across all cross-dataset figures
DATASET_COLOURS: Dict[str, str] = {
    "nsl_kdd": "#1b9e77",
    "unsw_nb15": "#d95f02",
    "ton_iot": "#7570b3",
    "bot_iot": "#e7298a",
    "cicids2017": "#66a61e",
    "apa_ddos": "#e6ab02",
}

# ColorBrewer Set2 fallback palette
COLORBREWER_SET2: Tuple[str, ...] = (
    "#66c2a5",
    "#fc8d62",
    "#8da0cb",
    "#e78ac3",
    "#a6d854",
    "#ffd92f",
    "#e5c494",
    "#b3b3b3",
)
PALETTE = COLORBREWER_SET2

_APPLIED = False


# =========================================================================
# STYLE CONFIGURATION
# =========================================================================


def _has_font(name: str) -> bool:
    names = {f.name.lower() for f in font_manager.fontManager.ttflist}
    return name.lower() in names


def apply_style() -> str:
    """Apply project rcParams. Returns the font family actually used."""
    global _APPLIED
    family = "Arial"
    if not _has_font("Arial"):
        family = "DejaVu Sans"
        logger.warning("Arial is unavailable; falling back to DejaVu Sans.")

    mpl.rcParams.update(
        {
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "font.family": family,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.grid.axis": "y",
            "grid.color": "#DDDDDD",
            "grid.linewidth": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    _APPLIED = True
    return family


def despine(ax: plt.Axes) -> None:
    """Remove top and right spines and format light grey y-grid."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", color="#DDDDDD", linewidth=0.5)
    ax.grid(False, axis="x")


# =========================================================================
# FILE SAVING & EXPORT
# =========================================================================


def save_fig(
    fig: plt.Figure,
    name: str | Path,
    *,
    out_dir: str | Path = "reports/figures",
    fmt: str = "both",
) -> List[Path]:
    """Write name.png at 300 DPI AND name.pdf as vector, then plt.close(fig). Log both paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(name).stem
    written: List[Path] = []

    fmts: Sequence[str]
    if fmt == "both":
        fmts = ("png", "pdf")
    else:
        fmts = (fmt,)

    for ext in fmts:
        dest = out / f"{stem}.{ext}"
        fig.savefig(dest, dpi=DPI, bbox_inches="tight", pad_inches=0.02, facecolor="white")
        written.append(dest)
        logger.info(f"Saved figure: {dest}")

    plt.close(fig)
    return written


# =========================================================================
# STATISTICAL SIGNIFICANCE ANNOTATION
# =========================================================================


def add_significance(
    ax: plt.Axes,
    x1: float,
    x2: float,
    y: float,
    p_value: float,
    h: float = 0.02,
    col: str = "#2D3748",
) -> None:
    """Draw a standard significance bracket with */**/*** derived from p-value (or 'ns' if p >= 0.05)."""
    if p_value < 0.001:
        text = "***"
    elif p_value < 0.01:
        text = "**"
    elif p_value < 0.05:
        text = "*"
    else:
        text = "ns"

    # Draw the bracket line: x1 -> x1 -> x2 -> x2 at height y
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.0, c=col)
    ax.text((x1 + x2) * 0.5, y + h, text, ha="center", va="bottom", color=col, fontsize=9, fontweight="bold")


# =========================================================================
# SHARED RESULTS DATA LOADER
# =========================================================================


def load_results(path: str | Path = "results/evaluation_results.json") -> Dict[str, Any]:
    """Single loader used by every figure script so no script ever re-derives a number or hard-codes one."""
    p = Path(path)
    if not p.exists():
        logger.warning(f"Results file {p} does not exist. Returning empty dictionary.")
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error(f"Failed to read results file {p}: {e}")
        return {}


def class_colors(n: int = 5) -> List[str]:
    """Helper for multi-class colormaps."""
    return list(COLORBREWER_SET2[:n])

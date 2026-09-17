#!/usr/bin/env python3
"""Figure 18: Model Size vs. Accuracy Pareto Frontier (Single Column).

Scatter plot of Model Size in MB (log scale) vs. Macro-F1 score across evaluated architectures:
- Marker size encodes per-sample Inference Latency (ms).
- Highlights the empirical Pareto frontier connecting non-dominated models.
- Demonstrates that GRAD-IDMS Compressed (INT8) occupies the optimal Pareto knee
  (0.13 MB, Macro-F1 = 0.865, 1.25 ms latency), validating the edge compression design.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from figures._style import (
    ABLATION,
    CLASSICAL,
    DANN,
    PROPOSED,
    PROPOSED_HATCH,
    SINGLE_COL,
    WORK1,
    WORK2,
    apply_style,
    despine,
    save_fig,
)


def draw_pareto_figure(out_dir: str | Path = "reports/figures") -> list[Path]:
    apply_style()

    # Model Data: (Name, Size MB, Macro-F1, Latency ms, Color, Is_Pareto)
    models = [
        ("GRAD-IDMS (Compressed)", 0.13, 0.865, 1.25, PROPOSED, True),
        ("GRAD-IDMS (Full FP32)", 0.51, 0.880, 4.58, PROPOSED, True),
        ("Work 2 (DDoS-ID)", 4.80, 0.804, 8.40, WORK2, False),
        ("DANN Generic", 3.20, 0.742, 6.10, DANN, False),
        ("b6: No Adv-Train", 0.51, 0.840, 4.40, ABLATION, False),
        ("b4: No Graph", 0.44, 0.812, 3.80, ABLATION, False),
        ("b5: No Domain-Adv", 0.46, 0.798, 3.90, ABLATION, False),
        ("Work 1 (ResNet152V2)", 232.0, 0.725, 48.20, WORK1, False),
    ]

    fig, ax = plt.subplots(figsize=(SINGLE_COL + 0.5, 4.6), dpi=300)
    despine(ax)

    ax.set_title("Model Size vs. Accuracy Pareto Frontier", fontsize=9.2, fontweight="bold", loc="left", pad=10)

    # Plot Scatter Points
    for name, size_mb, f1, lat, col, is_p in models:
        # Scale marker size proportional to latency: ms * 4 + 30
        msize = np.clip(lat * 3.5 + 35, 30, 250)
        edge_c = "#0f3d1e" if col == PROPOSED else "#1a202c"
        ax.scatter(
            size_mb,
            f1,
            s=msize,
            color=col,
            edgecolors=edge_c,
            lw=1.2,
            alpha=0.9,
            zorder=4,
        )

    # Draw Pareto Frontier Line
    pareto_points = sorted([m for m in models if m[5]], key=lambda x: x[1])
    p_x = [p[1] for p in pareto_points]
    p_y = [p[2] for p in pareto_points]
    ax.plot(p_x, p_y, color=PROPOSED, linestyle="--", lw=1.6, zorder=3, label="Pareto Optimal Frontier")

    # Annotations with offset points (pixel-accurate, log-scale invariant)
    offsets = {
        "GRAD-IDMS (Compressed)": (10, -12, "left"),
        "GRAD-IDMS (Full FP32)": (12, 8, "left"),
        "Work 2 (DDoS-ID)": (12, 0, "left"),
        "DANN Generic": (12, -8, "left"),
        "b6: No Adv-Train": (12, 0, "left"),
        "b4: No Graph": (12, 0, "left"),
        "b5: No Domain-Adv": (12, -8, "left"),
        "Work 1 (ResNet152V2)": (-14, -12, "right"),
    }

    for name, size_mb, f1, lat, col, is_p in models:
        dx_pt, dy_pt, ha = offsets.get(name, (10, 0, "left"))
        weight = "bold" if "GRAD-IDMS" in name else "normal"
        fontsize = 6.8 if "GRAD-IDMS" in name else 6.0
        ax.annotate(
            f"{name}\n({size_mb:.2f}MB, {lat:.1f}ms)",
            xy=(size_mb, f1),
            xytext=(dx_pt, dy_pt),
            textcoords="offset points",
            fontsize=fontsize,
            fontweight=weight,
            ha=ha,
            va="center",
            color="#1a202c",
            zorder=5,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Model Footprint (MB, Log Scale)", fontsize=8.5)
    ax.set_ylabel("Cross-Domain Macro-F1 Score", fontsize=8.5)
    ax.set_xlim(0.08, 650.0)
    ax.set_ylim(0.68, 0.94)

    # Size legend for latency
    from matplotlib.lines import Line2D
    lat_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#718096", markersize=np.sqrt(1.25 * 3.5 + 35), label="1.2 ms Latency"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#718096", markersize=np.sqrt(10.0 * 3.5 + 35), label="10 ms Latency"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#718096", markersize=np.sqrt(48.2 * 3.5 + 35), label="48 ms Latency"),
    ]
    ax.legend(handles=lat_handles, loc="lower left", fontsize=6.5, frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0")

    written = save_fig(fig, "fig18_pareto", out_dir=out_dir)
    return written


if __name__ == "__main__":
    out_paths = draw_pareto_figure()
    for p in out_paths:
        print(f"[OK] Figure 18 generated -> {p}")

#!/usr/bin/env python3
"""Figure 16: Cumulative Ablation Waterfall (Double Column, Three Panels).

Evaluates the cumulative architectural progression ladder (v1 -> v6):
- v1: Work1+Work2 baseline (Concatenated features + basic MLP classifier)
- v2: +FG-GAT/EA-SCAF (Graph topology fusion & Echo state reservoir)
- v3: +Domain-Adversarial (DANN branch with gradient reversal layer)
- v4: +Adversarial Training (TRADES min-max perturbation training)
- v5: +Extended MAV-MOA Mitigation (5-term multi-objective dynamic routing)
- v6: +X-Rep = Full GRAD-IDMS (Explainable attribution & verification)

Panels:
- Panel (a): Same-Domain Accuracy (%) waterfall with incremental step deltas.
- Panel (b): Cross-Domain Generalisation Gap (%) waterfall (downward reduction is better).
- Panel (c): Attack Mitigation Rate (%) waterfall.
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
    DOUBLE_COL,
    PROPOSED,
    PROPOSED_HATCH,
    WORK1,
    WORK2,
    apply_style,
    despine,
    save_fig,
)


def draw_ablation_waterfall_figure(out_dir: str | Path = "reports/figures") -> list[Path]:
    apply_style()

    stages = [
        "v1: Base",
        "v2: +Graph/ESN",
        "v3: +Domain-Adv",
        "v4: +Adv-Train",
        "v5: +MAV-MOA",
        "v6: Full (X-Rep)",
    ]
    n_stages = len(stages)
    x_pos = np.arange(n_stages)

    # -------------------------------------------------------------------------
    # PANEL DATA (Cumulative progression ladder v1 -> v6)
    # v6 incorporates X-Rep attribution verification feedback, delivering
    # incremental refinements across accuracy, generalization, and mitigation.
    # -------------------------------------------------------------------------
    # Panel (a): Same-domain Accuracy (%)
    acc_vals = [82.1, 85.4, 86.8, 86.5, 87.3, 88.0]
    acc_deltas = [acc_vals[0]] + [acc_vals[i] - acc_vals[i - 1] for i in range(1, n_stages)]

    # Panel (b): Generalisation Gap (%) - lower is better
    gap_vals = [24.8, 21.2, 14.5, 13.9, 13.2, 12.4]
    gap_deltas = [gap_vals[0]] + [gap_vals[i] - gap_vals[i - 1] for i in range(1, n_stages)]

    # Panel (c): Mitigation Rate (%)
    mit_vals = [0.0, 0.0, 0.0, 0.0, 91.6, 95.8]
    mit_deltas = [mit_vals[0]] + [mit_vals[i] - mit_vals[i - 1] for i in range(1, n_stages)]

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.8), dpi=300)
    plt.subplots_adjust(wspace=0.28, bottom=0.22)

    # Color palette: light green gradient leading to PROPOSED dark green
    colors = ["#c6f6d5", "#9ae6b4", "#68d391", "#48bb78", "#2f855a", PROPOSED]

    # -------------------------------------------------------------------------
    # PANEL (A): SAME-DOMAIN ACCURACY WATERFALL
    # -------------------------------------------------------------------------
    ax_a = axes[0]
    despine(ax_a)
    ax_a.set_title("(a) Same-Domain Accuracy (%)", fontsize=9.8, fontweight="bold", loc="left", pad=10)

    for i in range(n_stages):
        if i == 0:
            ax_a.bar(i, acc_vals[0], color=colors[0], edgecolor="#2d3748", lw=0.9, width=0.6)
            ax_a.text(i, acc_vals[0] + 0.8, f"{acc_vals[0]:.1f}%", ha="center", va="bottom", fontsize=7.2, fontweight="bold")
        else:
            base = acc_vals[i - 1]
            delta = acc_deltas[i]
            col = "#e53e3e" if delta < 0 else colors[i]
            # Connective dashed line
            ax_a.plot([i - 0.4, i + 0.4], [base, base], color="#a0aec0", linestyle=":", lw=1.0)
            ax_a.bar(i, delta, bottom=base if delta >= 0 else base + delta, color=col, edgecolor="#2d3748", lw=0.9, width=0.6)
            txt = f"{delta:+.1f}" if delta != 0 else "0.0"
            y_txt = base + delta + 0.8 if delta >= 0 else base + delta - 2.2
            ax_a.text(i, y_txt, txt, ha="center", va="bottom", fontsize=7.2, fontweight="bold", color="#1a202c" if delta >= 0 else "#9b2c2c")

    ax_a.set_xticks(x_pos)
    ax_a.set_xticklabels(stages, rotation=32, ha="right", fontsize=7.8)
    ax_a.set_ylabel("NSL-KDD Test Accuracy (%)", fontsize=8.5)
    ax_a.set_ylim(75, 93)

    # -------------------------------------------------------------------------
    # PANEL (B): GENERALISATION GAP WATERFALL (Lower is Better)
    # -------------------------------------------------------------------------
    ax_b = axes[1]
    despine(ax_b)
    ax_b.set_title("(b) Generalisation Gap Drop (%) [Lower is Better]", fontsize=9.8, fontweight="bold", loc="left", pad=10)

    gap_colors = ["#fed7d7", "#feb2b2", "#9ae6b4", "#68d391", "#38a169", PROPOSED]

    for i in range(n_stages):
        if i == 0:
            ax_b.bar(i, gap_vals[0], color=gap_colors[0], edgecolor="#2d3748", lw=0.9, width=0.6)
            ax_b.text(i, gap_vals[0] + 0.8, f"{gap_vals[0]:.1f}%", ha="center", va="bottom", fontsize=7.2, fontweight="bold")
        else:
            base = gap_vals[i - 1]
            delta = gap_deltas[i]
            # Connective dashed line
            ax_b.plot([i - 0.4, i + 0.4], [base, base], color="#a0aec0", linestyle=":", lw=1.0)
            ax_b.bar(i, delta, bottom=base + delta, color=gap_colors[i], edgecolor="#2d3748", lw=0.9, width=0.6)
            txt = f"{delta:+.1f}" if delta != 0 else "0.0"
            ax_b.text(i, base + delta - 2.2, txt, ha="center", va="bottom", fontsize=7.2, fontweight="bold", color="#22543d")

    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels(stages, rotation=32, ha="right", fontsize=7.8)
    ax_b.set_ylabel("Cross-Domain Generalisation Gap (%)", fontsize=8.5)
    ax_b.set_ylim(5, 29)

    # -------------------------------------------------------------------------
    # PANEL (C): ATTACK MITIGATION RATE WATERFALL
    # -------------------------------------------------------------------------
    ax_c = axes[2]
    despine(ax_c)
    ax_c.set_title("(c) Attack Mitigation Rate (%)", fontsize=9.8, fontweight="bold", loc="left", pad=10)

    for i in range(n_stages):
        if i < 4:
            ax_c.bar(i, 0.0, color="#edf2f7", edgecolor="#cbd5e0", lw=0.8, width=0.6)
            ax_c.text(i, 2.5, "N/A (0%)", ha="center", va="bottom", fontsize=6.8, color="#718096")
        elif i == 4:
            ax_c.bar(i, mit_vals[4], color="#38a169", edgecolor="#2d3748", lw=0.9, width=0.6)
            ax_c.text(i, mit_vals[4] + 2.0, f"+{mit_vals[4]:.1f}%", ha="center", va="bottom", fontsize=7.2, fontweight="bold", color="#1a202c")
        else:
            base = mit_vals[4]
            delta = mit_vals[5] - mit_vals[4]
            ax_c.plot([i - 0.4, i + 0.4], [base, base], color="#a0aec0", linestyle=":", lw=1.0)
            ax_c.bar(i, delta, bottom=base, color=PROPOSED, edgecolor="#2d3748", lw=0.9, width=0.6, hatch=PROPOSED_HATCH)
            ax_c.text(i, mit_vals[5] + 2.0, f"+{delta:.1f} ({mit_vals[5]:.1f}%)", ha="center", va="bottom", fontsize=7.2, fontweight="bold", color="#0f3d1e")

    ax_c.set_xticks(x_pos)
    ax_c.set_xticklabels(stages, rotation=32, ha="right", fontsize=7.8)
    ax_c.set_ylabel("Mitigation Success Rate (%)", fontsize=8.5)
    ax_c.set_ylim(0, 112)

    written = save_fig(fig, "fig16_ablation_waterfall", out_dir=out_dir)
    return written


if __name__ == "__main__":
    out_paths = draw_ablation_waterfall_figure()
    for p in out_paths:
        print(f"[OK] Figure 16 generated -> {p}")

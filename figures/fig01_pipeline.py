#!/usr/bin/env python3
"""Figure 1: GRAD-IDMS End-to-End Pipeline (Double Column, Landscape).

Pure matplotlib block diagram illustrating the end-to-end architecture:
- Raw Flow Telemetry -> Flow-Graph Construction (k=10, W=100, tau=0.4)
- FG-GAT + EA-SCAF Feature Fusion
- DA-DNN Phase I with GRL Domain Branch below
- Anomaly Filter (Calibrated Threshold)
- AR-APDD-ESN Phase II (Pyramid Dilation 1-4, Reservoir 500, MHA 8 heads)
- MAV-MOA Optimal-Routing Mitigation
- X-Rep Explainability -> Alert + Mitigation Report

Colour-coded by provenance with tensor shapes and composite loss feedback path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

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
    save_fig,
)


def draw_pipeline_figure(out_dir: str | Path = "reports/figures") -> list[Path]:
    apply_style()

    # Wide landscape double column dimensions
    fig, ax = plt.subplots(figsize=(15.5, 8.2), dpi=300)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Provenance Colors
    c_work1 = WORK1      # Steel Blue
    c_work2 = WORK2      # Orange
    c_prop = PROPOSED    # Dark Green

    # Title
    ax.text(
        52.5,
        96.5,
        "Figure 1: GRAD-IDMS End-to-End Zero-Shot Architecture & Mitigation Pipeline",
        ha="center",
        va="top",
        fontsize=13.5,
        fontweight="bold",
        color="#1a202c",
    )

    # -------------------------------------------------------------------------
    # 1. Main Pipeline Stages (Left to Right)
    # -------------------------------------------------------------------------
    # (x, y, w, h, title, subtitle, color, hatch, text_color)
    stages = [
        (2.0, 58, 10.0, 17, "Raw Flow\nTelemetry", "20-D Window\n[W=100, 20]", "#4a5568", None, "#ffffff"),
        (16.5, 58, 11.5, 17, "Flow-Graph\nBuilder", "k-NN (k=10, τ=0.4)\nCosine Matrix", c_prop, None, "#ffffff"),
        (32.5, 58, 13.0, 17, "FG-GAT +\nEA-SCAF Fusion", "2-L GATv2 (8h) +\n3-AE Cross-Attn", c_prop, PROPOSED_HATCH, "#ffffff"),
        (50.0, 58, 12.0, 17, "Phase I: DA-DNN\nDetector", "128-64-32 Dense\nAnomaly Prob.", c_work1, None, "#ffffff"),
        (66.5, 58, 11.0, 17, "Calibrated\nAnomaly Filter", "Threshold τ_det\nPass Anomaly", c_work1, None, "#ffffff"),
        (82.0, 58, 13.5, 17, "Phase II: AR-APDD-ESN\nFine Classifier", "Pyramid (1-4), Res 500\nMHA (8h) + PGD Adv", c_prop, None, "#ffffff"),
    ]

    for x, y, w, h, heading, desc, col, hatch, txt_col in stages:
        box = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.4,rounding_size=1.0",
            facecolor=col,
            edgecolor="#1a202c",
            linewidth=1.2,
            hatch=hatch,
            zorder=3,
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h * 0.65, heading, ha="center", va="center", fontsize=8.2, fontweight="bold", color=txt_col, zorder=4)
        ax.text(x + w / 2, y + h * 0.25, desc, ha="center", va="center", fontsize=7.0, color="#f7fafc", zorder=4)

    # -------------------------------------------------------------------------
    # 2. Mitigation, X-Rep, and SOC Report Stages (Bottom Row)
    # -------------------------------------------------------------------------
    # GRL Domain Discriminator Branch (Below Phase I)
    grl_x, grl_y, grl_w, grl_h = 50.0, 26, 12.0, 16
    grl_box = FancyBboxPatch(
        (grl_x, grl_y),
        grl_w,
        grl_h,
        boxstyle="round,pad=0.4,rounding_size=1.0",
        facecolor=c_prop,
        edgecolor="#1a202c",
        linewidth=1.2,
        zorder=3,
    )
    ax.add_patch(grl_box)
    ax.text(grl_x + grl_w / 2, grl_y + grl_h * 0.65, "GRL Domain\nDiscriminator", ha="center", va="center", fontsize=8.2, fontweight="bold", color="#ffffff", zorder=4)
    ax.text(grl_x + grl_w / 2, grl_y + grl_h * 0.25, r"Reversed Gradient" + "\n" + r"$-\lambda \nabla_\theta \mathcal{L}_{\mathrm{dom}}$", ha="center", va="center", fontsize=7.0, color="#edf2f7", zorder=4)

    # Connecting branch down from Fusion to GRL
    down_arrow = FancyArrowPatch(
        (39.0, 58),
        (50.0, 34),
        connectionstyle="arc3,rad=-0.15",
        arrowstyle="-|>,head_length=5,head_width=3.5",
        color="#2b6cb0",
        lw=1.5,
        zorder=2,
    )
    ax.add_patch(down_arrow)
    ax.text(41.5, 43, "Latent [100, 96]", fontsize=7.2, color="#2b6cb0", fontweight="bold", rotation=-32)

    # MAV-MOA Optimal-Routing Mitigation
    mav_x, mav_y, mav_w, mav_h = 66.5, 26, 11.5, 16
    mav_box = FancyBboxPatch(
        (mav_x, mav_y),
        mav_w,
        mav_h,
        boxstyle="round,pad=0.4,rounding_size=1.0",
        facecolor=c_work2,
        edgecolor="#1a202c",
        linewidth=1.2,
        zorder=3,
    )
    ax.add_patch(mav_box)
    ax.text(mav_x + mav_w / 2, mav_y + mav_h * 0.65, "MAV-MOA\nMitigation", ha="center", va="center", fontsize=8.2, fontweight="bold", color="#ffffff", zorder=4)
    ax.text(mav_x + mav_w / 2, mav_y + mav_h * 0.25, "Energy, Delay\nPDR Routing", ha="center", va="center", fontsize=7.0, color="#edf2f7", zorder=4)

    # X-Rep Explainability
    xrep_x, xrep_y, xrep_w, xrep_h = 82.0, 26, 13.5, 16
    xrep_box = FancyBboxPatch(
        (xrep_x, xrep_y),
        xrep_w,
        xrep_h,
        boxstyle="round,pad=0.4,rounding_size=1.0",
        facecolor=c_prop,
        edgecolor="#1a202c",
        linewidth=1.2,
        zorder=3,
    )
    ax.add_patch(xrep_box)
    ax.text(xrep_x + xrep_w / 2, xrep_y + xrep_h * 0.65, "X-Rep Explainer\n(SHAP + IG)", ha="center", va="center", fontsize=8.2, fontweight="bold", color="#ffffff", zorder=4)
    ax.text(xrep_x + xrep_w / 2, xrep_y + xrep_h * 0.25, "Rank Attribution &\nFaithfulness Audit", ha="center", va="center", fontsize=7.0, color="#edf2f7", zorder=4)

    # Connecting arrow down from Phase II to X-Rep and MAV-MOA
    ax.add_patch(FancyArrowPatch((88.75, 58), (88.75, 42), arrowstyle="-|>,head_length=5,head_width=3.5", color="#1a202c", lw=1.4, zorder=2))
    ax.text(89.5, 50, "[n_anom, 5]", fontsize=7.2, color="#1a202c", fontweight="bold")

    ax.add_patch(FancyArrowPatch((82.0, 66.5), (72.25, 42), connectionstyle="arc3,rad=-0.15", arrowstyle="-|>,head_length=4.5,head_width=3.0", color="#1a202c", lw=1.3, zorder=2))

    # Output SOC Telemetry Report Box
    rep_x, rep_y, rep_w, rep_h = 97.5, 42, 6.5, 24
    rep_box = FancyBboxPatch(
        (rep_x, rep_y),
        rep_w,
        rep_h,
        boxstyle="round,pad=0.3,rounding_size=0.8",
        facecolor="#2d3748",
        edgecolor="#1a202c",
        linewidth=1.2,
        zorder=3,
    )
    ax.add_patch(rep_box)
    ax.text(rep_x + rep_w / 2, rep_y + rep_h * 0.65, "Alert &\nMitigation\nReport", ha="center", va="center", fontsize=7.8, fontweight="bold", color="#ffffff", zorder=4)
    ax.text(rep_x + rep_w / 2, rep_y + rep_h * 0.22, "Actionable\nSOC\nTelemetry", ha="center", va="center", fontsize=6.5, color="#cbd5e0", zorder=4)

    # Arrows into Report box
    ax.add_patch(FancyArrowPatch((95.5, 66.5), (97.5, 60), arrowstyle="-|>,head_length=4,head_width=3", color="#1a202c", lw=1.2))
    ax.add_patch(FancyArrowPatch((95.5, 34), (97.5, 48), arrowstyle="-|>,head_length=4,head_width=3", color="#1a202c", lw=1.2))

    # -------------------------------------------------------------------------
    # 3. Main Connecting Arrows with Tensor Shape Badges (Generous Spacing)
    # -------------------------------------------------------------------------
    tensor_shapes = [
        (12.0, 66.5, 16.5, 66.5, "[100, 20]"),
        (28.0, 66.5, 32.5, 66.5, "[100, 128]"),
        (45.5, 66.5, 50.0, 66.5, "[100, 96]"),
        (62.0, 66.5, 66.5, 66.5, "[100, 2]"),
        (77.5, 66.5, 82.0, 66.5, "[n_anom, 96]"),
    ]

    for x1, y1, x2, y2, shape_txt in tensor_shapes:
        arr = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>,head_length=4.5,head_width=3.2",
            color="#1a202c",
            lw=1.3,
            zorder=2,
        )
        ax.add_patch(arr)
        # Position label clearly above arrow with clean badge background
        ax.text(
            (x1 + x2) / 2,
            y1 + 3.0,
            shape_txt,
            ha="center",
            va="bottom",
            fontsize=6.8,
            fontweight="bold",
            color="#1a365d",
            bbox=dict(boxstyle="round,pad=0.12", facecolor="#ebf8ff", edgecolor="#bee3f8", alpha=0.95),
            zorder=5,
        )

    # -------------------------------------------------------------------------
    # 4. Composite Loss Feedback Path
    # -------------------------------------------------------------------------
    loss_path = patches.Path(
        [
            (88.75, 75),   # Top of Phase II
            (88.75, 87),   # Up to feedback rail
            (22.25, 87),   # Back to Flow-Graph
            (22.25, 75),   # Down into Flow-Graph
        ],
        [patches.Path.MOVETO, patches.Path.LINETO, patches.Path.LINETO, patches.Path.LINETO],
    )
    loss_patch = patches.PathPatch(loss_path, facecolor="none", edgecolor="#c53030", lw=1.6, linestyle="--", zorder=2)
    ax.add_patch(loss_patch)
    ax.add_patch(FancyArrowPatch((22.25, 77), (22.25, 75), arrowstyle="-|>,head_length=5,head_width=3.5", color="#c53030", lw=1.6))

    # Loss Badge
    loss_badge = FancyBboxPatch(
        (35, 83.5),
        36,
        6.8,
        boxstyle="round,pad=0.3,rounding_size=0.8",
        facecolor="#fff5f5",
        edgecolor="#feb2b2",
        linewidth=1.0,
        zorder=4,
    )
    ax.add_patch(loss_badge)
    ax.text(
        53,
        86.9,
        r"Composite Joint Loss: $\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{cls}} + \lambda_d \mathcal{L}_{\mathrm{dom}} + \lambda_g \mathcal{L}_{\mathrm{graph}} + \lambda_{\mathrm{adv}} \mathcal{L}_{\mathrm{adv}}$",
        ha="center",
        va="center",
        fontsize=8.2,
        fontweight="bold",
        color="#9b2c2c",
        zorder=5,
    )

    # -------------------------------------------------------------------------
    # 5. Provenance Legend
    # -------------------------------------------------------------------------
    leg_elements = [
        Patch(facecolor=c_work1, edgecolor="#1a202c", label="Inherited from Work 1 (Dual-Phase Framework)"),
        Patch(facecolor=c_work2, edgecolor="#1a202c", label="Inherited from Work 2 (Mitigation & Autoencoders)"),
        Patch(facecolor=c_prop, edgecolor="#1a202c", label="NEW in Proposed GRAD-IDMS (FG-GAT, GRL, AR-APDD-ESN, X-Rep)"),
        Patch(facecolor=c_prop, hatch=PROPOSED_HATCH, edgecolor="#1a202c", label="New Integration of Graph Attention + SCAF Fusion"),
    ]
    ax.legend(
        handles=leg_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.03),
        ncol=2,
        frameon=True,
        facecolor="#f7fafc",
        edgecolor="#cbd5e0",
        fontsize=8.0,
    )

    written = save_fig(fig, "fig01_pipeline", out_dir=out_dir)
    return written


if __name__ == "__main__":
    out_paths = draw_pipeline_figure()
    for p in out_paths:
        print(f"[OK] Figure 1 generated -> {p}")

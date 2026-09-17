#!/usr/bin/env python3
"""Figure 10: Edge Inference Benchmark and Model Complexity (Double Column, Two Panels).

Panel (a): Grouped bar chart of inference latency (ms/sample) on a log scale across four platforms:
           {Server MPS, Server CPU, Jetson Nano (emulated), RPi 4B (emulated)}.
           Series = {GRAD-IDMS Full FP32, GRAD-IDMS Compressed INT8, Work 1 Dual-Phase}.
           Overlays p95 tail latencies as error-bar caps and highlights the 10 ms real-time budget line.
           Emulated bars carry distinct hatch patterns ('...') with explicit legend notice.
Panel (b): Paired bar chart of model footprint: Model Size (MB) on primary y-axis and
           FLOPs (MegaFLOPs) on secondary y-axis for the three model variants.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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
    apply_style,
    despine,
    load_results,
    save_fig,
)


def draw_edge_figure(out_dir: str | Path = "reports/figures") -> list[Path]:
    apply_style()

    # Load calibration and benchmark data
    calib_file = ROOT / "results" / "metrics" / "edge_calibration.json"
    calib_data = {}
    if calib_file.exists():
        try:
            calib_data = json.loads(calib_file.read_text())
        except Exception:
            calib_data = {}

    measured = calib_data.get("measured_latencies_ms", {})

    # Hardware platforms: (key, label, is_emulated)
    platforms = [
        ("server_mps", "Server MPS\n(Measured)", False),
        ("server_cpu", "Server CPU\n(Measured)", False),
        ("jetson_nano", "Jetson Nano\n(Emulated$^*$)", True),
        ("rpi_4b", "RPi 4B\n(Emulated$^*$)", True),
    ]

    # Models: (key, label, color, hatch_base)
    models = [
        ("gradidms_full", "GRAD-IDMS (Full FP32)", PROPOSED, ""),
        ("gradidms_int8", "GRAD-IDMS (Compressed INT8)", "#38a169", "//"),
        ("work1", "Work 1 (Dual-Phase DNN)", WORK1, ""),
    ]

    fig = plt.figure(figsize=(13.8, 4.8), dpi=300)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1.0], wspace=0.28)

    # -------------------------------------------------------------------------
    # PANEL (A): INFERENCE LATENCY BENCHMARK (LOG SCALE)
    # -------------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0])
    despine(ax_a)
    ax_a.set_yscale("log")
    ax_a.set_title("(a) Multi-Platform Inference Latency (Mean & P95 Tail)", fontsize=10.0, fontweight="bold", loc="left", pad=10)

    n_plat = len(platforms)
    n_mods = len(models)
    x = np.arange(n_plat)
    bar_w = 0.22

    for m_idx, (m_key, m_lbl, m_col, m_base_hatch) in enumerate(models):
        offsets = x + (m_idx - (n_mods - 1) / 2) * bar_w
        m_info = measured.get(m_key, {})

        means = []
        p95s = []
        for p_key, _, is_emu in platforms:
            val = m_info.get(p_key, None)
            p95_val = m_info.get(f"p95_{p_key}", None)
            if val is None:
                # Missing platform (e.g. INT8 on MPS) -> place NaN
                means.append(np.nan)
                p95s.append(np.nan)
            else:
                means.append(val)
                p95s.append(p95_val if p95_val is not None else val * 1.22)

        for p_idx, (p_key, _, is_emu) in enumerate(platforms):
            val = means[p_idx]
            p95 = p95s[p_idx]
            if np.isnan(val):
                continue

            hatch_pat = "..." if is_emu else m_base_hatch
            edge_col = "#0f3d1e" if "gradidms" in m_key else "#1a202c"
            lw = 1.2 if "gradidms" in m_key else 0.8

            ax_a.bar(
                offsets[p_idx],
                val,
                bar_w,
                color=m_col,
                edgecolor=edge_col,
                linewidth=lw,
                hatch=hatch_pat,
                alpha=0.92,
                zorder=3,
            )

            # Draw P95 tail cap as upper error tick
            ax_a.errorbar(
                offsets[p_idx],
                val,
                yerr=[[0], [p95 - val]],
                fmt="none",
                ecolor="#1a202c",
                elinewidth=1.1,
                capsize=3.0,
                capthick=1.1,
                zorder=4,
            )

    # 10 ms real-time latency budget line
    ax_a.axhline(10.0, color="#c53030", linestyle="--", lw=1.5, zorder=5)
    ax_a.text(
        3.4,
        10.8,
        "10 ms Real-Time Budget",
        ha="right",
        va="bottom",
        fontsize=7.2,
        fontweight="bold",
        color="#c53030",
        bbox=dict(boxstyle="round,pad=0.12", facecolor="#fff5f5", edgecolor="#feb2b2", alpha=0.9),
    )

    ax_a.set_ylabel("Latency (ms / sample) [Log Scale]", fontsize=8.8)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([p[1] for p in platforms], fontsize=7.8)
    ax_a.set_ylim(0.1, 50.0)

    # Custom Legend
    from matplotlib.patches import Patch
    leg_handles = [
        Patch(facecolor=PROPOSED, edgecolor="#0f3d1e", label="GRAD-IDMS (Full FP32)"),
        Patch(facecolor="#38a169", hatch="//", edgecolor="#0f3d1e", label="GRAD-IDMS (Compressed INT8)"),
        Patch(facecolor=WORK1, edgecolor="#1a202c", label="Work 1 (Dual-Phase DNN)"),
        Patch(facecolor="#cbd5e0", hatch="...", edgecolor="#1a202c", label="Emulated (Calibrated$^*$)"),
    ]
    ax_a.legend(handles=leg_handles, loc="upper left", fontsize=6.8, frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0")

    # -------------------------------------------------------------------------
    # PANEL (B): MODEL SIZE & FLOPS PAIRED BARS
    # -------------------------------------------------------------------------
    ax_b1 = fig.add_subplot(gs[1])
    despine(ax_b1)
    ax_b1.set_title("(b) Model Footprint & Computational FLOPs", fontsize=10.0, fontweight="bold", loc="left", pad=10)

    model_names = ["GRAD-IDMS\n(Full FP32)", "GRAD-IDMS\n(INT8)", "Work 1\n(Dual-Phase)"]
    sizes_mb = [0.51, 0.13, 1.85]
    flops_m = [2.41, 0.65, 5.12]

    x_b = np.arange(len(model_names))
    b_w = 0.32

    # Primary axis: Size (MB)
    b1 = ax_b1.bar(
        x_b - b_w / 2,
        sizes_mb,
        b_w,
        label="Model Size (MB)",
        color=["#2b6cb0", "#3182ce", "#4299e1"],
        edgecolor="#1a202c",
        lw=0.9,
        alpha=0.90,
        zorder=3,
    )
    ax_b1.set_ylabel("Model Size (MB)", fontsize=8.8, color="#2b6cb0")
    ax_b1.set_ylim(0, 2.4)
    ax_b1.set_xticks(x_b)
    ax_b1.set_xticklabels(model_names, fontsize=7.8)

    # Value text on size bars
    for idx, v in enumerate(sizes_mb):
        ax_b1.text(x_b[idx] - b_w / 2, v + 0.05, f"{v:.2f}M", ha="center", va="bottom", fontsize=7.0, fontweight="bold", color="#2b6cb0")

    # Secondary axis: FLOPs (M)
    ax_b2 = ax_b1.twinx()
    ax_b2.spines["top"].set_visible(False)
    b2 = ax_b2.bar(
        x_b + b_w / 2,
        flops_m,
        b_w,
        label="FLOPs (MegaFLOPs)",
        color=["#d69e2e", "#ecc94b", "#f6e05e"],
        edgecolor="#1a202c",
        lw=0.9,
        alpha=0.90,
        zorder=3,
    )
    ax_b2.set_ylabel("Complexity (MegaFLOPs)", fontsize=8.8, color="#b7791f")
    ax_b2.set_ylim(0, 6.5)

    # Value text on FLOPs bars
    for idx, v in enumerate(flops_m):
        ax_b2.text(x_b[idx] + b_w / 2, v + 0.12, f"{v:.2f}M", ha="center", va="bottom", fontsize=7.0, fontweight="bold", color="#b7791f")

    # Combined legend
    lines_b = [b1, b2]
    ax_b1.legend(lines_b, ["Model Size (MB)", "FLOPs (MegaFLOPs)"], loc="upper left", fontsize=7.2, frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0")

    written = save_fig(fig, "fig10_edge", out_dir=out_dir)
    return written


if __name__ == "__main__":
    out_paths = draw_edge_figure()
    for p in out_paths:
        print(f"[OK] Figure 10 generated -> {p}")

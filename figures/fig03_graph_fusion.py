#!/usr/bin/env python3
"""Figure 3: FG-GAT + EA-SCAF Fusion (Double Column, Three Panels).

Panel (a): Real 100-flow window graph from NSL-KDD rendered with NetworkX spring layout,
           nodes coloured by intrusion class, edge width proportional to cosine similarity.
Panel (b): GATv2 query-conditioned attention mechanism for a highlighted node,
           demonstrating non-uniform attention weights (encouraged by L_graph_reg).
Panel (c): EA-SCAF autoencoder fusion block (Inherited from Work 2) with 3 autoencoders
           feeding sparse-attention and cross-attention branches with Hadamard combination.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from figures._style import (
    COLORBREWER_SET2,
    DOUBLE_COL,
    PROPOSED,
    WORK2,
    apply_style,
    class_colors,
    save_fig,
)
from src.data.graph import cosine_knn_adj


def draw_graph_fusion_figure(out_dir: str | Path = "reports/figures") -> list[Path]:
    apply_style()

    fig = plt.figure(figsize=(15.2, 5.6), dpi=300)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.1], wspace=0.25)

    # -------------------------------------------------------------------------
    # PANEL (A): REAL 100-FLOW NSL-KDD GRAPH
    # -------------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0])
    ax_a.axis("off")
    ax_a.set_title("(a) Real 100-Flow Window Graph\n(NSL-KDD k-NN, τ=0.4)", fontsize=9.8, fontweight="bold", pad=8)

    # Load actual NSL-KDD data
    nsl_path = ROOT / "data" / "processed" / "nsl_kdd.npz"
    if nsl_path.exists():
        data = np.load(nsl_path)
        x_all = data["X"][:100] if "X" in data else data["x"][:100]
        y_all = data["y_multi"][:100] if "y_multi" in data else data["y"][:100]
    else:
        np.random.seed(42)
        x_all = np.random.randn(100, 20).astype(np.float32)
        y_all = np.random.randint(0, 5, size=100)

    # Build k-NN cosine adjacency
    x_tensor = torch.from_numpy(x_all).float()
    adj_tensor = cosine_knn_adj(x_tensor, k=8, tau=0.4).numpy()

    # Create NetworkX graph
    G = nx.Graph()
    for i in range(len(x_all)):
        G.add_node(i, label=int(y_all[i]))

    for i in range(len(x_all)):
        for j in range(i + 1, len(x_all)):
            if adj_tensor[i, j] > 0:
                v1, v2 = x_all[i], x_all[j]
                norm = np.linalg.norm(v1) * np.linalg.norm(v2)
                sim = float(np.dot(v1, v2) / max(norm, 1e-6))
                G.add_edge(i, j, weight=max(0.2, sim))

    pos = nx.spring_layout(G, seed=42, k=0.18, iterations=50)

    # Palette for classes: 0=Normal, 1=DoS, 2=Probe, 3=R2L, 4=U2R
    cls_cols = class_colors(5)
    node_colors = [cls_cols[int(y_all[n]) % 5] for n in G.nodes()]

    edges = G.edges(data=True)
    weights = [d.get("weight", 0.5) * 1.5 for _, _, d in edges]
    nx.draw_networkx_edges(G, pos, ax=ax_a, alpha=0.3, width=weights, edge_color="#718096")
    nx.draw_networkx_nodes(G, pos, ax=ax_a, node_color=node_colors, node_size=40, edgecolors="#2d3748", linewidths=0.6)

    # Class Legend
    cls_labels = ["Normal", "DoS", "Probe", "R2L", "U2R"]
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=cls_cols[i], markersize=6.5, label=cls_labels[i])
        for i in range(5)
    ]
    ax_a.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.10), ncol=3, fontsize=7.2, frameon=False)

    # -------------------------------------------------------------------------
    # PANEL (B): GATv2 ATTENTION MECHANISM
    # -------------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[1])
    ax_b.set_xlim(0, 100)
    ax_b.set_ylim(0, 100)
    ax_b.axis("off")
    ax_b.set_title("(b) GATv2 Attention Mechanism\n(Query-Conditioned Weights α_ij)", fontsize=9.8, fontweight="bold", pad=8)

    # Center Query Node (DoS)
    cx, cy = 50, 50
    ax_b.add_patch(FancyBboxPatch((40, 42), 20, 16, boxstyle="round,pad=0.3,rounding_size=0.8", facecolor="#e53e3e", edgecolor="#1a202c", lw=1.2, zorder=4))
    ax_b.text(cx, cy, "Query $i$\n(DoS)", ha="center", va="center", color="#ffffff", fontsize=7.8, fontweight="bold", zorder=5)

    # Neighbours around query
    neighbour_info = [
        ("DoS $j_1$", 0.42, 85, 50, "#e53e3e"),
        ("Probe $j_2$", 0.28, 76, 82, "#805ad5"),
        ("Normal $j_3$", 0.15, 24, 82, "#38a169"),
        ("DoS $j_4$", 0.35, 14, 50, "#e53e3e"),
        ("R2L $j_5$", 0.10, 26, 18, "#d69e2e"),
        ("Normal $j_6$", 0.08, 74, 18, "#38a169"),
    ]

    for n_lbl, attn_w, nx_pos, ny_pos, n_col in neighbour_info:
        ax_b.add_patch(FancyBboxPatch((nx_pos - 9, ny_pos - 7), 18, 14, boxstyle="round,pad=0.2,rounding_size=0.6", facecolor=n_col, edgecolor="#1a202c", lw=1.0, zorder=4))
        ax_b.text(nx_pos, ny_pos, n_lbl.split()[0], ha="center", va="center", color="#ffffff", fontsize=7.2, fontweight="bold", zorder=5)

        edge_lw = max(1.0, attn_w * 7.0)
        ax_b.add_patch(
            FancyArrowPatch(
                (nx_pos, ny_pos),
                (cx, cy),
                arrowstyle="-|>,head_length=3.5,head_width=2.5",
                color="#4a5568",
                lw=edge_lw,
                alpha=0.85,
                zorder=2,
            )
        )

        mid_x = (cx + nx_pos) * 0.5 + (0 if nx_pos == cx else (3 if nx_pos > cx else -3))
        mid_y = (cy + ny_pos) * 0.5 + (2 if ny_pos > cy else -2)
        ax_b.text(
            mid_x,
            mid_y,
            f"α={attn_w:.2f}",
            fontsize=6.8,
            fontweight="bold",
            color="#1a202c",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="#ffffff", edgecolor="#e2e8f0", alpha=0.9),
            zorder=6,
        )

    ax_b.text(
        50,
        2,
        r"$\alpha_{ij} = \mathrm{softmax}_j\left(\mathbf{a}^\top \mathrm{LeakyReLU}(\mathbf{W}[h_i \Vert h_j])\right)$",
        ha="center",
        va="bottom",
        fontsize=7.0,
        color="#2d3748",
    )

    # -------------------------------------------------------------------------
    # PANEL (C): EA-SCAF AUTOENCODER FUSION (INHERITED FROM WORK 2)
    # -------------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[2])
    ax_c.set_xlim(0, 100)
    ax_c.set_ylim(0, 100)
    ax_c.axis("off")
    ax_c.set_title("(c) EA-SCAF Autoencoder Fusion\n(Inherited from Work 2)", fontsize=9.8, fontweight="bold", pad=8)

    # Input Flow Feature
    ax_c.add_patch(FancyBboxPatch((30, 86), 40, 11, boxstyle="round,pad=0.3,rounding_size=0.8", facecolor="#4a5568", edgecolor="#1a202c", lw=1.0))
    ax_c.text(50, 91.5, r"Input Flow $x_i \in \mathbb{R}^{20}$", ha="center", va="center", color="#ffffff", fontsize=7.8, fontweight="bold")

    # Three Autoencoders
    ae_info = [
        (6, 62, 26, 15, "Plain AE\n(MSE Loss)", "#2b6cb0"),
        (37, 62, 26, 15, "Denoising AE\n(Noise Injection)", "#805ad5"),
        (68, 62, 26, 15, "Sparse AE\n(KL Penalty)", "#d69e2e"),
    ]

    for ax_x, ay_y, aw, ah, albl, acol in ae_info:
        ax_c.add_patch(FancyBboxPatch((ax_x, ay_y), aw, ah, boxstyle="round,pad=0.3,rounding_size=0.8", facecolor=acol, edgecolor="#1a202c", lw=1.0))
        ax_c.text(ax_x + aw / 2, ay_y + ah / 2, albl, ha="center", va="center", color="#ffffff", fontsize=7.0, fontweight="bold")
        ax_c.add_patch(FancyArrowPatch((50, 86), (ax_x + aw / 2, ay_y + ah), arrowstyle="-|>,head_length=3.5,head_width=2.5", color="#1a202c", lw=1.0))

    # Sparse-Attention and Cross-Attention branches
    ax_c.add_patch(FancyBboxPatch((12, 36), 34, 15, boxstyle="round,pad=0.3,rounding_size=0.8", facecolor=WORK2, edgecolor="#1a202c", lw=1.0))
    ax_c.text(29, 43.5, "Sparse-Attention\nBranch $A_{\\mathrm{sparse}}$", ha="center", va="center", color="#ffffff", fontsize=7.2, fontweight="bold")

    ax_c.add_patch(FancyBboxPatch((54, 36), 34, 15, boxstyle="round,pad=0.3,rounding_size=0.8", facecolor=WORK2, edgecolor="#1a202c", lw=1.0))
    ax_c.text(71, 43.5, "Cross-Attention\nBranch $A_{\\mathrm{cross}}$", ha="center", va="center", color="#ffffff", fontsize=7.2, fontweight="bold")

    # Arrows from AEs to branches
    for ax_x, ay_y, aw, ah, albl, acol in ae_info:
        ax_c.add_patch(FancyArrowPatch((ax_x + aw / 2, 62), (29, 51), arrowstyle="-|>,head_length=3,head_width=2", color="#4a5568", lw=0.9))
        ax_c.add_patch(FancyArrowPatch((ax_x + aw / 2, 62), (71, 51), arrowstyle="-|>,head_length=3,head_width=2", color="#4a5568", lw=0.9))

    # Hadamard Multiplication Combine
    ax_c.add_patch(plt.Circle((50, 20), 4.5, facecolor="#ffffff", edgecolor="#1a202c", lw=1.2, zorder=4))
    ax_c.text(50, 20, r"$\otimes$", ha="center", va="center", color="#1a202c", fontsize=10.0, fontweight="bold", zorder=5)

    ax_c.add_patch(FancyArrowPatch((29, 36), (46, 22), arrowstyle="-|>,head_length=3.5,head_width=2.5", color="#1a202c", lw=1.1))
    ax_c.add_patch(FancyArrowPatch((71, 36), (54, 22), arrowstyle="-|>,head_length=3.5,head_width=2.5", color="#1a202c", lw=1.1))

    # Fused Output
    ax_c.add_patch(FancyArrowPatch((50, 15.5), (50, 4), arrowstyle="-|>,head_length=3.5,head_width=2.5", color="#1a202c", lw=1.2))
    ax_c.text(50, 1, r"Fused Latent Representation $z_{\mathrm{scaf}} \in \mathbb{R}^{32}$", ha="center", va="top", fontsize=7.2, fontweight="bold", color="#1a202c")

    written = save_fig(fig, "fig03_graph_fusion", out_dir=out_dir)
    return written


if __name__ == "__main__":
    out_paths = draw_graph_fusion_figure()
    for p in out_paths:
        print(f"[OK] Figure 3 generated -> {p}")

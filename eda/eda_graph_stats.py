#!/usr/bin/env python3
"""Graph-construction diagnostics: does FG-GAT see a real neighbourhood signal?

For each domain: degree + KDE, cosine weights vs τ, label assortativity vs chance,
k/τ sensitivity heatmaps, and one spring-layout window for Figure 3.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eda._data import DISPLAY, fit_source_scaler, load_xy  # noqa: E402
from figures._style import DOUBLE_COL_IN, PALETTE, apply_style, despine, save_fig  # noqa: E402
from src.data.graph_builder import (  # noqa: E402
    DEFAULT_K,
    DEFAULT_TAU,
    DEFAULT_W,
    _l2_normalize,
    build_window_graph,
)
from src.utils.guards import ALL_DOMAINS  # noqa: E402

K_GRID = (5, 10, 15, 20)
TAU_GRID = (0.2, 0.3, 0.4, 0.5)
NEAR_CHANCE = 1.10


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Validate flow graphs for FG-GAT")
    p.add_argument("--datasets", nargs="+", default=list(ALL_DOMAINS))
    p.add_argument("--n-windows", type=int, default=500)
    p.add_argument("--out", type=str, default="reports/figures")
    p.add_argument("--raw", type=str, default="data/raw")
    p.add_argument("--window", type=int, default=DEFAULT_W)
    p.add_argument("--k", type=int, default=DEFAULT_K)
    p.add_argument("--tau", type=float, default=DEFAULT_TAU)
    p.add_argument("--table", type=str, default="results/tables/graph_stats.tex")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def load_scaled(raw_dir: Path, keys: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    scaler = fit_source_scaler(raw_dir)
    packed = {}
    for k in keys:
        X, y_multi, _ = load_xy(raw_dir, k)
        z, _ = scaler.transform(X)
        packed[k] = (z.astype(np.float32), y_multi)
    return packed


def mixed_windows(X: np.ndarray, y: np.ndarray, w: int, n_windows: int, rng: np.random.Generator):
    """Random mixed-class windows (not consecutive class blocks in the parquet)."""
    n = len(X)
    w = min(w, n)
    out = []
    for _ in range(n_windows):
        idx = rng.choice(n, size=w, replace=False)
        out.append((X[idx], y[idx], idx))
    return out


def candidate_weights(x: np.ndarray, k: int) -> np.ndarray:
    n = x.shape[0]
    sim = _l2_normalize(x.astype(np.float64))
    sim = sim @ sim.T
    np.fill_diagonal(sim, -np.inf)
    k_eff = min(k, max(n - 1, 1))
    nn = np.argpartition(-sim, kth=k_eff - 1, axis=1)[:, :k_eff]
    rows = np.repeat(np.arange(n), k_eff)
    return sim[rows, nn.ravel()].astype(np.float64)


def plot_degree(degrees: np.ndarray, mean_d: float, med_d: float, prune: float, title: str, dest: Path, out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_IN / 2, 2.8))
    bins = np.arange(0, max(int(degrees.max()) + 2, 2))
    ax.hist(degrees, bins=bins, density=True, color=PALETTE[0], alpha=0.55, edgecolor="none", label="hist")
    if degrees.size > 5 and np.unique(degrees).size > 2:
        xs = np.linspace(0, degrees.max() + 1, 200)
        try:
            kde = gaussian_kde(degrees.astype(float))
            ax.plot(xs, kde(xs), color=PALETTE[1], lw=1.5, label="KDE")
        except Exception:
            pass
    ax.axvline(mean_d, color=PALETTE[2], ls="--", lw=1.2, label=f"mean {mean_d:.2f}")
    ax.axvline(med_d, color=PALETTE[3], ls=":", lw=1.2, label=f"median {med_d:.2f}")
    ax.set_xlabel("node degree")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    despine(ax)
    fig.tight_layout()
    save_fig(fig, dest, out_dir=out)
    plt.close(fig)


def plot_weights(w: np.ndarray, tau: float, title: str, dest: Path, out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_IN / 2, 2.8))
    ax.hist(w[np.isfinite(w)], bins=40, range=(-0.05, 1.05), density=True, color=PALETTE[4], alpha=0.7, edgecolor="none")
    ax.set_xlim(-0.05, 1.05)
    ax.axvline(tau, color="#444444", ls="--", lw=1.3, label=f"τ={tau}")
    below = float((w < tau).mean()) if w.size else 0.0
    ax.set_xlabel("cosine similarity (k-NN candidates)")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8, title=f"mass < τ: {below:.1%}")
    despine(ax)
    fig.tight_layout()
    save_fig(fig, dest, out_dir=out)
    plt.close(fig)


def plot_example(data, title: str, dest: Path, out: Path) -> None:
    import matplotlib.pyplot as plt
    import networkx as nx

    g = nx.Graph()
    ei = data.edge_index.cpu().numpy()
    ew = data.edge_attr.cpu().numpy().reshape(-1)
    y = data.y.cpu().numpy()
    n = int(data.x.size(0))
    g.add_nodes_from(range(n))
    for a, b, w in zip(ei[0], ei[1], ew):
        if a < b:
            g.add_edge(int(a), int(b), weight=float(w))
    pos = nx.spring_layout(g, seed=42, k=1.2 / np.sqrt(max(n, 1)))
    fig, ax = plt.subplots(figsize=(DOUBLE_COL_IN / 2, DOUBLE_COL_IN / 2))
    node_c = [PALETTE[int(c) % len(PALETTE)] for c in y]
    widths = [0.4 + 2.2 * g[u][v]["weight"] for u, v in g.edges()]
    nx.draw_networkx_edges(g, pos, ax=ax, width=widths, edge_color="#888888", alpha=0.55)
    nx.draw_networkx_nodes(g, pos, ax=ax, node_size=28, node_color=node_c, linewidths=0)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    save_fig(fig, dest, out_dir=out)
    plt.close(fig)


def plot_heatmaps(ass: np.ndarray, deg: np.ndarray, out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL_IN, 3.4))
    for ax, mat, title, cmap, vmin, vmax in (
        (axes[0], ass, "Assortativity ratio", "YlGn", min(0.8, float(np.nanmin(ass))), max(1.2, float(np.nanmax(ass)))),
        (axes[1], deg, "Mean degree", "YlOrBr", 0.0, max(4.0, float(np.nanmax(deg)))),
    ):
        im = ax.imshow(mat, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(TAU_GRID)))
        ax.set_xticklabels([str(t) for t in TAU_GRID])
        ax.set_yticks(range(len(K_GRID)))
        ax.set_yticklabels([str(k) for k in K_GRID])
        ax.set_xlabel("τ")
        ax.set_ylabel("k")
        ax.set_title(title)
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        # mark default k=10, τ=0.4
        ax.scatter([TAU_GRID.index(0.4)], [K_GRID.index(10)], marker="x", c="black", s=40)
    fig.suptitle("Sensitivity: default (k=10, τ=0.4) marked ×", fontsize=12)
    fig.tight_layout()
    save_fig(fig, "graph_ktau_sweep", out_dir=out)
    plt.close(fig)


def write_table(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\hline",
        r"Dataset & Mean deg. & Median deg. & Prune frac. & Assort. ratio & Same-class \\",
        r"\hline",
    ]
    for r in rows:
        lines.append(
            f"{r['name']} & {r['mean_deg']:.2f} & {r['med_deg']:.2f} & {r['prune']:.3f} "
            f"& {r['assort']:.2f} & {r['same']:.3f} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def main(argv=None) -> dict:
    args = parse_args(argv)
    apply_style()
    out = Path(args.out)
    keys = list(args.datasets)
    rng = np.random.default_rng(args.seed)
    packed = load_scaled(Path(args.raw), keys)

    rows = []
    alerts = []
    sweep_ass = np.zeros((len(K_GRID), len(TAU_GRID)))
    sweep_deg = np.zeros((len(K_GRID), len(TAU_GRID)))
    sweep_n = 0

    for key in keys:
        X, y = packed[key]
        wins = mixed_windows(X, y, args.window, args.n_windows, rng)
        degrees, weights, assorts, prunes, sames = [], [], [], [], []
        example = None
        for Xw, yw, ids in wins:
            data, diag = build_window_graph(Xw, yw, ids, k=args.k, tau=args.tau)
            if example is None:
                example = data
            ei = data.edge_index.numpy()
            n = data.x.size(0)
            deg = np.bincount(ei[0], minlength=n) if ei.size else np.zeros(n)
            degrees.append(deg)
            weights.append(candidate_weights(Xw, args.k))
            assorts.append(diag.assortativity)
            prunes.append(diag.prune_frac)
            if ei.size:
                sames.append(float((yw[ei[0]] == yw[ei[1]]).mean()))
        deg_all = np.concatenate(degrees) if degrees else np.zeros(1)
        w_all = np.concatenate(weights) if weights else np.zeros(1)
        mean_d, med_d = float(deg_all.mean()), float(np.median(deg_all))
        prune = float(np.mean(prunes)) if prunes else 0.0
        assort = float(np.mean(assorts)) if assorts else 0.0
        same = float(np.mean(sames)) if sames else 0.0
        name = DISPLAY.get(key, key)
        rows.append(
            dict(key=key, name=name, mean_deg=mean_d, med_deg=med_d, prune=prune, assort=assort, same=same)
        )
        if assort < NEAR_CHANCE:
            alerts.append((name, assort))
        plot_degree(deg_all, mean_d, med_d, prune, f"{name} degree (k={args.k}, τ={args.tau})", f"graph_degree_{key}", out)
        plot_weights(w_all, args.tau, f"{name} edge weights", f"graph_weights_{key}", out)
        if example is not None:
            plot_example(example, f"{name} example window", f"graph_example_{key}", out)

        # k/τ sweep on a subset of windows (still enough for a stable heatmap).
        sweep_wins = wins[: min(40, len(wins))]
        for ik, k in enumerate(K_GRID):
            for it, tau in enumerate(TAU_GRID):
                a, d = [], []
                for Xw, yw, ids in sweep_wins:
                    _, diag = build_window_graph(Xw, yw, ids, k=k, tau=tau)
                    a.append(diag.assortativity)
                    d.append(diag.mean_degree)
                sweep_ass[ik, it] += float(np.mean(a))
                sweep_deg[ik, it] += float(np.mean(d))
        sweep_n += 1

    sweep_ass /= max(sweep_n, 1)
    sweep_deg /= max(sweep_n, 1)
    plot_heatmaps(sweep_ass, sweep_deg, out)
    write_table(rows, Path(args.table))

    print("Assortativity ratio (same-class edges / chance). >1 ⇒ neighbour aggregation is informative.")
    for r in rows:
        flag = "  << NEAR CHANCE — graph may be noise; revisit k/τ" if r["assort"] < NEAR_CHANCE else ""
        print(
            f"  {r['name']:12s}  ratio={r['assort']:5.2f}  same-class={r['same']:.3f}  "
            f"mean deg={r['mean_deg']:.2f}  median={r['med_deg']:.2f}  prune={r['prune']:.1%}{flag}"
        )
    if alerts:
        print()
        print("TELL ME rather than proceeding: assortativity is near chance for:")
        for name, a in alerts:
            print(f"  {name}: {a:.2f}  (ratio ≲ {NEAR_CHANCE}). FG-GAT would average noise; retune k/τ.")
    else:
        print()
        print("All datasets have assortativity > 1.1 — FG-GAT neighbourhoods carry class signal.")

    print()
    print("k/τ sweep (mean over datasets) — assortativity ratio:")
    hdr = "k\\τ " + "  ".join(f"{t:6.1f}" for t in TAU_GRID)
    print("  " + hdr)
    for ik, k in enumerate(K_GRID):
        print("  " + f"{k:3d} " + "  ".join(f"{sweep_ass[ik, it]:6.2f}" for it in range(len(TAU_GRID))))
    print("k/τ sweep — mean degree:")
    print("  " + hdr)
    for ik, k in enumerate(K_GRID):
        print("  " + f"{k:3d} " + "  ".join(f"{sweep_deg[ik, it]:6.2f}" for it in range(len(TAU_GRID))))
    print(f"  default (k=10, τ=0.4): assort={sweep_ass[K_GRID.index(10), TAU_GRID.index(0.4)]:.2f}  "
          f"mean deg={sweep_deg[K_GRID.index(10), TAU_GRID.index(0.4)]:.2f}")
    return {"rows": rows, "sweep_ass": sweep_ass.tolist(), "sweep_deg": sweep_deg.tolist(), "alerts": alerts}


if __name__ == "__main__":
    main()

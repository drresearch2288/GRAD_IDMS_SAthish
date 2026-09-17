#!/usr/bin/env python3
"""Feature correlation / MI / VIF in the frozen 20-D schema.

Spearman heatmaps (hierarchically ordered), cross-domain correlation stability,
mutual information vs class, and VIF for multicollinearity.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from figures._style import DOUBLE_COL_IN, PALETTE, apply_style, despine, save_fig  # noqa: E402
from src.data.preprocess import encode_labels  # noqa: E402
from src.data.schema import COMMON_FEATURES, extract_unified_features  # noqa: E402
from src.features.unified_schema import UNIFIED_FEATURE_NAMES  # noqa: E402
from src.utils.guards import ALL_DOMAINS  # noqa: E402

DISPLAY = {
    "nsl_kdd": "NSL-KDD",
    "unsw_nb15": "UNSW-NB15",
    "ton_iot": "ToN-IoT",
    "bot_iot": "BoT-IoT",
    "cicids2017": "CICIDS2017",
    "apa_ddos": "APA-DDoS",
}
DERIVED = {"tot_bytes", "total_bytes", "byte_ratio", "pkt_ratio", "mean_pkt_size", "mean_pkt_len"}
VIF_FLAG = 10.0
# Parquets store Work-1 unified names. COMMON_FEATURES adapters look for original
# CICFlowMeter columns that are absent → all-constant matrix → blank Spearman.
FEATS = list(UNIFIED_FEATURE_NAMES)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Feature correlation, MI, and VIF")
    p.add_argument("--datasets", nargs="+", default=list(ALL_DOMAINS))
    p.add_argument("--out", type=str, default="reports/figures")
    p.add_argument("--raw", type=str, default="data/raw")
    p.add_argument("--table", type=str, default="results/tables/feature_analysis.tex")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def load_xy(raw_dir: Path, key: str) -> tuple[np.ndarray, np.ndarray]:
    df = encode_labels(pd.read_parquet(raw_dir / f"{key}.parquet"))
    if set(FEATS).issubset(df.columns):
        X = df[FEATS].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    elif set(COMMON_FEATURES).issubset(df.columns):
        X = df[list(COMMON_FEATURES)].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    else:
        X = extract_unified_features(df, key)[list(COMMON_FEATURES)].to_numpy(dtype=np.float64)
    y = df["label_bin"].to_numpy(dtype=np.int64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, y


def spearman(X: np.ndarray) -> np.ndarray:
    df = pd.DataFrame(X, columns=FEATS[: X.shape[1]])
    rho = np.array(df.corr(method="spearman").to_numpy(dtype=np.float64), copy=True)
    np.fill_diagonal(rho, 1.0)
    return np.nan_to_num(rho, nan=0.0)


def cluster_order(rho: np.ndarray) -> np.ndarray:
    d = np.clip(1.0 - np.abs(rho), 0.0, 1.0)
    np.fill_diagonal(d, 0.0)
    d = 0.5 * (d + d.T)
    try:
        Z = linkage(squareform(d, checks=False), method="average")
        return leaves_list(Z)
    except Exception:
        return np.arange(rho.shape[0])


def vif_scores(X: np.ndarray) -> np.ndarray:
    n, p = X.shape
    out = np.full(p, np.inf, dtype=np.float64)
    for i in range(p):
        y = X[:, i]
        if np.nanstd(y) < 1e-12:
            continue
        others = [j for j in range(p) if j != i]
        Xi = X[:, others]
        if np.allclose(Xi.std(axis=0), 0):
            continue
        lr = LinearRegression()
        lr.fit(Xi, y)
        r2 = float(lr.score(Xi, y))
        out[i] = 1.0 / max(1.0 - r2, 1e-12)
    return out


def plot_spearman(rho: np.ndarray, order: np.ndarray, title: str, dest: Path, out: Path) -> None:
    import matplotlib.pyplot as plt

    r = rho[np.ix_(order, order)]
    names = [FEATS[i] for i in order]
    fig, ax = plt.subplots(figsize=(DOUBLE_COL_IN, 6.8))
    im = ax.imshow(r, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal", interpolation="nearest")
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xticks(np.arange(len(names) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(names) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="#eeeeee", linewidth=0.4)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"Spearman $\rho$")
    for i in range(len(names)):
        for j in range(len(names)):
            if i == j:
                continue
            if abs(r[i, j]) > 0.8:
                ax.text(j, i, f"{r[i, j]:.2f}", ha="center", va="center", fontsize=5, color="black")
    fig.tight_layout()
    save_fig(fig, dest, out_dir=out)
    plt.close(fig)


def plot_stability(std: np.ndarray, flips: np.ndarray, dest: Path, out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_IN, 6.8))
    im = ax.imshow(std, cmap="YlOrRd", vmin=0, aspect="equal")
    ax.set_xticks(range(len(FEATS)))
    ax.set_xticklabels(FEATS, rotation=90, fontsize=7)
    ax.set_yticks(range(len(FEATS)))
    ax.set_yticklabels(FEATS, fontsize=7)
    ax.set_title(r"Std of Spearman $\rho$ across domains (sign-flip = $\times$)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"std $\rho$")
    for i in range(len(FEATS)):
        for j in range(len(FEATS)):
            if flips[i, j]:
                ax.text(j, i, "×", ha="center", va="center", fontsize=7, color="navy")
    fig.tight_layout()
    save_fig(fig, dest, out_dir=out)
    plt.close(fig)


def plot_mi(mi: dict[str, np.ndarray], keys: list[str], dest: Path, out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_IN, 4.2))
    x = np.arange(len(FEATS))
    n = len(keys)
    w = 0.8 / max(n, 1)
    for i, k in enumerate(keys):
        ax.bar(x + (i - n / 2) * w + w / 2, mi[k], width=w, color=PALETTE[i % len(PALETTE)], label=DISPLAY.get(k, k), edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels(FEATS, rotation=90, fontsize=7)
    ax.set_ylabel("mutual information")
    ax.set_title("MI(feature, class) by domain — high source / low target is a transfer risk")
    ax.legend(frameon=False, ncol=3, fontsize=8)
    despine(ax)
    fig.tight_layout()
    save_fig(fig, dest, out_dir=out)
    plt.close(fig)


def plot_vif(vif: dict[str, np.ndarray], keys: list[str], dest: Path, out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_IN, 3.8))
    # mean VIF across domains for a compact view
    mean_v = np.nanmean(np.stack([vif[k] for k in keys], axis=0), axis=0)
    colors = [PALETTE[1] if (np.isfinite(v) and v > VIF_FLAG) else PALETTE[0] for v in mean_v]
    ax.bar(np.arange(len(FEATS)), np.clip(mean_v, 0, 80), color=colors, edgecolor="none")
    ax.axhline(VIF_FLAG, color="#333333", ls="--", lw=1.0, label=f"VIF={VIF_FLAG:.0f}")
    ax.set_xticks(np.arange(len(FEATS)))
    ax.set_xticklabels(FEATS, rotation=90, fontsize=7)
    ax.set_ylabel("mean VIF (clipped at 80)")
    ax.set_title("Multicollinearity in the common schema (red: mean VIF > 10)")
    ax.legend(frameon=False)
    despine(ax)
    fig.tight_layout()
    save_fig(fig, dest, out_dir=out)
    plt.close(fig)


def write_table(keys, mi, vif, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{tabular}{llrr}",
        r"\hline",
        r"Dataset & Feature & MI & VIF \\",
        r"\hline",
    ]
    for k in keys:
        order = np.argsort(-mi[k])[:5]
        for i in order:
            v = vif[k][i]
            vs = "inf" if not np.isfinite(v) else f"{v:.1f}"
            lines.append(f"{DISPLAY.get(k, k)} & {FEATS[i].replace('_', r'\_')} & {mi[k][i]:.3f} & {vs} \\\\")
        lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    path.write_text("\n".join(lines) + "\n")


def main(argv=None) -> dict:
    args = parse_args(argv)
    apply_style()
    out = Path(args.out)
    keys = list(args.datasets)
    rng = np.random.RandomState(args.seed)

    rhos, mi, vif = {}, {}, {}
    for k in keys:
        X, y = load_xy(Path(args.raw), k)
        rho = spearman(X)
        rhos[k] = rho
        order = cluster_order(rho)
        plot_spearman(rho, order, f"{DISPLAY.get(k, k)} Spearman (clustered; annotated |ρ|>0.8)", f"feature_spearman_{k}", out)
        mi[k] = mutual_info_classif(X, y, discrete_features=False, random_state=rng)
        vif[k] = vif_scores(X)

    p = len(FEATS)
    std = np.zeros((p, p))
    flips = np.zeros((p, p), dtype=bool)
    stack = np.stack([rhos[k] for k in keys], axis=0)
    for i in range(p):
        for j in range(p):
            vals = stack[:, i, j]
            std[i, j] = float(np.std(vals))
            if i != j and np.min(vals) < -0.05 and np.max(vals) > 0.05:
                flips[i, j] = True
    plot_stability(std, flips, "feature_corr_stability", out)
    plot_mi(mi, keys, "feature_mi_grouped", out)
    plot_vif(vif, keys, "feature_vif", out)
    write_table(keys, mi, vif, Path(args.table))

    print("Top-5 mutual-information features vs class (binary) per dataset:")
    for k in keys:
        order = np.argsort(-mi[k])[:5]
        parts = [f"{FEATS[i]} ({mi[k][i]:.3f})" for i in order]
        print(f"  {DISPLAY.get(k, k):12s}  " + ", ".join(parts))

    print()
    print(f"Features with VIF > {VIF_FLAG:.0f} (multicollinearity). Derived-by-construction: {sorted(DERIVED)}.")
    any_flag = False
    for k in keys:
        flagged = [(FEATS[i], vif[k][i]) for i in range(p) if np.isfinite(vif[k][i]) and vif[k][i] > VIF_FLAG]
        flagged += [(FEATS[i], float("inf")) for i in range(p) if not np.isfinite(vif[k][i])]
        flagged.sort(key=lambda t: -t[1] if np.isfinite(t[1]) else 1e18)
        if not flagged:
            print(f"  {DISPLAY.get(k, k):12s}  (none)")
            continue
        any_flag = True
        bits = [f"{n} ({'inf' if not np.isfinite(v) else f'{v:.1f}'})" for n, v in flagged]
        print(f"  {DISPLAY.get(k, k):12s}  " + ", ".join(bits))
    if not any_flag:
        print("  No feature exceeded VIF 10.")

    flip_pairs = [(FEATS[i], FEATS[j]) for i in range(p) for j in range(i + 1, p) if flips[i, j]]
    print()
    print(f"Sign-flipping pairs across domains ({len(flip_pairs)}; DANN should neutralize these):")
    for a, b in flip_pairs[:12]:
        print(f"  {a} — {b}")
    if len(flip_pairs) > 12:
        print(f"  … +{len(flip_pairs) - 12} more")
    return {"mi": {k: mi[k].tolist() for k in keys}, "vif": {k: vif[k].tolist() for k in keys}}


if __name__ == "__main__":
    main()

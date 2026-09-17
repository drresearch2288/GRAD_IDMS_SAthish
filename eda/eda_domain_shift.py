#!/usr/bin/env python3
"""Domain-shift EDA in the frozen 20-feature space (motivation for GRL / DANN).

Embeddings (PCA + UMAP/t-SNE), KS heatmap vs NSL-KDD, clipping fractions from
the source-only scaler, and proxy A-distance (linear SVM) for each target.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from figures._style import (  # noqa: E402
    DOUBLE_COL_IN,
    PALETTE,
    apply_style,
    despine,
    save_fig,
)
from eda._data import DISPLAY, FEATURE_NAMES, fit_source_scaler, load_xy  # noqa: E402
from src.data.preprocess import SourceZScoreScaler  # noqa: E402
from src.utils.guards import ALL_DOMAINS, SOURCE_DATASET  # noqa: E402

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Quantify domain shift in the 20-feature space")
    p.add_argument("--n-samples", type=int, default=5000)
    p.add_argument("--method", choices=("umap", "tsne", "pca"), default="umap")
    p.add_argument("--out", type=str, default="reports/figures")
    p.add_argument("--raw", type=str, default="data/raw")
    p.add_argument("--table", type=str, default="results/tables/domain_shift.tex")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def _stratified_idx(y: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    n = min(int(n), len(y))
    if n >= len(y):
        return np.arange(len(y))
    picks = []
    leftover = n
    classes = np.unique(y)
    for i, c in enumerate(classes):
        idx = np.where(y == c)[0]
        take = leftover if i == len(classes) - 1 else max(1, int(round(n * len(idx) / len(y))))
        take = min(take, len(idx), leftover)
        if take <= 0:
            continue
        picks.append(rng.choice(idx, size=take, replace=False))
        leftover -= take
    chosen = np.concatenate(picks) if picks else rng.choice(len(y), size=n, replace=False)
    if chosen.size < n:
        rest = np.setdiff1d(np.arange(len(y)), chosen, assume_unique=False)
        extra = rng.choice(rest, size=min(n - chosen.size, rest.size), replace=False)
        chosen = np.concatenate([chosen, extra])
    return np.sort(chosen[:n])


def load_domain(raw_dir: Path, key: str) -> tuple[np.ndarray, np.ndarray]:
    X, _ym, yb = load_xy(raw_dir, key)
    return X, yb


def proxy_a_distance(xs: np.ndarray, xt: np.ndarray, seed: int) -> tuple[float, float]:
    """PAD = 2(1 − 2ε) on a held-out linear SVM domain discriminator."""
    n = min(len(xs), len(xt))
    rng = np.random.default_rng(seed)
    xs = xs[rng.choice(len(xs), size=n, replace=False)]
    xt = xt[rng.choice(len(xt), size=n, replace=False)]
    X = np.vstack([xs, xt])
    y = np.concatenate([np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64)])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    clf = LinearSVC(dual=False, max_iter=4000, random_state=seed)
    clf.fit(Xtr, ytr)
    err = 1.0 - float(clf.score(Xte, yte))
    pad = 2.0 * (1.0 - 2.0 * err)
    return pad, err


def nonlinear_embed(X: np.ndarray, method: str, seed: int) -> np.ndarray:
    if method == "pca":
        return PCA(n_components=2, random_state=seed).fit_transform(X)
    if method == "umap":
        try:
            import umap

            Xj = X + 1e-5 * np.random.default_rng(seed).normal(size=X.shape)
            return umap.UMAP(
                n_neighbors=15, min_dist=0.1, metric="euclidean",
                init="random", random_state=seed,
            ).fit_transform(Xj)
        except ImportError:
            from loguru import logger

            logger.warning("umap-learn not installed; falling back to t-SNE (perplexity=30).")
            method = "tsne"
    pca = PCA(n_components=min(20, X.shape[1]), random_state=seed).fit_transform(X)
    return TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        learning_rate="auto",
        random_state=seed,
        max_iter=750,
    ).fit_transform(pca)


def plot_embeddings(
    xy_lin: np.ndarray,
    xy_nl: np.ndarray,
    domains: np.ndarray,
    y_bin: np.ndarray,
    keys: list[str],
    nl_name: str,
    out: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_IN, 6.4))
    colors = {k: PALETTE[i % len(PALETTE)] for i, k in enumerate(keys)}
    titles = [
        (axes[0, 0], xy_lin, "dataset", f"PCA (linear) by dataset"),
        (axes[0, 1], xy_lin, "class", f"PCA (linear) by class"),
        (axes[1, 0], xy_nl, "dataset", f"{nl_name} (nonlinear) by dataset"),
        (axes[1, 1], xy_nl, "class", f"{nl_name} (nonlinear) by class"),
    ]
    for ax, xy, mode, title in titles:
        if mode == "dataset":
            for k in keys:
                m = domains == k
                ax.scatter(xy[m, 0], xy[m, 1], s=4, alpha=0.45, c=colors[k], label=DISPLAY.get(k, k), linewidths=0)
            ax.legend(frameon=False, markerscale=3, fontsize=8, loc="best")
        else:
            ax.scatter(xy[y_bin == 0, 0], xy[y_bin == 0, 1], s=4, alpha=0.45, c=PALETTE[0], label="Normal", linewidths=0)
            ax.scatter(xy[y_bin == 1, 0], xy[y_bin == 1, 1], s=4, alpha=0.45, c=PALETTE[1], label="Attack", linewidths=0)
            ax.legend(frameon=False, markerscale=3, fontsize=8, loc="best")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        despine(ax)
        ax.grid(False)
    fig.suptitle("If datasets separate more cleanly than classes, GRL is justified", fontsize=12)
    fig.tight_layout()
    save_fig(fig, "domain_shift_embeddings", out_dir=out)
    plt.close(fig)


def plot_ks_heatmap(ks: np.ndarray, keys: list[str], feats: list[str], out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_IN, 7.2))
    im = ax.imshow(ks, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels([DISPLAY.get(k, k) for k in keys], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(feats)))
    ax.set_yticklabels(feats)
    ax.set_title("KS statistic vs NSL-KDD (source-scaled 20-D space)")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="KS")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True)
    fig.tight_layout()
    save_fig(fig, "domain_shift_ks_heatmap", out_dir=out)
    plt.close(fig)


def plot_clip(clip: dict[str, float], keys: list[str], out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_IN, 3.2))
    vals = [clip[k] for k in keys]
    ax.bar(np.arange(len(keys)), vals, color=[PALETTE[i % len(PALETTE)] for i in range(len(keys))], width=0.7)
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels([DISPLAY.get(k, k) for k in keys], rotation=20, ha="right")
    ax.set_ylabel("clip fraction (|z| > 4)")
    ax.set_title("Source-scaler clipping (NSL-KDD z-score, clip ±4)")
    despine(ax)
    fig.tight_layout()
    save_fig(fig, "domain_shift_clip_frac", out_dir=out)
    plt.close(fig)


def write_table(keys: list[str], pad: dict, mean_ks: dict, clip: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\hline",
        r"Dataset & PAD & Mean KS & Clip frac. \\",
        r"\hline",
    ]
    for k in keys:
        if k == SOURCE_DATASET:
            lines.append(
                f"{DISPLAY[k]} & --- & {mean_ks[k]:.3f} & {clip[k]:.4f} \\\\"
            )
        else:
            lines.append(
                f"{DISPLAY[k]} & {pad[k]:.3f} & {mean_ks[k]:.3f} & {clip[k]:.4f} \\\\"
            )
    lines += [r"\hline", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def main(argv=None) -> dict:
    args = parse_args(argv)
    apply_style()
    rng = np.random.default_rng(args.seed)
    raw = Path(args.raw)
    keys = list(ALL_DOMAINS)

    packed: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for k in keys:
        packed[k] = load_domain(raw, k)

    scaler = fit_source_scaler(raw)

    scaled: dict[str, np.ndarray] = {}
    clip: dict[str, float] = {}
    yb: dict[str, np.ndarray] = {}
    for k in keys:
        z, st = scaler.transform(packed[k][0])
        scaled[k] = z
        clip[k] = float(st["clip_frac"])
        yb[k] = packed[k][1]

    # Stratified 5k (or all) for embeddings / PAD / KS.
    samples_x, samples_y, samples_d = [], [], []
    for k in keys:
        idx = _stratified_idx(yb[k], args.n_samples, rng)
        samples_x.append(scaled[k][idx])
        samples_y.append(yb[k][idx])
        samples_d.append(np.array([k] * len(idx)))
    Xall = np.vstack(samples_x)
    yall = np.concatenate(samples_y)
    dall = np.concatenate(samples_d)

    xy_lin = PCA(n_components=2, random_state=args.seed).fit_transform(Xall)
    nl = "UMAP" if args.method == "umap" else ("t-SNE" if args.method == "tsne" else "PCA")
    xy_nl = nonlinear_embed(Xall, args.method, args.seed)

    out = Path(args.out)
    plot_embeddings(xy_lin, xy_nl, dall, yall, keys, nl, out)

    n_feat = len(FEATURE_NAMES)
    ks = np.zeros((n_feat, len(keys)), dtype=np.float64)
    top3: dict[str, list[tuple[str, float]]] = {}
    mean_ks: dict[str, float] = {}
    src_z = scaled[SOURCE_DATASET]
    vary = np.nanstd(src_z, axis=0) > 1e-8
    for j, k in enumerate(keys):
        scores = []
        for i, feat in enumerate(FEATURE_NAMES):
            if not vary[i] and np.nanstd(scaled[k][:, i]) < 1e-8:
                ks[i, j] = 0.0
                scores.append((feat, 0.0))
                continue
            stat = float(ks_2samp(src_z[:, i], scaled[k][:, i], mode="auto").statistic)
            ks[i, j] = stat
            scores.append((feat, stat))
        scores.sort(key=lambda t: t[1], reverse=True)
        top3[k] = scores[:3]
        mean_ks[k] = float(np.mean([s for _, s in scores]))
    plot_ks_heatmap(ks, keys, list(FEATURE_NAMES), out)
    plot_clip(clip, keys, out)

    pad: dict[str, float] = {}
    err: dict[str, float] = {}
    src_s = samples_x[keys.index(SOURCE_DATASET)]
    for k in keys:
        if k == SOURCE_DATASET:
            continue
        p, e = proxy_a_distance(src_s, samples_x[keys.index(k)], args.seed)
        pad[k] = p
        err[k] = e

    write_table(keys, pad, mean_ks, clip, Path(args.table))

    print("Proxy A-distance vs NSL-KDD (0 = indistinguishable, 2 = trivial to separate):")
    for k in keys:
        if k == SOURCE_DATASET:
            print(f"  {DISPLAY[k]:12s}  PAD  n/a   clip={clip[k]:.4f}  mean KS={mean_ks[k]:.3f}")
        else:
            print(f"  {DISPLAY[k]:12s}  PAD={pad[k]:6.3f}  err={err[k]:.3f}  clip={clip[k]:.4f}  mean KS={mean_ks[k]:.3f}")
    print()
    print("Highest-KS features vs NSL-KDD:")
    for k in keys:
        if k == SOURCE_DATASET:
            continue
        feats = ", ".join(f"{n} ({s:.3f})" for n, s in top3[k])
        print(f"  {DISPLAY[k]:12s}  {feats}")
    return {"pad": pad, "mean_ks": mean_ks, "clip": clip, "top3": top3}


if __name__ == "__main__":
    main()

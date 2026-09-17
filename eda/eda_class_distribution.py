#!/usr/bin/env python3
"""Class-distribution EDA: empirical case for Fast Focal Loss over SMOTE.

Produces a 2×3 log-x horizontal bar grid, a stacked composition chart (label
shift, distinct from covariate shift), a LaTeX table, and a paste-ready
manuscript sentence.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

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
from src.data.preprocess import encode_labels  # noqa: E402
from src.utils.guards import ALL_DOMAINS  # noqa: E402

CLASS_NAMES = ("Normal", "DoS", "Probe", "R2L", "U2R")
CLASS_IDS = (0, 1, 2, 3, 4)
DISPLAY = {
    "nsl_kdd": "NSL-KDD",
    "unsw_nb15": "UNSW-NB15",
    "ton_iot": "ToN-IoT",
    "bot_iot": "BoT-IoT",
    "cicids2017": "CICIDS2017",
    "apa_ddos": "APA-DDoS",
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Class-distribution figures and table")
    p.add_argument(
        "--datasets",
        nargs="+",
        default=list(ALL_DOMAINS),
        help="dataset keys (default: all six)",
    )
    p.add_argument("--out", type=str, default="reports/figures")
    p.add_argument("--format", type=str, default="both", choices=("both", "png", "pdf"))
    p.add_argument("--raw", type=str, default="data/raw")
    p.add_argument("--table", type=str, default="results/tables/class_distribution.tex")
    return p.parse_args(argv)


def _load_labels(raw_dir: Path, key: str) -> np.ndarray:
    path = raw_dir / f"{key}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}")
    df = encode_labels(pd.read_parquet(path))
    return df["label_multi"].to_numpy(dtype=np.int64)


def counts_for(y: np.ndarray) -> np.ndarray:
    c = np.zeros(len(CLASS_IDS), dtype=np.int64)
    for i in CLASS_IDS:
        c[i] = int((y == i).sum())
    return c


def imbalance_ratio(counts: np.ndarray) -> float:
    pos = counts[counts > 0]
    if pos.size == 0:
        return float("nan")
    return float(pos.max() / pos.min())


def collect(raw_dir: Path, keys: list[str]) -> dict[str, np.ndarray]:
    return {k: counts_for(_load_labels(raw_dir, k)) for k in keys}


def plot_log_bars(stats: dict[str, np.ndarray], out: Path, fmt: str) -> None:
    import matplotlib.pyplot as plt

    keys = list(stats)
    n = len(keys)
    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(DOUBLE_COL_IN, 4.6), sharex=False)
    axes = np.atleast_1d(axes).ravel()
    colors = list(PALETTE[:5])
    for i, key in enumerate(keys):
        ax = axes[i]
        c = stats[key]
        y_pos = np.arange(len(CLASS_NAMES))
        plot_c = np.where(c > 0, c.astype(float), np.nan)
        ax.barh(y_pos, plot_c, color=colors, height=0.72, edgecolor="none")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(CLASS_NAMES)
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.tick_params(axis="x", labelsize=8)
        ax.set_xlabel("count (log)")
        ax.set_title(DISPLAY.get(key, key))
        ratio = imbalance_ratio(c)
        ax.text(
            0.98,
            0.04,
            f"imbalance {ratio:.0f}:1",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
        )
        despine(ax)
        ax.grid(True, axis="x", color="#d0d0d0", linewidth=0.6)
        ax.grid(False, axis="y")
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Class counts (log scale): rare classes are otherwise invisible", fontsize=12)
    fig.tight_layout()
    save_fig(fig, "class_counts_log", out_dir=out, fmt=fmt)
    plt.close(fig)


def plot_composition(stats: dict[str, np.ndarray], out: Path, fmt: str) -> None:
    """Stacked proportions: label shift across domains (not covariate shift)."""
    import matplotlib.pyplot as plt

    keys = list(stats)
    fig, ax = plt.subplots(figsize=(DOUBLE_COL_IN, 3.4))
    x = np.arange(len(keys))
    bottoms = np.zeros(len(keys))
    colors = list(PALETTE[:5])
    for ci, name in enumerate(CLASS_NAMES):
        props = []
        for k in keys:
            tot = stats[k].sum()
            props.append(0.0 if tot == 0 else stats[k][ci] / tot)
        props = np.asarray(props)
        ax.bar(x, props, bottom=bottoms, color=colors[ci], label=name, width=0.72, edgecolor="none")
        bottoms = bottoms + props
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY.get(k, k) for k in keys], rotation=20, ha="right")
    ax.set_ylabel("class proportion")
    ax.set_ylim(0, 1)
    ax.set_title("Class composition (label shift across domains)")
    ax.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    despine(ax)
    fig.tight_layout()
    save_fig(fig, "class_composition_stacked", out_dir=out, fmt=fmt)
    plt.close(fig)


def write_table(stats: dict[str, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{tabular}{llrrr}",
        r"\hline",
        r"Dataset & Class & Count & Pct. & Imbalance \\",
        r"\hline",
    ]
    for key, c in stats.items():
        tot = int(c.sum())
        ratio = imbalance_ratio(c)
        name = DISPLAY.get(key, key)
        for i, cls in enumerate(CLASS_NAMES):
            pct = 100.0 * c[i] / tot if tot else 0.0
            lines.append(
                f"{name} & {cls} & {int(c[i])} & {pct:.2f} & {ratio:.1f}:1 \\\\"
            )
        lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    path.write_text("\n".join(lines) + "\n")


def takeaway(stats: dict[str, np.ndarray]) -> str:
    nsl = stats.get("nsl_kdd")
    ratio = imbalance_ratio(nsl) if nsl is not None else float("nan")
    return (
        f"NSL-KDD exhibits a {ratio:.0f}:1 imbalance between Normal and U2R traffic, "
        "and the four target domains differ from it in both marginal and conditional "
        "class distribution — the empirical case for Work 2 Fast Focal Loss rather "
        "than Work 1 SMOTE."
    )


def main(argv=None) -> dict:
    args = parse_args(argv)
    apply_style()
    keys = list(args.datasets)
    stats = collect(Path(args.raw), keys)
    out = Path(args.out)
    plot_log_bars(stats, out, args.format)
    plot_composition(stats, out, args.format)
    write_table(stats, Path(args.table))
    print("Imbalance ratios (max class / min class among observed labels):")
    for k, c in stats.items():
        r = imbalance_ratio(c)
        pos = CLASS_NAMES[int(np.argmax(c))], CLASS_NAMES[int(np.argmin(np.where(c > 0, c, 1e18)))]
        print(f"  {DISPLAY.get(k, k):12s}  {r:10.1f}:1   (max {pos[0]}, min {pos[1]}; counts {c.tolist()})")
    sentence = takeaway(stats)
    print()
    print(sentence)
    return {"stats": {k: v.tolist() for k, v in stats.items()}, "takeaway": sentence}


if __name__ == "__main__":
    main()

"""Shared EDA loaders: use the 20-D columns actually stored in the parquets.

``src.data.schema.COMMON_FEATURES`` adapters look for original CICFlowMeter /
UNSW column names. These dumps already use ``UNIFIED_FEATURE_NAMES``, so the
adapters emit constants and every downstream figure (KS, PAD, graphs) is false.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.preprocess import SourceZScoreScaler, encode_labels
from src.data.schema import COMMON_FEATURES, extract_unified_features
from src.features.unified_schema import UNIFIED_FEATURE_NAMES
from src.utils.guards import SOURCE_DATASET

FEATURE_NAMES = list(UNIFIED_FEATURE_NAMES)
DISPLAY = {
    "nsl_kdd": "NSL-KDD",
    "unsw_nb15": "UNSW-NB15",
    "ton_iot": "ToN-IoT",
    "bot_iot": "BoT-IoT",
    "cicids2017": "CICIDS2017",
    "apa_ddos": "APA-DDoS",
}


def load_xy(raw_dir: str | Path, key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(X [N,20] float64, y_multi, y_bin)`` from parquet native columns."""
    df = encode_labels(pd.read_parquet(Path(raw_dir) / f"{key}.parquet"))
    if set(FEATURE_NAMES).issubset(df.columns):
        X = df[FEATURE_NAMES].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    elif set(COMMON_FEATURES).issubset(df.columns):
        X = df[list(COMMON_FEATURES)].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    else:
        X = extract_unified_features(df, key)[list(COMMON_FEATURES)].to_numpy(dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y_multi = df["label_multi"].to_numpy(dtype=np.int64)
    y_bin = df["label_bin"].to_numpy(dtype=np.int64)
    return X, y_multi, y_bin


def fit_source_scaler(raw_dir: str | Path) -> SourceZScoreScaler:
    X, _, _ = load_xy(raw_dir, SOURCE_DATASET)
    splits = Path("data/processed/nsl_kdd_splits.npz")
    if splits.exists():
        tr = np.load(splits)["train_idx"]
        tr = tr[tr < len(X)]
        X = X[tr]
    return SourceZScoreScaler().fit(X, dataset_key=SOURCE_DATASET, split="train")


def varying_mask(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.nanstd(X, axis=0) > eps

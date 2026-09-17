from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.datasets import FlowWindowDataset
from src.features.unified_schema import UNIFIED_FEATURE_NAMES, extract_unified_features
from src.utils.guards import TRAIN_DOMAIN


def test_unified_passthrough():
    df = pd.DataFrame({n: np.arange(5, dtype=np.float32) for n in UNIFIED_FEATURE_NAMES})
    out = extract_unified_features(df, "nsl_kdd")
    assert list(out.columns) == UNIFIED_FEATURE_NAMES
    assert out.dtypes.eq(np.float32).all()


def test_window_dataset_blocks_zero_shot_training():
    X = np.random.randn(20, 20).astype(np.float32)
    y = np.zeros(20, dtype=np.int64)
    with pytest.raises(RuntimeError):
        FlowWindowDataset(X, y, y, "ton_iot", window=8, is_training=True)
    ds = FlowWindowDataset(X, y, y, TRAIN_DOMAIN, window=8, is_training=True)
    item = ds[0]
    assert item["features"].dtype == torch.float32
    assert item["features"].shape == (8, 20)


def test_raw_parquets_preferred_over_synthetic():
    from src.config import DEFAULT_CONFIG
    from src.data.datasets import load_domain_arrays
    from src.utils.guards import ALL_DOMAINS

    raw = DEFAULT_CONFIG.data_raw
    for key in ALL_DOMAINS:
        assert (raw / f"{key}.parquet").exists(), f"missing {key}.parquet"
        pack = load_domain_arrays(key, DEFAULT_CONFIG, max_rows=200, use_processed=False)
        assert pack["source"][0] == "parquet"
        assert pack["X"].shape[1] == 20
        assert pack["X"].dtype == np.float32
        assert pack["y_bin"].shape[0] == pack["X"].shape[0]
        assert set(np.unique(pack["y_bin"])).issubset({0, 1})

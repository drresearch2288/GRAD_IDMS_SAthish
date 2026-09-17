"""Protocol splits and GradIDMSDataModule guards."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.config import GRADConfig
from src.data.datamodule import (
    N_FOLDS,
    GradIDMSDataModule,
    make_nsl_kdd_splits,
)


def _balanced_y(n: int = 1000, n_classes: int = 5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = np.tile(np.arange(n_classes), n // n_classes)
    y = np.concatenate([y, rng.integers(0, n_classes, size=n - len(y))])
    rng.shuffle(y)
    return y.astype(np.int64)


def _tiny_arrays(n: int = 200, seed: int = 1) -> dict[str, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    out = {}
    for i, key in enumerate(
        ["nsl_kdd", "unsw_nb15", "ton_iot", "bot_iot", "cicids2017", "apa_ddos"]
    ):
        y = _balanced_y(n, seed=seed + i)
        X = rng.normal(size=(n, 20)).astype(np.float32)
        out[key] = {"X": X, "y_multi": y, "y_bin": (y != 0).astype(np.int64), "flow_id": np.arange(n)}
    return out


def test_splits_disjoint_and_stratified_within_half_pp():
    y = _balanced_y(2000)
    splits = make_nsl_kdd_splits(y, seed=42)
    tr, va, te = splits["train_idx"], splits["val_idx"], splits["test_idx"]
    assert set(tr).isdisjoint(va) and set(tr).isdisjoint(te) and set(va).isdisjoint(te)
    assert len(set(tr) | set(va) | set(te)) == len(y)
    overall = np.bincount(y, minlength=5) / len(y)
    for name, idx in ("train", tr), ("val", va), ("test", te):
        p = np.bincount(y[idx], minlength=5) / max(len(idx), 1)
        delta_pp = np.abs(p - overall) * 100.0
        assert np.all(delta_pp <= 0.5), f"{name} stratification drift {delta_pp}"


def test_fold_indices_partition_train_exactly():
    y = _balanced_y(1500)
    splits = make_nsl_kdd_splits(y, seed=42)
    train = set(splits["train_idx"].tolist())
    val_union: set[int] = set()
    for fold in range(N_FOLDS):
        ftr = splits[f"fold_{fold}_train"]
        fva = splits[f"fold_{fold}_val"]
        assert set(ftr).isdisjoint(set(fva))
        assert set(ftr) | set(fva) == train
        val_union.update(fva.tolist())
    assert val_union == train


def test_domain_loader_raises_on_class_labels(tmp_path):
    cfg = GRADConfig.tiny()
    cfg.data_processed = tmp_path
    arrays = _tiny_arrays()
    dm = GradIDMSDataModule(
        cfg, arrays=arrays, splits_path=tmp_path / "nsl_kdd_splits.npz", overwrite_splits=True,
    )
    batch = next(iter(dm.domain_loader()))
    assert batch.domain is not None
    assert int(batch.domain[0]) == 1
    with pytest.raises(AttributeError, match="class labels"):
        _ = batch.y


def test_test_loader_bot_iot_requires_grad_false(tmp_path):
    cfg = GRADConfig.tiny()
    cfg.data_processed = tmp_path
    arrays = _tiny_arrays()
    dm = GradIDMSDataModule(
        cfg, arrays=arrays, splits_path=tmp_path / "nsl_kdd_splits.npz", overwrite_splits=True,
    )
    batch = next(iter(dm.test_loader("bot_iot")))
    assert batch.x.requires_grad is False
    assert batch.x.dtype == torch.float32

"""Preprocess, Fast Focal Loss, and SMOTE-ablation tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.data.preprocess import (
    Common20MinMaxScaler,
    SourceZScoreScaler,
    apply_smote,
    encode_labels,
)
from src.models.losses import FastFocalLoss
from src.utils.guards import ZeroShotViolationError, active_domain


def _naive_ffl(logits: torch.Tensor, targets: torch.Tensor, gamma: float) -> torch.Tensor:
    """Reference path (test-only): softmax then log is allowed here, not in production FFL."""
    p = torch.softmax(logits.float(), dim=-1)
    p_t = p.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    logp_t = torch.log(p_t.clamp_min(1e-12))
    return (-((1.0 - p_t) ** gamma) * logp_t).mean()


def test_fast_focal_matches_naive_reference():
    torch.manual_seed(0)
    logits = torch.randn(32, 5, dtype=torch.float32)
    targets = torch.randint(0, 5, (32,))
    got = FastFocalLoss(gamma=2.0)(logits, targets, dataset_name="nsl_kdd")
    ref = _naive_ffl(logits, targets, gamma=2.0)
    assert torch.allclose(got, ref, atol=1e-6, rtol=1e-5)


def test_fast_focal_gamma0_equals_cross_entropy():
    torch.manual_seed(1)
    logits = torch.randn(16, 4, dtype=torch.float32)
    targets = torch.randint(0, 4, (16,))
    ffl = FastFocalLoss(gamma=0.0)(logits, targets, dataset_name="nsl_kdd")
    ce = F.cross_entropy(logits, targets)
    assert torch.allclose(ffl, ce, atol=1e-6, rtol=1e-5)


def test_smote_adjusts_k_neighbors_for_tiny_u2r():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 4)).astype(np.float32)
    y = np.array([0] * 20 + [1] * 12 + [4] * 8, dtype=np.int64)
    Xs, ys = apply_smote(X, y, seed=0, inside_cv_fold=True)
    assert Xs.shape[0] >= X.shape[0]
    assert 4 in set(ys.tolist())


def test_smote_rejected_before_split():
    X = np.ones((20, 3), dtype=np.float32)
    y = np.array([0] * 10 + [1] * 10)
    with pytest.raises(RuntimeError, match="INSIDE each CV fold"):
        apply_smote(X, y, inside_cv_fold=False)


def test_zscore_scaler_fit_source_train_only():
    X = np.random.randn(30, 5).astype(np.float32)
    scaler = SourceZScoreScaler()
    with pytest.raises(RuntimeError, match="NSL-KDD training split"):
        scaler.fit(X, dataset_key="unsw_nb15", split="train")
    with pytest.raises(RuntimeError, match="NSL-KDD training split"):
        scaler.fit(X, dataset_key="nsl_kdd", split="test")
    scaler.fit(X, dataset_key="nsl_kdd", split="train")
    z, stats = scaler.transform(X)
    assert z.dtype == np.float32
    assert "clip_frac" in stats


def test_minmax_scaler_fit_source_train_only():
    X = np.random.rand(20, 20).astype(np.float32)
    mm = Common20MinMaxScaler()
    with pytest.raises(RuntimeError, match="NSL-KDD training split"):
        mm.fit(X, dataset_key="cicids2017", split="train")
    mm.fit(X, dataset_key="nsl_kdd", split="train")
    out = mm.transform(X)
    assert out.min() >= -1e-5 and out.max() <= 1 + 1e-5


def test_ffl_blocked_on_zero_shot_domain():
    logits = torch.randn(4, 2)
    y = torch.tensor([0, 1, 0, 1])
    with active_domain("cicids2017"):
        with pytest.raises(ZeroShotViolationError, match="FastFocalLoss"):
            FastFocalLoss()(logits, y)


def test_encode_labels_five_class_and_binary():
    import pandas as pd

    df = pd.DataFrame({"label": ["normal", "neptune", "satan", "guess_passwd", "buffer_overflow"]})
    out = encode_labels(df)
    assert list(out["label_multi"]) == [0, 1, 2, 3, 4]
    assert list(out["label_bin"]) == [0, 1, 1, 1, 1]

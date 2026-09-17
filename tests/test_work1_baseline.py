from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.baselines.work1_dualphase import (
    PHASE2_DROPOUT,
    PHASE2_HEADS,
    PHASE2_LAYERS,
    PHASE2_LSTM,
    PhaseIIWork1,
    PhaseIWork1,
    Work1DualPhase,
)
from src.data.datamodule import SPLITS_PATH, ensure_nsl_kdd_splits, load_nsl_kdd_splits


def test_phase2_architecture():
    m = PhaseIIWork1(in_dim=20)
    assert m.n_layers == PHASE2_LAYERS == 2
    assert m.hidden == PHASE2_LSTM == 128
    assert m.n_heads == PHASE2_HEADS == 6
    assert abs(m.dropout_p - PHASE2_DROPOUT) < 1e-9
    assert m.lstm.num_layers == 2
    assert m.lstm.bidirectional
    assert m.lstm.hidden_size == 128
    assert m.mha.num_heads == 6
    x = torch.randn(4, 16, 20)
    y = m(x)
    assert y.shape == (4,)
    assert torch.isfinite(y).all()
    assert (y >= 0).all() and (y <= 1).all()


def test_phase1_sigmoid_and_dropout():
    m = PhaseIWork1(in_dim=20)
    assert abs(m.dropout.p - 0.5) < 1e-9
    p = m(torch.randn(8, 20))
    assert p.shape == (8,)
    assert (p >= 0).all() and (p <= 1).all()


def test_cascade_skips_phase2_on_normals():
    m = Work1DualPhase(in_dim=20)
    with torch.no_grad():
        m.phase1.out.bias.fill_(-10.0)  # force Phase I "normal"
        x = torch.randn(5, 20)
        out = m(x)
    assert out["flag"].sum() == 0
    assert torch.allclose(out["cascade"], out["phase1"])


def test_frozen_nsl_kdd_splits_consumed():
    path = Path(SPLITS_PATH)
    if not path.exists():
        y = np.concatenate([np.zeros(70, dtype=np.int64), np.ones(30, dtype=np.int64)])
        splits = ensure_nsl_kdd_splits(y, path, overwrite=False)
    else:
        splits = load_nsl_kdd_splits(path)
    for k in ("train_idx", "val_idx", "test_idx", "fold_0_train", "fold_0_val"):
        assert k in splits
        assert splits[k].size > 0
    # Frozen file is not rewritten by ensure when it exists.
    assert path.exists()
    a = load_nsl_kdd_splits(path)
    b = ensure_nsl_kdd_splits(np.zeros(10, dtype=np.int64), path, overwrite=False)
    np.testing.assert_array_equal(a["fold_0_train"], b["fold_0_train"])

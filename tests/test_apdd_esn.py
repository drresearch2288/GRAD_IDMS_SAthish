"""APDD-ESN reservoir, pyramid conv, and reservoir attention."""

from __future__ import annotations

import numpy as np
import torch
import pytest

from src.models.apdd_esn import ARAPDDESN, EchoStateReservoir, PyramidDilatedConv
from src.models.attention import ReservoirAttention


def test_pyramid_preserves_length():
    conv = PyramidDilatedConv(in_ch=1, out_ch=4)
    x = torch.randn(2, 1, 17)
    y = conv(x)
    assert y.shape[0] == 2 and y.shape[-1] == 17
    assert y.shape[1] == 16  # 4 branches × 4 ch


def test_reservoir_buffers_not_parameters():
    esn = EchoStateReservoir(in_dim=8, reservoir_size=32, sparsity=0.1, spectral_radius=0.9)
    names = {n for n, _ in esn.named_parameters()}
    assert "W_res" not in names and "W_in" not in names
    assert esn.W_res.requires_grad is False and esn.W_in.requires_grad is False
    eigs = np.linalg.eigvals(esn.W_res.cpu().numpy())
    rho = float(np.max(np.abs(eigs)))
    assert rho == pytest.approx(0.9, rel=0.05)
    u = torch.randn(3, 5, 8)
    s = esn(u)
    assert s.shape == (3, 5, 32)


def test_reservoir_attention_weights_btt():
    att = ReservoirAttention(reservoir_size=16, num_heads=4, dropout=0.0)
    s = torch.randn(2, 6, 16)
    out, w = att(s)
    assert out.shape == (2, 6, 16)
    assert w.shape == (2, 6, 6)


def test_arapddesn_logits_and_attention():
    model = ARAPDDESN(in_dim=20, n_classes=5, reservoir_size=16, conv_channels=4, n_heads=4)
    x = torch.randn(3, 20)
    logits, attn = model(x, return_attention=True)
    assert logits.shape == (3, 5)
    assert attn.dim() == 3 and attn.size(0) == 3
    logits.sum().backward()
    # reservoir frozen
    assert model.reservoir.W_res.grad is None
    assert model.readout.weight.grad is not None

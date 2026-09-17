from __future__ import annotations

import inspect

import pytest
import torch

from src.utils.device import assert_no_cuda, get_device, to_float32


def test_get_device_never_cuda():
    device = get_device()
    assert "cuda" not in str(device).lower()
    assert device.type in {"mps", "cpu"}
    src = inspect.getsource(get_device)
    assert "cuda" not in src.lower()


def test_assert_no_cuda():
    with pytest.raises(RuntimeError):
        assert_no_cuda("cuda:0")
    assert_no_cuda("cpu")
    assert_no_cuda("mps")


def test_float32_cast():
    x = torch.ones(3, dtype=torch.float64)
    y = to_float32(x)
    assert y.dtype == torch.float32

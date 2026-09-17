"""Protocol-integrity tests for GRAD-IDMS zero-shot gradient guards."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.utils.guards import (
    GradientGuard,
    ZeroShotViolationError,
    active_domain,
    assert_task_loss_source_only,
)


def _guarded_sgd() -> tuple[nn.Parameter, GradientGuard]:
    param = nn.Parameter(torch.ones(4, dtype=torch.float32))
    opt = torch.optim.SGD([param], lr=0.1)
    return param, GradientGuard(opt)


def test_optimizer_step_allowed_on_nsl_kdd():
    param, opt = _guarded_sgd()
    with active_domain("nsl_kdd", allow_grad=True):
        loss = (param ** 2).sum()
        loss.backward()
        opt.step()
    assert param.grad is not None


def test_optimizer_step_raises_on_bot_iot():
    _, opt = _guarded_sgd()
    with active_domain("bot_iot"):
        with pytest.raises(ZeroShotViolationError, match="bot_iot"):
            opt.step()


def test_task_loss_forbidden_on_cicids2017():
    with pytest.raises(ZeroShotViolationError):
        assert_task_loss_source_only("cicids2017")


def test_domain_loss_allowed_on_unsw_nb15():
    assert_task_loss_source_only("unsw_nb15", loss_kind="domain")


def test_zero_shot_inference_mode_drops_grad_fn():
    x = torch.ones(8, dtype=torch.float32, requires_grad=True)
    with active_domain("ton_iot"):
        y = x * 3.0
        assert y.grad_fn is None

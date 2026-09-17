"""GRL identity/backward, Ganin λ schedule, domain-batch balance."""

from __future__ import annotations

import torch

from src.models.grl import (
    GradientReversalFunction,
    LambdaWarmupGuard,
    assert_balanced_domain_batch,
    ganin_lambda,
    grad_reverse,
)


def test_grl_forward_is_identity():
    x = torch.randn(5, 7, requires_grad=True)
    lam = x.new_tensor(0.4)
    y = GradientReversalFunction.apply(x, lam)
    assert torch.equal(y, x)
    assert torch.allclose(y, x.view_as(x))


def test_grl_backward_is_neg_lambda_grad():
    x = torch.randn(6, 3, requires_grad=True)
    lam = 0.37
    y = GradientReversalFunction.apply(x, x.new_tensor(lam))
    incoming = torch.randn_like(x)
    y.backward(incoming)
    assert torch.allclose(x.grad, -lam * incoming)


def test_grad_reverse_matches_manual():
    x = torch.randn(4, 2, requires_grad=True)
    y = grad_reverse(x, 1.0)
    y.sum().backward()
    assert torch.allclose(x.grad, -torch.ones_like(x))


def test_ganin_lambda_bounds_and_monotonic():
    assert ganin_lambda(0.0) == 0.0
    assert abs(ganin_lambda(1.0) - 0.9999) < 5e-4
    prev = -1.0
    for t in [i / 20 for i in range(21)]:
        v = ganin_lambda(t)
        assert v >= prev - 1e-12
        prev = v
    assert ganin_lambda(1.0, lambda_max=0.5) == 0.5 * ganin_lambda(1.0, lambda_max=1.0)


def test_lambda_warmup_guard_freezes_after_three_drops():
    g = LambdaWarmupGuard(drop_pp=0.02, patience=3)
    assert g.update(0.90) is False
    assert g.update(0.87) is False  # -3pp
    assert g.update(0.86) is False
    assert g.update(0.85) is True
    assert g.frozen is True


def test_balanced_domain_batch_assertion():
    assert_balanced_domain_batch(8, 8)
    try:
        assert_balanced_domain_batch(8, 7)
        raise AssertionError("expected balance assert")
    except AssertionError as exc:
        assert "balanced" in str(exc).lower()

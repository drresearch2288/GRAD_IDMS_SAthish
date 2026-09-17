from __future__ import annotations

import math

import torch

from src.config import GRADConfig
from src.losses.composite import CompositeLoss
from src.models.da_dnn import lambda_schedule
from src.models.grad_idms import GRADIDMS
from src.utils.device import get_device


def test_lambda_schedule_bounds():
    assert lambda_schedule(0.0) == 0.0
    assert abs(lambda_schedule(1.0) - (2 / (1 + math.exp(-10)) - 1)) < 1e-6
    assert 0 < lambda_schedule(0.5) < 1


def test_tiny_forward_and_composite_loss():
    cfg = GRADConfig.tiny()
    device = torch.device("cpu")
    model = GRADIDMS(cfg).to(device)
    x = torch.randn(2, cfg.graph.window, cfg.unified_dim, dtype=torch.float32)
    out = model(x, lambd=0.3, run_phase2=True)
    assert out["phase1_logits"].shape[:2] == (2, cfg.graph.window)
    assert out["phase2_logits"].shape == (2, cfg.phase2.n_classes)
    assert out["fused"].dtype == torch.float32
    y = torch.randint(0, 2, (2 * cfg.graph.window,))
    loss_fn = CompositeLoss()
    loss = loss_fn(
        out["phase1_logits"].reshape(-1, 2),
        y,
        domain_logits=out["domain_logits"].mean(1),
        domain_target=torch.zeros(2, dtype=torch.long),
        lambd=0.3,
        gat_alpha=out["gat_alpha"],
    )
    loss.backward()
    grads = [p.grad.abs().sum() for p in model.parameters() if p.grad is not None]
    assert sum(g.item() for g in grads) > 0


def test_get_device_available():
    d = get_device()
    assert d.type in {"cpu", "mps"}

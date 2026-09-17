"""Domain-valid FGSM/PGD/DeepFool and Phase II adversarial mix."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.adversarial.adv_train import adversarial_step
from src.adversarial.attacks import (
    NON_PERTURBABLE,
    apply_feature_mask,
    deepfool,
    fgsm,
    pgd,
    project_to_valid_flow,
)
from src.config import GRADConfig
from src.data.schema import COMMON_FEATURES


def test_fgsm_pgd_box_and_mask():
    linear = nn.Linear(20, 2)

    def fn(t):
        return linear(t)

    x = torch.rand(4, 20)
    y = torch.tensor([0, 1, 0, 1])
    adv = fgsm(fn, x, y, 0.05)
    assert adv.shape == x.shape
    assert float(adv.min()) >= 0 and float(adv.max()) <= 1
    # protocol dims should match clean
    proto = [COMMON_FEATURES.index(n) for n in ("proto_tcp", "proto_udp", "proto_icmp")]
    assert torch.allclose(adv[:, proto], x[:, proto], atol=1e-5)
    adv_p = pgd(fn, x, y, eps=0.03, steps=3, random_start=True)
    assert adv_p.shape == x.shape


def test_project_to_valid_flow_derived():
    x = torch.rand(2, 20)
    x[:, 1] = 0.3
    x[:, 2] = 0.1
    out = project_to_valid_flow(x, x_clean=x)
    assert torch.allclose(out[:, 3], out[:, 1] + out[:, 2], atol=1e-5)
    tot = (out[:, 1] + out[:, 2]).clamp_min(1e-8)
    assert torch.allclose(out[:, 4], out[:, 1] / tot, atol=1e-5)


def test_apply_feature_mask_zeros_non_perturbable():
    delta = torch.ones(1, 20)
    masked = apply_feature_mask(delta)
    for name in NON_PERTURBABLE:
        i = COMMON_FEATURES.index(name)
        assert float(masked[0, i]) == 0.0
    assert float(masked[0, COMMON_FEATURES.index("src_bytes")]) == 1.0


def test_deepfool_eval_shape():
    linear = nn.Linear(20, 3)
    x = torch.rand(2, 20)
    y = torch.tensor([0, 1])
    adv = deepfool(linear, x, y, max_iter=5)
    assert adv.shape == x.shape


def test_adversarial_step_scalar():
    cfg = GRADConfig.tiny()
    model = nn.Linear(cfg.unified_dim, cfg.phase2.n_classes)
    x = torch.rand(3, cfg.unified_dim)
    y = torch.randint(0, cfg.phase2.n_classes, (3,))
    loss = adversarial_step(model, x, y, cfg)
    assert loss.ndim == 0
    loss.backward()
    assert model.weight.grad is not None

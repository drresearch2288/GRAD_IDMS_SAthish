"""Composite loss terms, graph-attention entropy, float32 backward."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.losses import CompositeLoss, FastFocalLoss, GraphAttentionEntropyRegulariser
from src.utils.device import get_device
from src.utils.guards import active_domain


def _complete_graph(n: int = 4):
    src, dst = [], []
    for i in range(n):
        for j in range(n):
            src.append(j)
            dst.append(i)
    ei = torch.tensor([src, dst], dtype=torch.long)
    alpha = torch.ones(n * n, dtype=torch.float32)
    return ei, alpha


def test_composite_zero_weights_equals_ffl():
    ffl = FastFocalLoss(gamma=2.0)
    cl = CompositeLoss(gamma_graph=0.0, beta_adv=0.0, ffl_gamma=2.0, log_path=None)
    x = torch.randn(8, 5, dtype=torch.float32)
    y = torch.randint(0, 5, (8,))
    with active_domain("nsl_kdd", allow_grad=True):
        plain = ffl(x, y, dataset_name="nsl_kdd")
        total, comps = cl(
            {"logits": x, "domain_logits": None, "adv_logits": None, "gat_alpha": None},
            {"y": y, "dataset_name": "nsl_kdd"},
            lambda_t=0.0,
            beta_t=0.0,
        )
    assert torch.allclose(total, plain, atol=1e-6)


def test_graph_entropy_uniform_vs_onehot():
    reg = GraphAttentionEntropyRegulariser()
    ei_u, a_u = _complete_graph(6)
    h_uni = reg([(ei_u, a_u)])
    # one-hot: a single self-loop per node
    n = 6
    ei_o = torch.stack([torch.arange(n), torch.arange(n)])
    a_o = torch.ones(n, dtype=torch.float32)
    h_one = reg([(ei_o, a_o)])
    assert float(h_uni) > float(h_one)
    assert float(h_one) < 0.05
    assert float(h_uni) > 0.5


def test_components_sum_to_total():
    cl = CompositeLoss(gamma_graph=0.1, beta_adv=0.5, log_path=None)
    logits = torch.randn(6, 2, dtype=torch.float32, requires_grad=True)
    y = torch.tensor([0, 1, 0, 1, 0, 1])
    dlog = torch.randn(6, 2, dtype=torch.float32, requires_grad=True)
    dy = torch.tensor([0, 0, 1, 1, 0, 1])
    adv = torch.randn(6, 2, dtype=torch.float32, requires_grad=True)
    ei, alpha = _complete_graph(4)
    alpha = alpha.clone().requires_grad_(True)
    with active_domain("nsl_kdd", allow_grad=True):
        total, comps = cl(
            {"logits": logits, "domain_logits": dlog, "adv_logits": adv, "gat_alpha": [(ei, alpha)]},
            {"y": y, "domain_y": dy, "dataset_name": "nsl_kdd"},
            lambda_t=0.3,
            beta_t=0.5,
        )
    recon = comps["FFL"] + comps["lambda_L_domain"] + comps["beta_L_adv"] + comps["gamma_L_graph_reg"]
    assert torch.allclose(total, recon, atol=1e-6)
    assert torch.allclose(total, comps["total"], atol=1e-6)


def test_float32_and_backward_mps_or_cpu():
    device = get_device()
    cl = CompositeLoss(gamma_graph=0.1, beta_adv=0.5, log_path=None).to(device)
    lin = nn.Linear(8, 2).to(device)
    x = torch.randn(5, 8, dtype=torch.float32, device=device)
    y = torch.zeros(5, dtype=torch.long, device=device)
    with active_domain("nsl_kdd", allow_grad=True):
        logits = lin(x)
        total, comps = cl(
            {"logits": logits},
            {"y": y, "dataset_name": "nsl_kdd"},
            lambda_t=0.0,
            beta_t=0.0,
        )
    assert total.dtype == torch.float32
    for k in ("FFL", "L_domain", "L_adv", "L_graph_reg", "total"):
        assert comps[k].dtype == torch.float32
    total.backward()
    assert lin.weight.grad is not None
    assert lin.weight.grad.dtype == torch.float32

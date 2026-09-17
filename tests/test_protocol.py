"""Protocol and component tests (CPU/MPS, synthetic domains)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.config import GRADConfig
from src.data.graph import cosine_knn_adj
from src.data.synthetic import generate_domain
from src.models.grad_idms import GRADIDMS
from src.models.losses import dann_lambda
from src.utils.device import get_device, to_float32
from src.utils.guards import check_target_domain_guard, check_zero_shot_guard


def test_lambda_schedule_bounds():
    assert dann_lambda(0.0) == pytest.approx(0.0, abs=1e-6)
    assert 0.9 < dann_lambda(1.0) <= 1.0
    assert dann_lambda(0.5) > dann_lambda(0.1)


def test_cuda_never_selected():
    d = get_device()
    assert "cuda" not in str(d).lower()


def test_unsw_task_guard():
    with pytest.raises(RuntimeError, match="UNSW-NB15"):
        check_target_domain_guard("unsw_nb15", "task")
    check_target_domain_guard("unsw_nb15", "domain")


def test_zero_shot_guard():
    with pytest.raises(RuntimeError, match="zero-shot"):
        check_zero_shot_guard("ton_iot", is_training=True)
    check_zero_shot_guard("ton_iot", is_training=False)


def test_knn_graph_self_loops_and_tau():
    x = torch.eye(8, dtype=torch.float32)
    adj = cosine_knn_adj(x, k=2, tau=0.4)
    assert adj.shape == (8, 8)
    assert torch.allclose(adj.diag(), torch.ones(8))
    assert adj.dtype == torch.float32


def test_float32_cast():
    t = to_float32(torch.arange(4, dtype=torch.float64))
    assert t.dtype == torch.float32


def test_forward_shapes_mps_or_cpu():
    cfg = GRADConfig.tiny()
    device = torch.device("cpu")
    model = GRADIDMS(cfg).to(device)
    n, f = 16, cfg.unified_dim
    x = torch.randn(n, f, device=device, dtype=torch.float32)
    adj = cosine_knn_adj(x, k=4, tau=0.0)
    logits, domain, alpha, h2 = model.forward_phase1(x, adj, lambd=0.5, return_domain=True)
    assert logits.shape == (n, 2)
    assert domain.shape == (n, 2)
    assert alpha.shape == (n, n)
    assert h2.shape[0] == n
    y2, rec = model.forward_phase2(x)
    assert y2.shape == (n, cfg.phase2.n_classes)
    assert rec.ndim == 0 or rec.shape == x.shape


def test_synthetic_domains_differ():
    a = generate_domain("nsl_kdd", 200, 0)
    b = generate_domain("ton_iot", 200, 0)
    assert a["X"].dtype == np.float32
    assert not np.allclose(a["X"].mean(0), b["X"].mean(0), atol=0.05)

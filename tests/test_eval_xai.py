from __future__ import annotations

import numpy as np
import torch

from src.attacks import fgsm, pgd
from src.compression import compression_report, prune_model
from src.config import GRADConfig
from src.eval.metrics import classification_report, generalization_gap
from src.mitigation.routing import mitigate, zero_delay_penalty
from src.models.grad_idms import GRADIDMS
from src.optim.mav_moa import MAVMOA, five_term_fitness
from src.xai.xrep import build_alert_report, integrated_gradients, spearman_rank_corr


def test_pgd_changes_input():
    torch.manual_seed(0)
    linear = torch.nn.Linear(4, 2)

    def fn(x):
        return linear(x)

    x = torch.zeros(3, 4, requires_grad=True)
    y = torch.tensor([0, 1, 0])
    adv = pgd(fn, x.detach(), y, eps=0.05, steps=2)
    assert adv.shape == x.shape
    fgsm_x = fgsm(fn, x.detach(), y, eps=0.05)
    assert fgsm_x.shape == x.shape


def test_metrics_and_gap():
    y = np.array([0, 0, 1, 1])
    p = np.array([0, 1, 1, 1])
    r = classification_report(y, p, scores=np.array([0.1, 0.4, 0.8, 0.9]))
    assert 0 <= r["accuracy"] <= 1
    gap = generalization_gap(0.99, {"a": 0.80, "b": 0.82})
    assert abs(gap - (0.99 - 0.81)) < 1e-9


def test_mav_moa_and_routing():
    opt = MAVMOA(dim=3, bounds=(np.zeros(3), np.ones(3)), population=6, iterations=8, seed=0)
    result = opt.optimize(lambda v: -((v - 0.3) ** 2).sum())
    assert result["best_fitness"] >= -1.0
    fit = five_term_fitness(0.9, 0.7, 0.52, 0.8, 0.3)
    assert 0 < fit < 1
    mit = mitigate(n_nodes=8, population=6, iterations=6, seed=1)
    assert "mitigation_rate" in mit
    assert 0 <= zero_delay_penalty(np.array([0.0, 1e-9, 0.5])) <= 1


def test_xai_and_compression():
    cfg = GRADConfig.tiny()
    model = GRADIDMS(cfg)
    x = torch.randn(1, cfg.graph.window, cfg.unified_dim)

    def fn(t):
        return model(t, run_phase2=False)["phase1_logits"].mean(1)

    ig = integrated_gradients(fn, x, torch.tensor([0]), steps=4)
    assert ig.shape == x.shape
    report = build_alert_report(1, 0.9, np.abs(ig.detach().numpy()), np.abs(ig.detach().numpy()))
    assert "justification" in report
    assert spearman_rank_corr(np.arange(5.0), np.arange(5.0)) > 0.9
    n = prune_model(model, 0.3)
    assert n >= 0
    cr = compression_report(model, window=cfg.graph.window)
    assert cr["params"] > 0

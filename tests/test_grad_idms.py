from __future__ import annotations

import torch

from src.config import GRADConfig
from src.models.grad_idms import GradIDMS, GRADIDMS


def test_cascade_window_shapes():
    cfg = GRADConfig.tiny()
    model = GRADIDMS(cfg)
    x = torch.randn(2, cfg.graph.window, cfg.unified_dim)
    out = model(x, lambda_t=0.3, return_all=True)
    assert out["coarse_logits"].shape[:2] == (2, cfg.graph.window)
    assert out["phase2_logits"].shape == (2, cfg.phase2.n_classes)
    assert "forwarding_rate" in out
    assert out["fused"].shape[-1] == model.ea_scaf.out_dim
    assert out["shared_fused"].shape == (2 * cfg.graph.window, model.ea_scaf.out_dim)


def test_empty_mask_skips_phase2():
    cfg = GRADConfig.tiny()
    model = GradIDMS(cfg)
    model.phase1.forward_threshold = 1.1  # nothing exceeds 1.0
    x = torch.randn(2, cfg.graph.window, cfg.unified_dim)
    out = model(x, run_phase2=True)
    assert int(out["mask"].sum()) == 0
    assert torch.all(out["final_pred"] == 0)
    assert out["phase2_logits"].shape == (2, 5)


def test_use_graph_false_raw_features():
    cfg = GRADConfig.tiny()
    model = GradIDMS(cfg, use_graph=False)
    assert model.ea_scaf.ensemble.ae1.enc1.in_features == cfg.unified_dim
    x = torch.randn(2, cfg.graph.window, cfg.unified_dim)
    out = model(x)
    assert out["fused"].shape[-1] == model.ea_scaf.out_dim
    assert out["phase1_logits"].shape[-1] == 2


def test_use_domain_forces_lambda_zero():
    cfg = GRADConfig.tiny()
    model = GradIDMS(cfg, use_domain=False)
    x = torch.randn(1, cfg.graph.window, cfg.unified_dim)
    out = model(x, lambda_t=1.0)
    assert out["domain_logits"] is None


def test_count_parameters_and_flops():
    cfg = GRADConfig.tiny()
    model = GradIDMS(cfg)
    counts = model.count_parameters()
    assert counts["phase2_reservoir"]["non_trainable"] > 0
    assert counts["fg_gat"]["trainable"] > 0
    x = torch.randn(1, cfg.graph.window, cfg.unified_dim)
    n = model.flops(x)
    assert n == model.last_flops["total"]
    for key in ("gat_message_passing", "ea_scaf", "phase1", "phase2_reservoir", "attention"):
        assert key in model.last_flops


def test_phase2_does_not_overrule_phase1_normal():
    cfg = GRADConfig.tiny()
    model = GradIDMS(cfg)
    x = torch.randn(1, cfg.graph.window, cfg.unified_dim)
    out = model(x)
    mask = out["mask"].view(-1)
    final = out["final_pred"].view(-1)
    assert torch.all(final[~mask] == 0)

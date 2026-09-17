from __future__ import annotations

import torch

from src.ablations import (
    INDEPENDENT_NAMES,
    LADDER_NAMES,
    ablation_path,
    build_model,
    ladder_flag_diff,
    load_named_ablation,
)
from src.baselines.dann_generic import GenericDANN
from src.config import GRADConfig


def test_every_ablation_yaml_builds():
    cfg = GRADConfig.tiny()
    names = INDEPENDENT_NAMES + LADDER_NAMES
    x = torch.randn(1, cfg.graph.window, cfg.unified_dim)
    for name in names:
        assert ablation_path(name).exists(), name
        flags = load_named_ablation(name)
        model = build_model(cfg, flags)
        model.eval()
        with torch.no_grad():
            out = model(x, lambda_t=0.2 if flags.use_domain else 0.0)
        assert "phase1_logits" in out or "coarse_logits" in out
        if flags.compressed:
            assert model.cfg.phase2.reservoir_size <= cfg.phase2.reservoir_size
        if not flags.use_graph:
            assert model.ea_scaf.ensemble.ae1.enc1.in_features == cfg.unified_dim
        if not flags.use_domain:
            assert out.get("domain_logits") is None


def test_ladder_one_flag_at_a_time():
    flags = [load_named_ablation(n) for n in LADDER_NAMES]
    for prev, nxt, name in zip(flags, flags[1:], LADDER_NAMES[1:]):
        diff = ladder_flag_diff(prev, nxt)
        assert len(diff) == 1, f"{name} changed {diff}"
    # Independent variants each flip a single default-True switch (except compressed).
    full = load_named_ablation("v6_full")
    assert full.use_graph and full.use_domain and full.use_adv and full.use_zd_penalty and full.use_xai
    assert load_named_ablation("b4_no_graph").use_graph is False
    assert load_named_ablation("b5_no_domain").use_domain is False
    assert load_named_ablation("b6_no_adv").use_adv is False
    assert load_named_ablation("b7_smote").use_ffl is False
    assert load_named_ablation("b8_compressed").compressed is True


def test_generic_dann_has_no_phase2_or_mitigation():
    m = GenericDANN(in_dim=20, n_classes=5)
    assert m.phase2 is None
    assert m.mitigation is None
    assert not hasattr(m, "fg_gat")
    x = torch.randn(4, 20)
    out = m(x, lambda_t=0.5)
    assert out["logits"].shape == (4, 5)
    assert out["domain_logits"].shape[0] == 4
    assert out["h"].shape[-1] == 32

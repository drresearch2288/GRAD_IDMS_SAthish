from __future__ import annotations

import torch

from src.compression.distillation import kd_loss, make_student, make_student_config
from src.compression.edge import calibrated_latency, compression_report, prune_model, reservoir_fraction
from src.compression.pruning import structured_prune_model
from src.compression.qat import prepare_qat, qat_report, reservoir_spectral_radius
from src.config import GRADConfig
from src.models.grad_idms import GradIDMS
from src.models.losses import FastFocalLoss
from src.utils.guards import active_domain


def test_reservoir_fraction_and_rho():
    cfg = GRADConfig.tiny()
    model = GradIDMS(cfg)
    frac = reservoir_fraction(model)
    assert frac["reservoir_numel"] > 0
    assert 0 < frac["reservoir_fraction"] < 1
    rho = reservoir_spectral_radius(model)
    assert 0.5 < rho < 1.2
    cr = compression_report(model, window=cfg.graph.window)
    assert cr["params"] > 0
    assert "reservoir_fraction" in cr


def test_structured_prune_rebuilds_smaller_readout():
    cfg = GRADConfig.tiny()
    model = GradIDMS(cfg)
    old_in = model.phase2.readout.in_features
    rep = structured_prune_model(model, 0.30)
    assert any("readout" in r["module"] for r in rep["rebuilt"] if "module" in r)
    slim = model.phase2.readout
    assert hasattr(slim, "linear")
    assert slim.linear.in_features <= old_in
    x = torch.randn(2, slim.linear.in_features)
    y = slim.linear(x)
    assert y.shape[-1] == cfg.phase2.n_classes
    n = prune_model(model, 0.3)
    assert n >= 0


def test_kd_loss_and_student_width():
    cfg = GRADConfig.tiny()
    teacher = GradIDMS(cfg)
    student = make_student(teacher, cfg)
    sc = make_student_config(cfg)
    assert student.fg_gat.hidden_dim == sc.graph.gat_hidden
    assert student.phase2.reservoir_size == sc.phase2.reservoir_size
    s = torch.randn(4, 5, requires_grad=True)
    t = torch.randn(4, 5)
    y = torch.zeros(4, dtype=torch.long)
    with active_domain("nsl_kdd", allow_grad=True):
        loss = kd_loss(s, t, y, ffl=FastFocalLoss())
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()


def test_qat_skips_reservoir_and_stays_cpu():
    cfg = GRADConfig.tiny()
    model = GradIDMS(cfg).cpu()
    rho0 = reservoir_spectral_radius(model)
    prepare_qat(model)
    rho1 = reservoir_spectral_radius(model)
    assert abs(rho1 - rho0) < 1e-5
    w = model.phase2.reservoir.W_res
    assert w.dtype == torch.float32
    rep = qat_report(model)
    assert rep["engine"] == "qnnpack"
    assert rep["reservoir_fp32"] is True
    assert rep["mps_quantized_kernels"] is False


def test_calibrated_latency_protocol():
    n = {"i": 0}

    def fn():
        n["i"] += 1

    stats = calibrated_latency(fn, warmup=3, repeats=7)
    assert n["i"] == 10
    assert stats["n"] == 7
    assert stats["warmup"] == 3
    assert stats["p50_ms"] >= 0
    assert stats["p95_ms"] >= stats["p50_ms"]

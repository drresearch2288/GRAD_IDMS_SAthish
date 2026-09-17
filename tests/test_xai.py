from __future__ import annotations

import numpy as np
import torch

from src.config import GRADConfig
from src.data.schema import COMMON_FEATURES
from src.models.grad_idms import GRADIDMS
from src.xai.attention_rollout import attention_rollout, top_graph_neighbours
from src.xai.integrated_gradients import IGExplainer, shap_ig_stability, spearman_rank_corr
from src.xai.report import IDSAlertReport, build_ids_alert_report, explain_route
from src.xai.shap_explainer import SHAPExplainer, kmeans_background, on_cpu


def test_attention_rollout_abnar():
    t = 4
    a1 = torch.eye(t)
    a2 = torch.zeros(t, t)
    a2[:, 0] = 1.0
    R = attention_rollout([a1, a2], add_residual=True)
    assert R.shape == (t, t)
    assert torch.isfinite(R).all()
    # residual + row-norm keeps rows summing to 1
    assert torch.allclose(R.sum(-1), torch.ones(t), atol=1e-5)
    nbr = top_graph_neighbours(R, 1, k=2)
    assert 1 not in nbr and len(nbr) == 2


def test_ig_captum_and_spearman():
    lin = torch.nn.Linear(20, 2)
    x = torch.randn(2, 20, requires_grad=True)
    ig = IGExplainer(lin, n_steps=8).attribute(x, target=torch.zeros(2, dtype=torch.long))
    assert ig.shape == x.shape
    rho = spearman_rank_corr(np.arange(10.0), np.arange(10.0))
    assert rho > 0.9
    stab = shap_ig_stability(np.arange(20.0), np.arange(20.0) * 2)
    assert stab["mean_rho"] >= 0.85
    bad = shap_ig_stability(np.arange(20.0), np.arange(20.0)[::-1])
    assert bad["finding"] is not None


def test_explain_route_rejects_on_term():
    chosen = [0, 1, 2]
    comps = [
        {"path": chosen, "mitigation_rate": 0.9, "energy": 0.1, "zero_delay_penalty": 0.1, "w1": 0.25, "w2": 0.25, "fitness": 0.9 - 0.025 - 0.025},
        {"path": [0, 3, 2], "mitigation_rate": 0.4, "energy": 0.1, "zero_delay_penalty": 0.1, "w1": 0.25, "w2": 0.25, "fitness": 0.4 - 0.025 - 0.025},
        {"path": [0, 4, 2], "mitigation_rate": 0.9, "energy": 2.0, "zero_delay_penalty": 0.1, "w1": 0.25, "w2": 0.25, "fitness": 0.9 - 0.5 - 0.025},
        {"path": [0, 5, 2], "mitigation_rate": 0.9, "energy": 0.1, "zero_delay_penalty": 1.0, "w1": 0.25, "w2": 0.25, "fitness": 0.9 - 0.025 - 0.25},
    ]
    out = explain_route(chosen, comps)
    assert out["chosen_route"] == chosen
    assert len(out["rejected"]) == 3
    assert out["rejected"][0]["rejected_because"] in {"mitigation_rate", "energy", "zero_delay_penalty"}


def test_ids_alert_report_serializers():
    shap = np.zeros(20)
    shap[1] = 0.5
    x = np.ones(20) * 0.2
    ig = shap.copy()
    rep = build_ids_alert_report(
        alert_id="a1",
        flow_id="f9",
        dataset="nsl_kdd",
        predicted_class="DoS",
        confidence=0.91,
        phase1_decision="anomaly",
        phase2_decision="DoS",
        shap_values=shap,
        feature_values=x,
        ig_values=ig,
        temporal_attention=np.linspace(0, 1, 8),
        graph_neighbours=[3, 4],
        mitigation_action="reroute",
        chosen_route=[0, 1, 7],
        route_rationale="avoid compromised node 4",
        explanation_latency_ms=12.0,
    )
    assert isinstance(rep, IDSAlertReport)
    assert "src_bytes" in rep.to_markdown()
    assert "DoS" in rep.to_json()
    assert "<html" in rep.to_html().lower()
    assert len(rep.top5_features) == 5
    assert COMMON_FEATURES[1] == "src_bytes"


def test_shap_on_cpu_context_and_tiny_model():
    cfg = GRADConfig.tiny()
    model = GRADIDMS(cfg)
    device = torch.device("cpu")
    model.to(device)
    with on_cpu(model):
        assert next(model.parameters()).device.type == "cpu"
    bg = kmeans_background(np.random.randn(40, 20), k=8)
    expl = SHAPExplainer(model, background=bg, nsamples=8)
    x = np.random.randn(2, 20).astype(np.float32)
    attr = expl.explain(x, phase=1)
    assert attr.shape == (2, 20)
    assert expl.last_latency_ms >= 0.0

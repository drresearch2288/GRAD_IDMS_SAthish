"""Unit and Integration Tests for Robustness, Mitigation, and XAI Pipelines."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from scripts.run_mitigation import _run_single_cell
from scripts.run_robustness import (
    RobustnessModelWrapper,
    evaluate_adversarial_method,
    evaluate_transfer_attack,
    load_robustness_model,
)
from scripts.run_xai_eval import (
    compute_analyst_alignment,
    compute_explanation_stability,
    compute_faithfulness_curves,
)
from src.config import DEFAULT_CONFIG


def test_robustness_wrapper_and_shapes():
    """Verify RobustnessModelWrapper forward and predict 2D/3D tensor shapes."""
    cfg = DEFAULT_CONFIG
    device = torch.device("cpu")

    # 1. GradIDMS Wrapper
    wrapper = load_robustness_model("gradidms", cfg, device)
    x = torch.randn(10, 20)
    preds, probs = wrapper.predict(x)
    assert preds.shape == (10,)
    assert probs.shape == (10, 5)
    assert np.all((preds >= 0) & (preds < 5))

    # 2. Work1 Wrapper
    w1_wrapper = load_robustness_model("work1", cfg, device)
    w1_preds, w1_probs = w1_wrapper.predict(x)
    assert w1_preds.shape == (10,)
    assert w1_probs.shape[1] >= 2


def test_mitigation_single_cell_execution():
    """Verify single cell execution across optimizers in pure Python/NumPy."""
    # (dataset, n_nodes, opt_name, zd_on, seed, pop_size, max_iter)
    params = ("nsl_kdd", 50, "mavmoa", True, 42, 10, 10)
    res = _run_single_cell(params)

    assert res["dataset"] == "nsl_kdd"
    assert res["n_nodes"] == 50
    assert res["optimizer"] == "mavmoa"
    assert 0.0 <= res["mitigation_rate"] <= 1.0
    assert res["energy_j"] > 0.0
    assert res["delay_ms"] > 0.0
    assert 0.0 <= res["pdr_pct"] <= 100.0
    assert 0.0 <= res["security_score"] <= 1.0
    assert 0.0 <= res["zero_delay_mitigation_rate"] <= 1.0
    assert len(res["path"]) >= 2


def test_xai_faithfulness_and_stability():
    """Verify faithfulness progressive deletion/insertion and stability metrics."""
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(20, 5)

        def forward(self, x):
            if isinstance(x, dict):
                x = x["x"]
            if x.dim() == 3:
                x = x.mean(dim=1)
            return self.linear(x)

    model = DummyModel()
    n_samples = 20
    x_alerts = torch.randn(n_samples, 20)
    y_alerts = np.random.randint(0, 5, size=n_samples)
    attributions = np.random.randn(n_samples, 20)

    # 1. Faithfulness curves
    curves = compute_faithfulness_curves(model, x_alerts, y_alerts, attributions, baseline_value=0.0)
    assert len(curves["deletion_curve"]) == 21  # k=0..20
    assert len(curves["insertion_curve"]) == 21
    assert "auc_deletion" in curves
    assert "auc_insertion" in curves
    assert "faithfulness_gap" in curves

    # 2. Analyst alignment
    alignment = compute_analyst_alignment(attributions, y_alerts)
    assert 0.0 <= alignment["rule_match_precision_pct"] <= 100.0
    assert "sample_size_caveat" in alignment


def test_make_tables_generates_all_ten_tables(tmp_path):
    """Verify that scripts/make_tables.py generates all 10 canonical publication tables."""
    from scripts.make_tables import main as make_tables_main
    rc = make_tables_main(["--out-dir", str(tmp_path)])
    assert rc == 0

    expected_tables = [
        "table1_same_domain.tex",
        "table2_zero_shot.tex",
        "table3_mitigation.tex",
        "table4_robustness.tex",
        "table5_explainability.tex",
        "table6_efficiency.tex",
        "table7_hyperparameters.tex",
        "table8_ablation_ladder.tex",
        "table9_theorems.tex",
        "table10_dataset_summary.tex",
    ]

    for t_name in expected_tables:
        t_file = tmp_path / t_name
        assert t_file.exists(), f"Missing table: {t_name}"
        content = t_file.read_text()
        assert "\\toprule" in content
        assert "\\bottomrule" in content
        assert "\\end{table" in content

"""Unit and statistical tests for src.utils.metrics."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils.metrics import (
    WILCOXON_5FOLD_LIMITATION_NOTE,
    bonferroni_correction,
    bootstrap_ci,
    cliffs_delta,
    compute_cross_domain_metrics,
    compute_efficiency_metrics,
    compute_robustness_metrics,
    compute_same_domain_metrics,
    paired_wilcoxon_test,
    significance_stars,
)


def test_metrics_match_sklearn_binary_and_multiclass():
    """Verify that compute_same_domain_metrics matches sklearn exactly."""
    rng = np.random.default_rng(123)
    y_true = rng.integers(0, 5, size=200)
    y_pred = rng.integers(0, 5, size=200)
    y_scores = rng.random((200, 5))
    y_scores = y_scores / y_scores.sum(axis=1, keepdims=True)

    metrics = compute_same_domain_metrics(y_true, y_pred, y_scores, n_classes=5)

    # 1. Accuracy
    expected_acc = accuracy_score(y_true, y_pred) * 100.0
    assert np.isclose(metrics["accuracy"], expected_acc)

    # 2. Precision & Recall
    expected_prec = precision_score(y_true, y_pred, average="macro", zero_division=0) * 100.0
    expected_rec = recall_score(y_true, y_pred, average="macro", zero_division=0) * 100.0
    assert np.isclose(metrics["precision_macro"], expected_prec)
    assert np.isclose(metrics["recall_macro"], expected_rec)

    # 3. Macro and weighted F1
    expected_macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    expected_weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    assert np.isclose(metrics["macro_f1"], expected_macro_f1)
    assert np.isclose(metrics["weighted_f1"], expected_weighted_f1)

    # 4. MCC & Cohen's kappa
    expected_mcc = matthews_corrcoef(y_true, y_pred)
    expected_kappa = cohen_kappa_score(y_true, y_pred)
    assert np.isclose(metrics["mcc"], expected_mcc)
    assert np.isclose(metrics["cohen_kappa"], expected_kappa)

    # 5. Confusion matrix
    expected_cm = confusion_matrix(y_true, y_pred, labels=list(range(5)))
    assert np.array_equal(np.array(metrics["confusion_matrix"]), expected_cm)

    # 6. Per-class F1
    per_class_expected = f1_score(y_true, y_pred, average=None, zero_division=0, labels=list(range(5)))
    for cls_idx, score in enumerate(per_class_expected):
        assert np.isclose(metrics["per_class_f1"][cls_idx], score)


def test_cliffs_delta_perfect_separation_and_identical():
    """Verify Cliff's delta on perfect separation, reverse separation, and identical sets."""
    x = [10.0, 11.0, 12.0, 13.0, 14.0]
    y = [1.0, 2.0, 3.0, 4.0, 5.0]

    # Perfect separation (x always > y -> delta = +1.0)
    res_pos = cliffs_delta(x, y)
    assert np.isclose(res_pos["delta"], 1.0)
    assert res_pos["interpretation"] == "large"

    # Inverted separation (y always < x -> delta = -1.0)
    res_neg = cliffs_delta(y, x)
    assert np.isclose(res_neg["delta"], -1.0)
    assert res_neg["interpretation"] == "large"

    # Identical distributions (delta = 0.0)
    res_zero = cliffs_delta(x, x)
    assert np.isclose(res_zero["delta"], 0.0)
    assert res_zero["interpretation"] == "negligible"


def test_bonferroni_correction_comparison_count():
    """Verify Bonferroni correction applies multiplier m = n_comparisons."""
    raw_p_values = {
        "svm": 0.01,
        "dnn": 0.005,
        "esn": 0.03,
        "work1": 0.20,
    }
    m = len(raw_p_values)  # 4 comparisons

    corrected = bonferroni_correction(raw_p_values)
    assert len(corrected) == 4

    assert np.isclose(corrected["svm"]["corrected_p"], min(1.0, 0.01 * 4))
    assert np.isclose(corrected["dnn"]["corrected_p"], min(1.0, 0.005 * 4))
    assert np.isclose(corrected["esn"]["corrected_p"], min(1.0, 0.03 * 4))
    assert np.isclose(corrected["work1"]["corrected_p"], min(1.0, 0.20 * 4))

    assert corrected["dnn"]["significant_05"] is True
    assert corrected["work1"]["significant_05"] is False


def test_bootstrap_ci_coverage_on_synthetic_data():
    """Verify that 95% bootstrap CI covers the true population mean ~95% of the time."""
    true_mu = 10.0
    true_sigma = 2.0
    n_trials = 100
    sample_size = 50
    covered = 0

    rng = np.random.default_rng(42)
    for i in range(n_trials):
        synthetic_sample = rng.normal(loc=true_mu, scale=true_sigma, size=sample_size)
        ci_res = bootstrap_ci(synthetic_sample, n_resamples=300, ci=95.0, seed=i)
        if ci_res["ci_lower"] <= true_mu <= ci_res["ci_upper"]:
            covered += 1

    coverage_rate = covered / n_trials
    # Empirical 95% coverage should be between 88% and 100% on 100 trials
    assert 0.88 <= coverage_rate <= 1.0, f"Expected ~95% coverage, got {coverage_rate * 100}%"


def test_paired_wilcoxon_and_limitation_disclosure():
    """Verify Paired Wilcoxon signed-rank test and 5-fold limitation disclosure."""
    x = [0.95, 0.94, 0.96, 0.95, 0.97]
    y = [0.80, 0.82, 0.79, 0.81, 0.80]

    stat, p_val = paired_wilcoxon_test(x, y)
    assert p_val <= 0.0625  # Lower bound on 5 folds
    assert "0.0625" in WILCOXON_5FOLD_LIMITATION_NOTE

    # Significance stars
    assert significance_stars(0.0005) == "***"
    assert significance_stars(0.005) == "**"
    assert significance_stars(0.04) == "*"
    assert significance_stars(0.0625) == "†"
    assert significance_stars(0.50) == ""


def test_robustness_and_efficiency_metrics():
    """Verify robustness and efficiency calculations."""
    rob = compute_robustness_metrics(clean_acc=90.0, robust_acc=72.0)
    assert np.isclose(rob["attack_success_rate"], 28.0)
    assert np.isclose(rob["robust_accuracy_retention"], 80.0)

    eff = compute_efficiency_metrics(
        latencies_ms=[1.0, 2.0, 3.0, 4.0, 5.0],
        model_size_mb=12.5,
        param_counts={"trainable": 1000, "non_trainable": 0, "total": 1000},
        analytic_flops=50000,
    )
    assert np.isclose(eff["latency_mean_ms"], 3.0)
    assert np.isclose(eff["latency_p50_ms"], 3.0)
    assert eff["total_params"] == 1000

from __future__ import annotations

import inspect

import numpy as np
import pytest

from src.optim.baselines_opt import (
    ArithmeticOptimization,
    BorderCollieOptimization,
    GorillaTroopsOptimizer,
    MOA,
)
from src.optim.fitness import HELD_OUT_ZERO_SHOT, five_term_fitness
from src.optim.mavmoa import MAVMOA, r_adaptive


BOUNDS = {"x": (-2.0, 2.0), "y": (-2.0, 2.0)}


def _sphere_max(p: dict) -> float:
    """Maximise −||p − 0.3||² so the peak is at 0.3."""
    return -((p["x"] - 0.3) ** 2 + (p["y"] - 0.3) ** 2)


def test_r_adaptive_eq10():
    assert r_adaptive(1.0, 1.0, 0.0) == pytest.approx(1.0)
    assert r_adaptive(0.0, 1.0, 0.0) == pytest.approx(0.0)
    assert 0.0 < r_adaptive(0.4, 1.0, 0.0) < 1.0


def test_five_term_weights_and_unsw_assert():
    fit = five_term_fitness(0.9, 0.7, 0.52, 0.8, 0.3, eval_key="unsw_nb15")
    expect = (
        0.35 * 0.9
        + 0.25 * 0.7
        + 0.15 * (1.0 - abs(0.5 - 0.52))
        + 0.15 * 0.8
        + 0.10 * 0.3
    )
    assert fit == pytest.approx(expect)
    for bad in HELD_OUT_ZERO_SHOT:
        with pytest.raises(AssertionError, match="unsw_nb15"):
            five_term_fitness(0.9, 0.7, 0.5, 0.8, 0.3, eval_key=bad)


def test_mavmoa_interface_and_history():
    opt = MAVMOA(_sphere_max, BOUNDS, pop_size=12, max_iter=15, seed=42, minimise=False)
    best, f, hist = opt.optimise()
    assert set(best) == {"x", "y"}
    assert isinstance(f, float)
    assert len(hist) == 16
    rec = hist[-1]
    assert {"best", "mean", "worst", "population"} <= set(rec)
    assert len(rec["population"]) == 12
    assert f >= hist[0]["best"] - 1e-9
    assert abs(best["x"] - 0.3) < 0.5


def test_work3_defaults_are_config_not_algorithm():
    src = inspect.getsource(inspect.getmodule(MAVMOA))
    assert "configuration choice" in src.lower()
    assert "not an algorithmic contribution" in src.lower()
    opt = MAVMOA(_sphere_max, BOUNDS)
    assert opt.pop_size == 30
    assert opt.max_iter == 100


def test_baselines_same_interface():
    kw = dict(objective_fn=_sphere_max, bounds=BOUNDS, pop_size=8, max_iter=6, seed=1)
    classes = (GorillaTroopsOptimizer, ArithmeticOptimization, BorderCollieOptimization, MOA)
    for cls in classes:
        opt = cls(**kw)
        best, f, hist = opt.optimise()
        assert set(best) == {"x", "y"}
        assert len(hist) == 7
        assert "best" in hist[-1] and "population" in hist[-1]
        assert np.isfinite(f)

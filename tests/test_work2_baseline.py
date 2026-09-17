from __future__ import annotations

import inspect

import pytest
import torch

from src.baselines.classical import PlainDNN, RidgeESN, TabularResNet152
from src.baselines.work2_mavmoa_apddesn import (
    WORK2_MAX_ITER,
    WORK2_POP_SIZE,
    WORK2_USE_ATTENTION,
    Work2Detector,
    make_work2_mavmoa,
)
from src.optim.fitness import (
    WORK2_FITNESS_TERMS,
    five_term_fitness,
    work2_fitness,
    work2_route_fitness,
)


def test_work2_mavmoa_pop_and_iters():
    opt = make_work2_mavmoa(lambda p: float(p["focal_gamma"]))
    assert opt.pop_size == WORK2_POP_SIZE == 10
    assert opt.max_iter == WORK2_MAX_ITER == 50


def test_work2_detector_has_no_attention_graft():
    m = Work2Detector(in_dim=20, n_classes=2, reservoir_size=16, use_attention=False)
    assert m.use_attention is False
    assert m.detector.use_attention is False
    assert WORK2_USE_ATTENTION is False
    x = torch.randn(3, 20)
    logits, recon = m(x)
    assert logits.shape == (3, 2)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(recon)


def test_work2_fitness_is_detection_only():
    assert WORK2_FITNESS_TERMS == ("f1_detection",)
    assert work2_fitness(0.90) == pytest.approx(0.90)
    src = inspect.getsource(work2_fitness)
    assert "detection-only" in src.lower() or "detection only" in src.lower()
    for term in ("f1_zeroshot", "domain_disc_acc", "robust_acc", "feature_reduction"):
        with pytest.raises(TypeError, match="detection-only"):
            work2_fitness(0.9, **{term: 0.1})
    # Five-term helper remains the Work 3 objective and is a different function.
    five = five_term_fitness(0.9, 0.7, 0.5, 0.8, 0.3, eval_key="unsw_nb15")
    assert work2_fitness(0.9) != pytest.approx(five)
    assert work2_route_fitness(0.9, 0.4, 0.25) == pytest.approx(0.9 - 0.25 * 0.4)


def test_classical_shapes():
    dnn = PlainDNN(20, 2)
    assert dnn.n_hidden == 3
    x = torch.randn(4, 20)
    assert dnn(x).shape == (4, 2)
    rn = TabularResNet152(20, 2)
    assert rn.vector_to_grid(x).shape == (4, 3, 32, 32)
    esn = RidgeESN(20, reservoir_size=16)
    assert not hasattr(esn, "pyramid")
    assert not hasattr(esn.reservoir, "attn")

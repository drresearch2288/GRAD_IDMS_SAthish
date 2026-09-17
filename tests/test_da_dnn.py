"""Phase I DA-DNN: GRL tap at h2, FFL+domain loss, forwarding gate."""

from __future__ import annotations

import torch
import pytest

from src.models.da_dnn import DADNN
from src.utils.guards import ZeroShotViolationError, active_domain


def test_forward_dict_and_h2_is_64d():
    m = DADNN(in_dim=96)
    x = torch.randn(7, 96)
    out = m(x, lambda_=0.2)
    assert out["logits"].shape == (7, 2)
    assert out["domain_logits"].shape == (7, 2)
    assert out["h2"].shape == (7, 64)
    assert out["h3"].shape == (7, 32)
    logits, domain, h3 = m(x, lambd=0.1)  # legacy unpack + lambd alias
    assert logits.shape[0] == 7 and h3.shape[-1] == 32 and domain.shape[-1] == 2


def test_grl_taps_hidden2_not_h3():
    m = DADNN(in_dim=20, hidden=(128, 64, 32))
    assert m.domain_disc.net[0].in_features == 64
    assert m.fc2.out_features == 64


def test_phase1_loss_guard_source_only():
    m = DADNN(in_dim=20)
    x = torch.randn(4, 20)
    y = torch.zeros(4, dtype=torch.long)
    out = m(x)
    with active_domain("nsl_kdd", allow_grad=True):
        loss = m.phase1_loss(out["logits"], y, out["domain_logits"], torch.zeros(4, dtype=torch.long), 0.5)
        loss.backward()
    with pytest.raises(ZeroShotViolationError):
        with active_domain("ton_iot"):
            m.phase1_loss(out["logits"].detach(), y, None, None, 0.0, dataset_name="ton_iot")


def test_filter_anomalies_and_threshold_cap():
    m = DADNN(in_dim=20)
    logits = torch.tensor([[4.0, -4.0], [-3.0, 3.0], [0.1, 0.0], [-2.0, 2.0]])
    y = torch.tensor([0, 1, 0, 1])
    m.fit_temperature(logits, y)
    t = m.tune_forwarding_threshold(logits, y, max_rate=0.35)
    mask, conf = m.filter_anomalies(torch.randn(4, 20), threshold=t, dataset_name="nsl_kdd")
    assert mask.dtype == torch.bool and mask.shape == (4,)
    assert conf.shape == (4,)
    assert 0.0 <= float(conf.min()) <= float(conf.max()) <= 1.0


def test_no_domain_branch():
    m = DADNN(in_dim=96, use_domain_branch=False)
    out = m(torch.randn(2, 96))
    assert out["domain_logits"] is None

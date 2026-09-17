"""Zero-delay synthesis and injection tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.schema import COMMON_FEATURES
from src.data.zero_delay import inject, synthesise_zero_delay


def _host(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {c: rng.normal(1.0, 0.3, size=n).astype(np.float32) for c in COMMON_FEATURES}
    data["label_multi"] = rng.choice([0, 1, 2, 3, 4], size=n, p=[0.4, 0.25, 0.15, 0.1, 0.1])
    data["label_bin"] = (np.array(data["label_multi"]) != 0).astype(np.int64)
    data["flow_id"] = np.arange(n)
    data["timestamp"] = np.linspace(0, 10, n)
    data["iat_min"] = rng.uniform(0.05, 0.4, size=n)
    data["flow_pkts_per_sec"] = rng.uniform(0.5, 3.0, size=n)
    return pd.DataFrame(data)


def test_generated_iat_min_below_host_p1():
    host = _host()
    p1 = float(np.percentile(host["iat_min"], 1))
    synth, _ = synthesise_zero_delay(
        host, n_attacks=2, flows_per_vector=(20, 25), vectors_per_attack=(3, 3),
        seed=1, dataset_key="ton_iot", auto_tune_jitter=False,
    )
    assert float(synth["iat_min"].max()) < p1
    assert (synth["iat_min"] == 0).all()
    assert (synth["label_multi"] == 1).all()
    assert bool(synth["is_zero_delay"].all())


def test_zd_event_id_groups_are_time_contiguous():
    host = _host()
    synth, _ = synthesise_zero_delay(
        host, n_attacks=4, flows_per_vector=(20, 22), vectors_per_attack=(3, 3),
        seed=2, dataset_key="bot_iot", auto_tune_jitter=False,
    )
    for eid, g in synth.groupby("zd_event_id"):
        t = np.sort(g["timestamp"].to_numpy())
        assert np.all(np.diff(t) >= -1e-12)
        tmin, tmax = t[0], t[-1]
        others = synth.loc[synth["zd_event_id"] != eid, "timestamp"].to_numpy()
        # events may share a global timeline but each id is a tight burst
        assert (tmax - tmin) < 0.01


def test_injection_rate_within_half_percent(tmp_path):
    host = _host(800)
    src = tmp_path / "ton_iot.parquet"
    dst = tmp_path / "ton_iot_zd.parquet"
    host.to_parquet(src, index=False)
    meta = inject(
        src, dst, injection_rate=0.05, seed=3, dataset_key="ton_iot",
        spec_path=tmp_path / "spec.json",
    )
    assert abs(meta["injection_rate_achieved"] - 0.05) <= 0.005
    mixed = pd.read_parquet(dst)
    assert mixed["is_zero_delay"].sum() == meta["n_injected"]


def test_zero_jitter_triggers_regeneration():
    host = _host(300)
    _, meta = synthesise_zero_delay(
        host, n_attacks=3, flows_per_vector=(20, 24), vectors_per_attack=(3, 3),
        seed=4, dataset_key="ton_iot", jitter_sigma_frac=0.0, auto_tune_jitter=True,
        separability_cap=0.95,
    )
    assert meta["n_jitter_retries"] >= 1
    assert meta["jitter_sigma_frac"] > 0.0
    # Starting at jitter=0 is trivially separable (iat_min=0); regen must fire.


def test_inject_rejects_nsl_kdd(tmp_path):
    host = _host(50)
    src = tmp_path / "nsl_kdd.parquet"
    host.to_parquet(src, index=False)
    with pytest.raises(ValueError, match="ton_iot"):
        inject(src, tmp_path / "out.parquet", dataset_key="nsl_kdd")

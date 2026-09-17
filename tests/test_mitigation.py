from __future__ import annotations

import numpy as np
import pytest

from src.mitigation.network_sim import (
    D_CROSSOVER,
    E_ELEC,
    EPS_FS,
    EPS_MP,
    NODE_COUNTS,
    NetworkTopology,
    hop_energy,
    inject_dataset_attacks,
    rx_energy,
    tx_energy,
)
from src.mitigation.routing import evaluate_path, mitigate, route_fitness, search_route
from src.mitigation.zero_delay_penalty import zero_delay_penalty


def test_radio_model_and_crossover():
    assert D_CROSSOVER == pytest.approx(np.sqrt(EPS_FS / EPS_MP))
    k = 1000.0
    d_short = 0.5 * D_CROSSOVER
    d_long = 2.0 * D_CROSSOVER
    assert tx_energy(k, d_short) == pytest.approx(k * E_ELEC + k * EPS_FS * d_short ** 2)
    assert tx_energy(k, d_long) == pytest.approx(k * E_ELEC + k * EPS_MP * d_long ** 4)
    assert rx_energy(k) == pytest.approx(k * E_ELEC)
    assert hop_energy(k, d_short) == pytest.approx(tx_energy(k, d_short) + rx_energy(k))


def test_rgg_connected_and_attributes(tmp_path):
    topo = NetworkTopology.generate(50, seed=42)
    assert topo.n == 50
    assert topo.xy.shape == (50, 2)
    assert np.allclose(topo.initial_energy, 0.5)
    assert topo.residual_energy.shape == (50,)
    assert topo.is_compromised.dtype == bool
    assert topo.edge_index.shape[0] == 2
    assert topo.distance.size == topo.edge_index.shape[1]
    path = topo.decode_path(np.ones(50), src=0, dst=49)
    assert path[0] == 0 and path[-1] == 49
    p = topo.save(tmp_path / "n50_seed42.npz")
    loaded = NetworkTopology.load(p)
    assert loaded.n == 50
    assert np.allclose(loaded.xy, topo.xy)


def test_attack_injection_proportional():
    topo = NetworkTopology.generate(40, seed=0)
    rng = np.random.default_rng(0)
    X = rng.random((200, 20))
    X[:, 3] = rng.exponential(1.0, size=200)
    y = np.zeros(200, dtype=np.int64)
    y[:80] = 1
    inject_dataset_attacks(topo, X, y, dataset_key="nsl_kdd")
    assert topo.is_compromised.any()
    assert topo.traffic_load.sum() > 0
    with pytest.raises(ValueError):
        inject_dataset_attacks(topo, X, y, dataset_key="not_a_dataset")


def test_zero_delay_penalty_three_terms_and_legacy():
    assert 0 <= zero_delay_penalty(np.array([0.0, 1e-9, 0.5])) <= 1
    path = [0, 1, 2, 3]
    low = zero_delay_penalty(
        path,
        {
            "path_latency_ms": 1.0,
            "hop_count": 3,
            "max_hops": 20,
            "time_to_first_reroute_ms": 0.1,
            "observed_attack_iat_min_ms": 5.0,
        },
        latency_budget_ms=10.0,
    )
    high = zero_delay_penalty(
        path,
        {
            "path_latency_ms": 50.0,
            "hop_count": 19,
            "max_hops": 20,
            "time_to_first_reroute_ms": 50.0,
            "observed_attack_iat_min_ms": 0.0,
        },
        latency_budget_ms=10.0,
    )
    assert 0 <= low < high
    p1 = (50.0 - 10.0) / 10.0
    p2 = 19 / 20
    p3 = 1.0
    assert high == pytest.approx(0.4 * p1 + 0.3 * p2 + 0.3 * p3)


def test_route_fitness_and_search():
    mr, e, z = 0.8, 0.1, 0.2
    assert route_fitness(mr, e, z, w1=0.25, w2=0.25) == pytest.approx(0.8 - 0.025 - 0.05)
    topo = NetworkTopology.generate(30, seed=1)
    rng = np.random.default_rng(1)
    X = rng.random((80, 20))
    y = rng.integers(0, 2, size=80)
    inject_dataset_attacks(topo, X, y, dataset_key="unsw_nb15")
    out = search_route(topo, population=8, iterations=5, seed=2, search_weights=True, iat_min_ms=0.0)
    assert 0.1 - 1e-9 <= out["w1"] <= 0.5 + 1e-9
    assert 0.1 - 1e-9 <= out["w2"] <= 0.5 + 1e-9
    assert out["path"][0] == 0
    ev = evaluate_path(topo, out["path"], w1=out["w1"], w2=out["w2"])
    assert "mitigation_rate" in ev
    m = mitigate(n_nodes=8, population=6, iterations=4, seed=1)
    assert "mitigation_rate" in m
    assert NODE_COUNTS == (50, 100, 150, 200, 250)

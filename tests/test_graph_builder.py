"""Hand-checked cosine k-NN window graphs."""

from __future__ import annotations

import numpy as np
import torch

from src.data.graph_builder import build_window_graph, default_stride
from src.utils.guards import SOURCE_DATASET


def test_five_node_hand_computed_edges():
    # Unit vectors in 20-D (pad with zeros). Cosine = dot product.
    X = np.zeros((5, 20), dtype=np.float32)
    X[0, 0] = 1.0
    X[1, 0] = 1.0  # cos(0,1) = 1.0
    X[2, 1] = 1.0  # cos(0,2) = 0.0  -> prune at tau=0.4
    X[3, 0] = 0.8
    X[3, 1] = 0.6  # ||v||=1, cos(0,3)=0.8
    X[4, 2] = 1.0  # orthogonal to 0
    y = np.array([0, 0, 1, 0, 1])
    ids = np.arange(5)
    data, diag = build_window_graph(X, y, ids, k=2, tau=0.4)

    assert data.x.dtype == torch.float32
    assert data.edge_index.dtype == torch.int64
    assert data.x.shape == (5, 20)
    assert data.edge_attr.ndim == 2 and data.edge_attr.shape[1] == 1

    edges = set(zip(data.edge_index[0].tolist(), data.edge_index[1].tolist()))
    assert (0, 0) not in edges
    # 0--1 (cos=1) and 0--3 (cos=0.8) survive tau; 0--2 (0.0) is pruned
    assert (0, 1) in edges and (1, 0) in edges
    assert (0, 3) in edges and (3, 0) in edges
    assert (0, 2) not in edges
    # no isolated nodes (restored if needed)
    src = data.edge_index[0].tolist()
    touched = set(src) | set(data.edge_index[1].tolist())
    assert set(range(5)).issubset(touched)
    assert diag.n_restored >= 0
    w = dict(zip(edges, data.edge_attr.view(-1).tolist()))
    assert w[(0, 1)] > 0.99
    assert w[(0, 3)] > 0.79


def test_tau_prune_and_dtypes():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, 20)).astype(np.float32)
    y = np.zeros(8, dtype=np.int64)
    data, _ = build_window_graph(X, y, np.arange(8), k=3, tau=0.99)
    assert data.x.dtype == torch.float32
    assert data.edge_index.dtype == torch.int64
    deg = torch.bincount(data.edge_index[0], minlength=8)
    assert int((deg == 0).sum()) == 0


def test_train_stride_overlap_not_used_on_zeroshot():
    assert default_stride(SOURCE_DATASET, 100) == 50
    assert default_stride("ton_iot", 100) == 100
    assert default_stride("cicids2017", 100) == 100

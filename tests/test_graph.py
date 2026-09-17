from __future__ import annotations

import torch

from src.data.graph import build_flow_similarity_graph


def test_knn_graph_shapes_and_prune():
    torch.manual_seed(0)
    x = torch.randn(2, 10, 8)
    adj, mask = build_flow_similarity_graph(x, k=3, tau=0.4)
    assert adj.shape == (2, 10, 10)
    assert mask.shape == adj.shape
    assert adj.dtype == torch.float32
    # self-loops present
    assert torch.diagonal(adj, dim1=1, dim2=2).eq(1).all()

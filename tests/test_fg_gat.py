"""FG-GAT: GATv2, residuals, edge features, incoming-attention entropy."""

from __future__ import annotations

import torch

from src.models.fg_gat import FGGAT, attention_entropy


def _toy_graph(n: int = 8, f: int = 20, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, f, generator=g)
    # ring + a few extra edges
    src = torch.arange(n)
    dst = (src + 1) % n
    extra_s = torch.randint(0, n, (n,), generator=g)
    extra_d = torch.randint(0, n, (n,), generator=g)
    edge_index = torch.stack([torch.cat([src, extra_s]), torch.cat([dst, extra_d])], dim=0)
    sim = torch.rand(edge_index.size(1), 1, generator=g)
    return x, edge_index, sim


def test_sparse_shapes_and_attention_graph():
    x, ei, ea = _toy_graph()
    model = FGGAT(in_dim=20, hidden_dim=64, heads=8, n_layers=2, dropout=0.0, use_residual=True)
    h, attn = model(x, ei, ea, return_attention=True)
    assert h.shape == (8, 64)
    assert isinstance(attn, list) and len(attn) == 2
    ei1, alpha = attn[0]
    assert ei1.shape[0] == 2
    assert alpha.shape[0] == ei1.shape[1]
    assert alpha.requires_grad  # not detached in forward
    h.sum().backward()
    assert model.proj.weight.grad is not None


def test_multiscale_concat():
    x, ei, ea = _toy_graph()
    model = FGGAT(in_dim=20, hidden_dim=64, heads=8, n_layers=2, dropout=0.0)
    h, _ = model(x, ei, ea, return_multiscale=True)
    assert h.shape == (8, 128)


def test_attention_entropy_is_incoming_not_global():
    # Two nodes, two edges into node 0 with masses (0.9, 0.1); node 1 unused.
    edge_index = torch.tensor([[1, 2], [0, 0]], dtype=torch.long)
    peaked = torch.tensor([[0.9], [0.1]])
    uniform = torch.tensor([[0.5], [0.5]])
    h_peak = attention_entropy([(edge_index, peaked)])
    h_uni = attention_entropy([(edge_index, uniform)])
    assert h_peak < h_uni
    # Global softmax would mix in other nodes; incoming entropy of a 2-edge
    # peaked distribution is well below ln(2).
    assert float(h_peak) < 0.5


def test_dense_window_compat_shapes():
    model = FGGAT(in_dim=20, hidden=64, heads=8, layers=2, dropout=0.0)
    x = torch.randn(2, 6, 20)
    h, alpha = model(x)
    assert h.shape == (2, 6, 64)
    assert alpha.shape[0] == 2 and alpha.shape[1] == 6 and alpha.shape[2] == 6
    assert alpha.shape[-1] == 8


def test_edge_attr_changes_output():
    x, ei, _ = _toy_graph(n=6)
    model = FGGAT(in_dim=20, hidden_dim=64, heads=8, n_layers=2, dropout=0.0)
    model.eval()
    ea1 = torch.ones(ei.size(1), 1)
    ea2 = torch.linspace(0.1, 1.0, ei.size(1)).unsqueeze(-1)
    with torch.no_grad():
        h1, _ = model(x, ei, ea1)
        h2, _ = model(x, ei, ea2)
    assert not torch.allclose(h1, h2)

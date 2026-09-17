"""EA-SCAF: diverse AEs, top-k sparse attn, Work 2 Eq. (3) fusion."""

from __future__ import annotations

import torch

from src.models.ea_scaf import Autoencoder, EASCAF, EnsembleAutoencoder, SparseAttention


def test_autoencoder_recon_latent_and_batch1_bn():
    ae = Autoencoder(in_dim=20, latent_dim=32)
    x = torch.randn(1, 20)
    recon, z = ae(x)
    assert recon.shape == (1, 20)
    assert z.shape == (1, 32)
    assert recon.dtype == torch.float32


def test_ensemble_three_latents_and_aux_loss():
    ens = EnsembleAutoencoder(in_dim=20)
    x = torch.randn(4, 20)
    zs, loss = ens(x)
    assert len(zs) == 3
    assert all(z.shape == (4, 32) for z in zs)
    assert loss.ndim == 0 and loss.requires_grad
    loss.backward()


def test_sparse_attention_uses_topk_not_full_boolean_mask():
    sa = SparseAttention(dim=96, n_tokens=8, k=8)
    x = torch.randn(3, 96)
    y = sa(x)
    assert y.shape == (3, 96)


def test_eascaf_eq3_shapes_and_in_dim_ablation():
    for in_dim in (20, 128):
        model = EASCAF(in_dim=in_dim)
        x = torch.randn(5, in_dim)
        fused, rec_loss, aux = model(x)
        assert fused.shape == (5, 96)
        assert rec_loss.ndim == 0
        assert aux["shallow"].shape == (5, 96)
        assert aux["deep"].shape == (5, 96)
        # Eq. (3): fused == cross * sparse
        assert torch.allclose(fused, aux["cross"] * aux["sparse"], atol=1e-5)


def test_eascaf_batch1_mps_safe():
    model = EASCAF(in_dim=20)
    fused, rec_loss, _ = model(torch.randn(1, 20))
    assert fused.shape == (1, 96)
    rec_loss.backward()

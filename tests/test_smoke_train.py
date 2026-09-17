"""Tiny end-to-end fit: one epoch, protocol guards stay on."""

from __future__ import annotations

from src.config import GRADConfig
from src.train.trainer import GRADTrainer


def test_one_epoch_smoke(tmp_path):
    cfg = GRADConfig()
    cfg.project_root = tmp_path
    cfg.phase2.reservoir_size = 16
    cfg.train.epochs = 1
    cfg.train.patience = 1
    cfg.train.batch_size = 128
    cfg.graph.window = 20
    cfg.graph.k = 4
    cfg.resolve()
    trainer = GRADTrainer(cfg)
    report = trainer.fit(max_epochs=1, n_source=80, n_unsw=40, n_zero_shot=40)
    assert "nsl_kdd_test" in report
    assert "accuracy" in report["nsl_kdd_test"]
    assert set(report["transfer"]).issuperset({"unsw_nb15", "ton_iot"})

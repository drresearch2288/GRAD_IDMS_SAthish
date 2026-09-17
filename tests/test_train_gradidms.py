from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from scripts.train_gradidms import (
    ablation_flags,
    beta_warmup,
    composite_selection_score,
    parse_args,
    run_training,
)
from src.config import GRADConfig
from src.data.datamodule import GradIDMSDataModule
from src.utils.device import get_device


def test_parse_cli_defaults():
    args = parse_args(["--fold", "2", "--ablation", "no_domain", "--epochs", "3"])
    assert args.fold == 2
    assert args.ablation == "no_domain"
    assert args.epochs == 3
    assert args.batch_size == 256
    assert args.patience == 10


def test_ablation_flags():
    assert ablation_flags("none")["use_graph"] is True
    assert ablation_flags("no_graph")["use_graph"] is False
    assert ablation_flags("no_domain")["use_domain"] is False
    assert ablation_flags("no_adv")["use_adv"] is False
    assert ablation_flags("smote")["use_ffl"] is False


def test_beta_warmup_and_composite_score():
    assert beta_warmup(0) == 0.0
    assert abs(beta_warmup(5) - 0.25) < 1e-9
    assert abs(beta_warmup(10) - 0.5) < 1e-9
    assert abs(beta_warmup(99) - 0.5) < 1e-9
    assert abs(composite_selection_score(1.0, 0.5) - 1.0) < 1e-9
    assert abs(composite_selection_score(1.0, 1.0) - 0.6) < 1e-9
    assert abs(composite_selection_score(1.0, 0.0) - 0.6) < 1e-9


def _tiny_arrays(n: int = 80, seed: int = 0):
    rng = np.random.default_rng(seed)
    out = {}
    for i, key in enumerate(
        ["nsl_kdd", "unsw_nb15", "ton_iot", "bot_iot", "cicids2017", "apa_ddos"]
    ):
        y = np.tile(np.arange(5), n // 5 + 1)[:n]
        rng.shuffle(y)
        X = rng.normal(size=(n, 20)).astype(np.float32)
        X = (X - X.min()) / (X.max() - X.min() + 1e-8)
        out[key] = {
            "X": X,
            "y_multi": y.astype(np.int64),
            "y_bin": (y != 0).astype(np.int64),
            "flow_id": np.arange(n, dtype=np.int64),
        }
    return out


def test_one_epoch_smoke_and_resume(tmp_path: Path):
    cfg = GRADConfig.tiny()
    cfg.data_processed = tmp_path
    cfg.train.batch_size = 4
    cfg.phase2.pgd_steps = 1
    arrays = _tiny_arrays()
    dm = GradIDMSDataModule(
        cfg, arrays=arrays, splits_path=tmp_path / "nsl_kdd_splits.npz", overwrite_splits=True,
    )
    out_dir = tmp_path / "models"
    args = parse_args(
        [
            "--fold", "0",
            "--epochs", "1",
            "--batch-size", "4",
            "--patience", "2",
            "--out", str(out_dir),
            "--limit-batches", "1",
            "--ablation", "no_adv",
            "--config", str(tmp_path / "missing.yaml"),
        ]
    )
    device = torch.device("cpu")
    summary = run_training(args, cfg=cfg, dm=dm, device=device)
    ckpt = Path(summary["ckpt_path"])
    assert ckpt.exists()
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    for key in ("model", "optimizer", "scheduler", "epoch", "rng", "metrics_history"):
        assert key in blob
    log = Path(summary["log_path"])
    assert log.exists()
    rec = json.loads(log.read_text().strip().splitlines()[-1])
    assert "source_val_macro_f1" in rec
    assert "domain_disc_accuracy" in rec
    assert "lambda" in rec and "beta" in rec
    assert "mps_allocated_bytes" in rec
    assert rec["beta"] == 0.0

    args.resume = True
    args.epochs = 2
    summary2 = run_training(args, cfg=cfg, dm=dm, device=device)
    assert summary2["epochs_ran"] >= 1
    _ = get_device()

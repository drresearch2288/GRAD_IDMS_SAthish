# GRAD-IDMS

Graph-Attention, Domain-Generalizable, Adversarially-Robust and eXplainable
Intrusion Detection and Mitigation System (Work 3).

## Protocol

| Dataset | Role | Gradients |
| --- | --- | --- |
| NSL-KDD | Train + 70/15/15 split | Task + domain + graph + Phase II adversarial |
| UNSW-NB15 | Validation / unlabeled DANN | Domain loss only |
| ToN-IoT, BoT-IoT, CICIDS2017, ACK/PUSH-ACK | Zero-shot test | Never |

Raw parquets live in `data/raw/<dataset_key>.parquet`. Source-only z-score is fit on the NSL-KDD **train** split.

## Hardware

PyTorch **MPS** on Apple Silicon via `src.utils.device.get_device()`. CUDA is never selected. Tensors are **float32**.

## Setup

```bash
make setup
# or
python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

## Commands

```bash
make test
make lint
make train
python scripts/preprocess.py
python scripts/train.py --tiny
```

Composite loss: `L = FFL + λ(t) L_domain + β L_adv + γ L_graph_reg`
with `λ(t) = 2/(1+e^{-10t})-1`, `β=0.5`, `γ=0.1`.

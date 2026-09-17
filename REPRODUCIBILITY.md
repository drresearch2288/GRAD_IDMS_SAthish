# Reproducibility Guide for GRAD-IDMS

This document provides exact, deterministic instructions to reproduce every benchmark, table, figure, and theorem bound presented in the GRAD-IDMS paper from a clean checkout.

---

## 1. System & Hardware Requirements

| Parameter | Specification | Notes |
| :--- | :--- | :--- |
| **Primary Platform** | Apple Silicon (M1 / M2 / M3 / M4) with MPS | macOS 14+ |
| **Alternative Platform** | Linux x86_64 with NVIDIA GPU (CUDA 12.0+) | Ubuntu 22.04 LTS / Debian 12 |
| **RAM** | 16 GB minimum (32 GB recommended) | Memory-mapped data loading |
| **Storage Requirement** | ~12 GB free disk space | Raw datasets, graph caches, checkpoints, reports |
| **Python Version** | Python 3.10, 3.11, or 3.12 | Managed via virtual environment |

---

## 2. Environment Setup

```bash
# 1. Clone repository and navigate to root
git clone <repo-url> GRAD-IDMS
cd GRAD-IDMS

# 2. Initialize and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Verify system environment and accelerator availability
python scripts/verify_env.py
```

---

## 3. Reproduction Modes

The reproduction engine `reproduce_all.py` coordinates 19 discrete, resumable pipeline stages.

### Mode A: Full Publication Reproduction (~4.5 Hours)
Reproduces all 5-fold cross-validation results across all 6 benchmark datasets, evaluates all 10 baseline comparisons, runs 1,500 mitigation simulations, and compiles all 10 tables and 19 figures:
```bash
python reproduce_all.py
```

### Mode B: Fast Smoke Test Path (< 25 Minutes)
Runs a 10% data sample on 1 cross-validation fold for CI and rapid end-to-end sanity verification:
```bash
python reproduce_all.py --quick
```

### Mode C: Stage-Specific or Resumed Execution
```bash
# Dry run to inspect stage plan and estimated timings without executing:
python reproduce_all.py --dry-run

# Resume pipeline from a previously interrupted run:
python reproduce_all.py --resume

# Run only Table and Figure generation:
python reproduce_all.py --stages tables,figures

# Resume from a specific stage to the end:
python reproduce_all.py --from mitigation
```

---

## 4. Pipeline Stages & Execution Commands

| Stage # | Stage Name | Output Artifacts | Exact Standalone Command |
| :---: | :--- | :--- | :--- |
| **1** | `verify_environment` | Hardware & Package Audit | `python scripts/verify_env.py` |
| **2** | `download` | `data/raw/` | `python scripts/download_all.py` |
| **3** | `build_data` | `data/processed/` | `python scripts/build_dataset.py` |
| **4** | `build_graphs` | `data/graphs/` | `python scripts/build_graphs.py` |
| **5** | `zero_delay` | `data/zero_delay/` | `python -c "from src.data.zero_delay import synthesize_zero_delay_dataset; synthesize_zero_delay_dataset()"` |
| **6** | `eda` | `reports/figures/` (EDA figures) | `python eda/eda_domain_shift.py` |
| **7** | `tune` | `results/metrics/optimizer_tuning.json` | `python scripts/tune_mavmoa.py` |
| **8** | `train` | `results/models/gradidms/` | `python scripts/train_gradidms.py --epochs 10 --folds 5` |
| **9** | `baselines` | `results/models/` (Baselines) | `python scripts/run_all_baselines.py` |
| **10** | `evaluate` | `results/evaluation_results.json` | `python scripts/evaluate_all.py` |
| **11** | `robustness` | `results/metrics/robustness.json` | `python scripts/run_robustness.py` |
| **12** | `mitigation` | `results/metrics/mitigation.json` | `python scripts/run_mitigation.py` |
| **13** | `xai` | `results/metrics/xai_eval.json` | `python scripts/run_xai_eval.py` |
| **14** | `compress` | `results/models/compressed_int8.pt` | `python scripts/compress_model.py` |
| **15** | `edge` | `results/metrics/edge_benchmark.json` | `python scripts/benchmark_edge.py` |
| **16** | `theorems` | `results/metrics/theorem_verification.json` | `python scripts/verify_theorems.py` |
| **17** | `tables` | `results/tables/` (10 LaTeX & 10 PNG) | `python scripts/make_tables.py && python scripts/render_table_images.py` |
| **18** | `figures` | `reports/figures/` (19 Figures) | `python figures/fig01_pipeline.py` ... `fig19_radar.py` |
| **19** | `report` | `results/RESULTS_SUMMARY.md` | `python reproduce_all.py --stages report` |

---

## 5. Headline Paper Numbers & Mapping Matrix

Every major experimental metric in the paper is validated automatically against pre-defined thresholds:

| Paper Result / Metric | Target Value | Verification Tolerance | Generating Script & Stage |
| :--- | :---: | :---: | :--- |
| **Same-Domain Accuracy** | **88.01%** | $\pm 0.5\,\text{pts}$ | `scripts/evaluate_all.py` (`evaluate`) |
| **Same-Domain Macro-F1** | **0.4110** | $\pm 0.010$ | `scripts/evaluate_all.py` (`evaluate`) |
| **Same-Domain FPR** | **1.85%** | $\le 2.0\%$ | `scripts/evaluate_all.py` (`evaluate`) |
| **Mean Cross-Domain Gen. Gap** | **12.40%** | $\le 15.0\%$ | `scripts/evaluate_all.py` (`evaluate`) |
| **PGD Robustness ($\epsilon=0.03$)** | **74.80%** | $\pm 0.5\,\text{pts}$ | `scripts/run_robustness.py` (`robustness`) |
| **Mitigation Success Rate** | **95.80%** | $\pm 0.5\,\text{pts}$ | `scripts/run_mitigation.py` (`mitigation`) |
| **Zero-Delay Mitigation Rate** | **94.80%** | $\pm 0.5\,\text{pts}$ | `scripts/run_mitigation.py` (`mitigation`) |
| **Compressed INT8 Size** | **0.13 MB** | $\le 0.15\,\text{MB}$ | `scripts/compress_model.py` (`compress`) |
| **Edge Inference Latency** | **1.25 ms** | $\le 1.50\,\text{ms}$ | `scripts/benchmark_edge.py` (`edge`) |
| **Explainability Agreement ($\rho$)** | **0.947** | $\ge 0.850$ | `scripts/run_xai_eval.py` (`xai`) |

---

## 6. Audit Logging & Checksum Tracking

During pipeline execution:
- Execution logs are streamed live and recorded into `results/logs/reproduce_<timestamp>.log`.
- All environment parameters, git hashes, package versions, input checksums (SHA-256), and output checksums are saved in `results/reproduction_manifest.json`.
- A failure in any stage prints the last 50 error log lines and outputs the exact command to re-run only that specific failed stage.

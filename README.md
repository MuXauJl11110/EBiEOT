<div align="center">

# Inverse Entropic Optimal Transport Solves Semi-supervised Learning via Data Likelihood Maximization

[Mikhail Persiianov](https://scholar.google.com/citations?user=YXX3VMQAAAAJ&hl=en),
[Arip Asadulaev](https://scholar.google.com/citations?user=wcdrgdYAAAAJ&hl=en),
[Nikita Andreev](https://scholar.google.com/citations?user=D0a1XNsAAAAJ&hl=en),
[Nikita Starodubcev](https://scholar.google.com/citations?user=o6pRm_gAAAAJ&hl=en), 
[Dmitry Baranchuk](https://scholar.google.com/citations?user=NiPmk8oAAAAJ&hl=en),
[Anastasis Kratsios](https://scholar.google.com/citations?user=9D-bHFgAAAAJ&hl=en),
[Evgeny Burnaev](https://scholar.google.com/citations?user=pCRdcOwAAAAJ&hl=en),
[Alexander Korotin](https://scholar.google.com/citations?user=1rIIvjAAAAAJ&hl=en)

[![arXiv Paper](https://img.shields.io/badge/arXiv-2410.02628-b31b1b)](https://arxiv.org/abs/2410.02628)
[![OpenReview Paper](https://img.shields.io/badge/OpenReview-PDF-8c1b13)](https://openreview.net/forum?id=0p617sK4Z4)
[![GitHub](https://img.shields.io/github/stars/MuXauJl11110/EBiEOT?style=social)](https://github.com/MuXauJl11110/EBiEOT)
![GitHub License](https://img.shields.io/github/license/MuXauJl11110/EBiEOT?style=flat&label=License)

</div>

This repository contains the official implementation of the paper *"Inverse Entropic Optimal Transport Solves Semi-supervised Learning via Data Likelihood Maximization"*. The framework utilizes Energy-Based Inverse Entropic Optimal Transport (**EBiEOT**) to handle semi-supervised learning setups by directly maximizing data likelihood.

---

## 📌 TL;DR

This work bridges **Inverse Entropic Optimal Transport (IEOT)** and **Semi-supervised Learning (SSL)**. By formulating SSL as an optimization problem over the data likelihood, our proposed methods (**EBiEOT-GMM** and **EBiEOT-NN**) parameterize cost functions and data potentials using neural networks and Gaussian Mixture Models. This approach effectively unifies paired and unpaired data streams without requiring ad-hoc regularizers or precomputed transport maps.

---

## 📦 Project Structure

```bash
.
├── conf/                  # Hydra configuration files
│   ├── baseline/          # Baselines model & training presets
│   ├── dataset/           # synthetic/, alae/, colored_mnist/, mnist_classification/, weather/
│   ├── ebieot/            # EBiEOT model, cost, and potential defaults
│   └── train/             # Core training loops and optimizers
│
├── notebooks/             # Active Jupyter tutorials & evaluation pipelines
│   ├── ALAE/              # High-dimensional latent space experiments (FFHQ)
│   ├── colored_mnist/     # Image translation tasks
│   ├── classification/    # MNIST energy classification
│   ├── swiss_roll/        # Toy/synthetic geometry evaluations
│   └── weather/           # TabRED weather domain translation
│
├── scripts/               # Execution entry points (CLI trainers & sweep automation)
│   ├── train.py           # Main EBiEOT training (Swiss-roll, MNIST classification)
│   ├── train_baseline.py  # Baseline training entry point
│   ├── train_colored_mnist.py  # Colored MNIST (MLP / CNN EBiEOT-NN)
│   └── run_*.py           # Papermill grids, Optuna, MNIST energy grid, utilities
│
├── src/                   # Core library source code
│   ├── ebieot/            # EBiEOT-GMM, EBiEOT-NN, classification_based.py
│   ├── baselines/         # Inverse-OT baseline architectures (CGAN, CNF, etc.)
│   ├── networks/          # Building blocks (MLPs, GAN components, Normalizing Flows)
│   └── utils/             # Categorized utility modules
│
├── tests/                 # pytest suite
├── data/                  # MNIST / colored-MNIST cache (auto-download at runtime)
├── logs/                  # Hydra training outputs
├── checkpoints/           # Saved runs and notebook artifacts
└── papermill-notebooks/   # Outputs from sweep scripts (created at runtime)

```

Figures from training and notebooks are written under `logs/` or notebook output directories, not a separate top-level `plots/` folder.

---

## 📥 Installation & Dependencies

This project manages environment dependencies with [uv](https://github.com/astral-sh/uv). **Python `>=3.10,<3.13`** is required ([pyproject.toml](pyproject.toml)). Pinned dependencies live in `pyproject.toml` and `uv.lock`; [requirements.txt](requirements.txt) only points to `uv sync`.

```bash
# Clone the repository
git clone https://github.com/MuXauJl11110/EBiEOT.git
cd EBiEOT

# Sync the environment and dependencies
uv sync
```

> **Working directory:** Run all commands from the **repository root** — the directory that contains `conf/`, `scripts/`, and `pyproject.toml`.

To run scripts or tests within the managed environment, prepend commands with `uv run`:

```bash
uv run python scripts/train.py --help

# Tests require the dev dependency group (pytest)
uv sync --group dev
uv run pytest
```

---

## 📂 Data Prerequisites

| Dataset | Setup |
| --- | --- |
| Swiss-roll | Synthetic — no download |
| MNIST classification | Auto-download to `data/mnist` ([mnist_classification.py](src/utils/datasets/mnist_classification.py)) |
| Colored MNIST | torchvision download under `data/` |
| FFHQ ALAE latents | Manual: place `latents.npy`, `gender.npy`, `age.npy` under `datasets/FFHQ/` ([conf/dataset/alae/ffhq_latents.yaml](conf/dataset/alae/ffhq_latents.yaml)) |
| Weather (TabRED) | External files under `../tabred/kal/weather` — see [notebooks/weather/README.md](notebooks/weather/README.md) |

For Comet ML logging (`logger=comet`) or weather notebook training cells, set `COMET_API_KEY` (or log in via the Comet CLI) before running.

---

## 🏋️ Training & CLI Usage

All experiments are powered by **Hydra**. Configuration files reside under `conf/`. All commands below assume the repository root as the current working directory.

> **Experiment naming:** Hydra presets use the prefix **`egeot_`** for EBiEOT-NN and MNIST classification (historical naming) and **`gmm_`** for EBiEOT-GMM. Related Swiss-roll variants include `egeot_swiss_roll_128`, `egeot_swiss_roll_16k`, and `gmm_swiss_roll_shared`.

### 1. Core EBiEOT Training (Swiss-Roll)

```bash
# Train EBiEOT Neural Network on Swiss-roll
uv run python scripts/train.py experiment=egeot_swiss_roll

# Train EBiEOT GMM with Comet ML logging enabled
uv run python scripts/train.py experiment=gmm_swiss_roll logger=comet

# Multi-run hyperparameter override example
uv run python scripts/train.py -m experiment=egeot_swiss_roll train.optimizer.paired.lr=1e-3,5e-4

# Debug run (writes logs to logs/train/runs/...)
uv run python scripts/train.py debug=default
```

> **Note:** You can override any configuration value dynamically from the command line (e.g., `train.steps_to=500`).

### 2. Inverse-OT Baselines (Swiss-Roll)

Baseline architectures are located under `src/baselines/` and configured via `conf/baseline/`.

```bash
# Train Conditional GAN baseline
uv run python scripts/train_baseline.py experiment=baseline_cgan_swiss_roll_16k

# Train Semi-supervised Continuous Normalizing Flow baseline with Comet
uv run python scripts/train_baseline.py experiment=baseline_cnf_swiss_roll_semi logger=comet

# Fast debug smoke-test for baselines
uv run python scripts/train_baseline.py experiment=baseline_reg_swiss_roll_16k train.steps_to=10 debug=default
```

### 3. MNIST Energy Classification

Run semi-supervised label prediction on standard MNIST:

```bash
uv run python scripts/train.py experiment=egeot_classification_mnist train.epochs_max=2
uv run python scripts/run_mnist_energy_grid.py --run-name smoke --paired-grid 20 --unpaired-grid 0 --epochs-max 2 --cpu
```

### 4. Colored MNIST

```bash
uv run python scripts/train_colored_mnist.py experiment=egeot_colored_mnist train.steps_to=10
```

Other presets: `egeot_colored_mnist_cnn_vanilla`, `egeot_colored_mnist_cnn_unet`, `egeot_colored_mnist_cnn_nonlocal` (under `conf/experiment/`).

### 5. Automated Batches & Optuna Sweeps

Automate execution or launch hyperparameter optimization sweeps using Papermill and Optuna:

```bash
uv run python scripts/run_swiss_roll.py              # EBiEOT-GMM grid -> notebooks/swiss_roll/ebieot_gmm.ipynb
uv run python scripts/run_neural_swiss_roll.py       # EBiEOT-NN grid  -> notebooks/swiss_roll/ebieot_nn.ipynb
uv run python scripts/run_optuna_search_swiss_roll.py  # Hyperparameter optimization over GMM
uv run python scripts/run_baselines_swiss_roll.py    # Comprehensive baseline sweep
```

---

## 📔 Active Notebooks

Jupyter notebooks are located under `notebooks/`. They compose Hydra configs directly via the API.

| Notebook | Role / Experiment Context |
| --- | --- |
| `swiss_roll/ebieot_nn.ipynb` | Evaluates EBiEOT-NN on standard synthetic Swiss-roll configurations. |
| `swiss_roll/ebieot_gmm.ipynb` | Evaluates EBiEOT-GMM variants (independent vs. shared parameters). |
| `swiss_roll/swiss_roll_plot.ipynb` | Utility to load saved Hydra checkpoints and plot data marginals. |
| `ALAE/ebieot_gmm_alae.ipynb` | Latent-space EBiEOT-GMM applied to FFHQ human face latents. |
| `ALAE/ebieot_gmm_alae_eval.ipynb` | Quantitative ALAE evaluation and optional comparison with FSBM models. |
| `colored_mnist/ebieot_colored_mnist.ipynb` | Image translation tasks mapping digits across color domains. |
| `classification/ebieot_classification_mnist.ipynb` | Energy-based classification benchmarks on MNIST. |

**Weather (TabRED)** — EBiEOT-GMM only; EBiEOT-NN is not supported in these notebooks. See [notebooks/weather/README.md](notebooks/weather/README.md).

| Notebook | Method | Hydra experiment |
| --- | --- | --- |
| `weather.ipynb`, `weather-ebm.ipynb` | EBiEOT-GMM | `gmm_weather` |
| `weather-gan.ipynb` | cGAN baseline | `baseline_cgan_weather` |
| `weather-reg.ipynb` | Regression baseline | `baseline_reg_weather` |
| `weather-cnf.ipynb` | CNF baseline | `baseline_cnf_weather` |
| `weather-ugan.ipynb` | UGAN baseline | `baseline_ugan_weather` |

---

## 🗺️ Library Layout & Architecture

### `src/ebieot/` — Core Models

* **`ebieot_gmm.py`**: Implementation of `EbieotGmm` (EBiEOT-GMM dual framework with a learnable Log-Sum-Exp cost structure).
* **`ebieot_nn.py`**: Implementation of `EbieotNn` (EBiEOT-NN with parameterized neural costs/potentials and Langevin pseudo-sampling).
* **`classification_based.py`**: Energy-based semi-supervised MNIST classification.
* **`costs/`** & **`potentials/`**: Modular definitions for `MLPCost`, `MLPLSECost`, and generic multilayer perceptron potentials.
* **`sampling/`**: Langevin dynamics stepping modules and sample experience buffers.

### `src/utils/` — Canonical API Packages

To ensure development reproducibility and modularity, please import utilities using the standardized paths outlined below:

| Package | Purpose & Target Modules |
| --- | --- |
| **`core/`** | Global seed setups (`seed.py`) and rank-aware command-line utilities (`pylogger.py`). |
| **`experiment/`** | Hydra integration decorators (`hydra_utils.py`), Rich tree logging formatting (`rich_utils.py`). |
| **`training/`** | Optimization helpers, standard loss calculation pipelines, and streaming averages (`helpers.py`). |
| **`evaluation/`** | MMD and Sinkhorn divergence (`metrics.py`). |
| **`samplers/`** | Standardized training/evaluation data streaming hooks (`synthetic.py`, `data.py`, `discrete_ot.py`). |
| **`datasets/`** | Underlying loaders handling pairing, ground truth matching, and domain-specific tokens. |
| **`plotting/`** | Distribution visualization assets, transport pairs, and automated image grids. |

💡 **Import Cheat Sheet:**

```python
from src.utils.core.seed import set_seed
from src.utils.core.pylogger import RankedLogger
from src.utils.experiment.hydra_utils import extras
from src.utils.training.helpers import compute_loss, update_average, compute_metrics
from src.utils.evaluation.metrics import compute_mmd, compute_sinkhorn_divergence
from src.utils.samplers.synthetic import build_swiss_roll_samplers
```

---

## 📊 Inverse-OT Reference Baselines

The table below describes the default configuration behaviors of the baseline environments used for benchmarking against the Swiss-roll target dataset.

| Experiment Config YAML | Method Identifier | Default Sample Profile (`Paired X–Y` / `Unpaired X` / `Unpaired Y`) |
| --- | --- | --- |
| `baseline_cgan_swiss_roll_16k` | `cgan` | 16,000 / 16,000 / 0 |
| `baseline_ugan_swiss_roll_16k` | `ugan` | 16,000 / 16,000 / 16,000 |
| `baseline_reg_swiss_roll_16k` | `regression` | 16,000 / 0 / 1,024 |
| `baseline_cnf_swiss_roll_small` | `cnf` | 16,000 / 0 / 1,024 |
| `baseline_cnf_swiss_roll_semi` | `cnf_semi` | 128 / 1,024 / 1,024 *(utilizes in-memory OT optimization)* |

> ⚡ **Quick CPU Smoke Test:** To quickly verify baseline execution environments without straining local resources, append the following parameter overrides to your terminal call: `train.steps_to=2 dataset.P_XY_paired=64 dataset.paired_cache.enabled=false`.

---

## 🎓 Citation

If you use this codebase or find our methodology helpful in your research, please cite our work using the following format:

```bibtex
@inproceedings{
  persiianov2026inverse,
  title={Inverse Entropic Optimal Transport Solves Semi-supervised Learning via Data Likelihood Maximization},
  author={Mikhail Persiianov and Arip Asadulaev and Nikita Andreev and Nikita Starodubcev and Dmitry Baranchuk and Anastasis Kratsios and Evgeny Burnaev and Alexander Korotin},
  booktitle={Forty-third International Conference on Machine Learning},
  year={2026},
  url={https://openreview.net/forum?id=0p617sK4Z4}
}
```

## 🙏 Credits

- [GeomLoss](https://www.kernel-operations.io/geomloss/) - toolkit for computing metrics between measures;
- [optuna](https://optuna.org) - toolkit for hyperparameter search;
- [comet ML](https://www.comet.com) — experiment-tracking and visualization toolkit;
- [inkscape](https://inkscape.org/) — an excellent open-source editor for vector graphics;
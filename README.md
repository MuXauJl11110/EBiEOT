# Inverse Entropic Optimal Transport Solves Semi-supervised Learning via Data Likelihood Maximization

Research code for semi-supervised domain translation via a likelihood objective linked to inverse entropic optimal transport. The mathematical development is in `icml2026.tex`. Executable experiments are driven mainly from Jupyter notebooks; `src/` holds shared PyTorch modules and utilities.

## Setup

### Recommended: uv (local virtual environment)

[uv](https://docs.astral.sh/uv/) installs dependencies into a project-local `.venv` (gitignored), not your global Python.

1. Install uv (see [installation](https://docs.astral.sh/uv/installation/)).
2. From the repository root:

```bash
uv venv --allow-existing
uv sync --group dev
```

3. Use the environment for any command:

```bash
uv run python -c "import torch; print(torch.__version__)"
uv run jupyter lab   # if Jupyter is installed in that env
```

4. **Run unit tests** (GMMEOT / stability checks):

```bash
./scripts/run_tests_uv.sh
```

Equivalent manual steps: `uv venv --allow-existing`, then `uv sync --group dev`, then `uv run pytest tests/test_gmm_based.py -v`.

Locked versions live in `uv.lock`; `pyproject.toml` defines runtime and dev (`pytest`) dependencies.

### Alternative: pip and `requirements.txt`

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest tests/test_gmm_based.py -v
```

Note: `requirements.txt` lists duplicate `torch` / `torchvision` lines for different indexes; adjust for your platform (CPU vs CUDA) if needed.

### Python version

`pyproject.toml` currently specifies `requires-python = ">=3.10,<3.13"`. Use a matching interpreter when creating the venv.

### Notebooks

Notebooks under `notebooks/` usually add the repo root to `sys.path` (for example `sys.path.append("..")` when the notebook lives one level down). Run Jupyter from an environment where `torch` and the rest of the stack are installed, with the working directory consistent with that path setup.

Optional: experiments may import `comet_ml`; it is not listed in `pyproject.toml` / `requirements.txt`—install it in the same environment if you enable logging.

## Repository layout

| Path | Role |
|------|------|
| `src/` | Core library: models (`gmm_based`, energy-based, etc.), costs (`lse`, MLP), potentials, samplers, plotting, metrics, training helpers. |
| `configs/` | Pydantic-style configs for GMM-based and energy-based training (datasets, optimizers, costs, train loop). |
| `notebooks/` | End-to-end experiments: `swiss_roll/`, `ALAE/`, `weather/`, `colored_mnist/`, etc. |
| `tests/` | Pytest suite (e.g. GMMEOT weight simplex and loss scaling). |
| `scripts/` | Helper scripts; `run_tests_uv.sh` creates/syncs `.venv` via uv and runs tests. |
| `plots/` | Saved figures from experiments. |
| `checkpoints/` | Saved model checkpoints (often created at runtime). |
| `toy_baselines/` | Standalone baseline scripts (e.g. CNF, CGAN, regression). |
| `run_swiss_roll.py`, `run_optuna_search_swiss_roll.py`, `run_ebm.py` | Entry points for Swiss Roll / Optuna / energy-based flows. |
| `icml2026.tex` | Paper manuscript. |
| `pyproject.toml` / `uv.lock` | uv project metadata and lockfile. |
| `requirements.txt` | Legacy pip dependency list. |

## Legacy branch (`legacy/ebieot-gmm-alae`)

Frozen **pre-Hydra GMM / ALAE latent IOT** line (formerly branches `ALAE`, `ebieot-gmm-alae`). Use **`main`** for active development (`conf/`, `src/ebieot/`, Hydra notebooks). Check out this branch only when you need to reproduce an experiment from the old layout.

If you previously tracked `origin/ALAE` or `origin/ebieot-gmm-alae`, run `git fetch origin && git checkout legacy/ebieot-gmm-alae`.

### Checkout

```bash
git checkout legacy/ebieot-gmm-alae
uv sync --group dev
```

On **`main`**, the same uv workflow applies (`uv sync`).

Preservation tag for the pre-docs code tip: `legacy/alae`. Documented tip: tag `legacy/ebieot-gmm-alae` at this branch tip.

### Experiment map

| Experiment | `legacy/ebieot-gmm-alae` | `main` |
|------------|--------------------------|--------|
| FFHQ / ALAE GMM | `notebooks/ALAE/GMMEOT_ALAE*.ipynb`, `LightSB_alae.ipynb` (wandb) | `notebooks/ALAE/ebieot_gmm_alae.ipynb`, `ebieot_gmm_alae_eval.ipynb` (Hydra + Comet) |
| AFHQ | `notebooks/GMMEOT_AFHQ.ipynb` | `conf/experiment/gmm_afhq.yaml` + Hydra layout (no flat `GMMEOT_AFHQ.ipynb`) |
| Colored MNIST | `notebooks/colored_mnist/EgEOT_colored_mnist_plotting.ipynb` | `notebooks/colored_mnist/ebieot_colored_mnist.ipynb` |
| Swiss roll GMM | `notebooks/swiss_roll/GMMEOT_swiss_roll*.ipynb` | `notebooks/swiss_roll/ebieot_gmm.ipynb` |
| Swiss roll EgEOT | `notebooks/swiss_roll/EgEOT_swiss_roll*.ipynb` | `notebooks/swiss_roll/ebieot_nn.ipynb` |
| Weather + baselines | `notebooks/weather/*.ipynb` (older copies) | Same paths; Hydra via `conf/` |
| MNIST 2→3 / EBM anatomy | **Not on this branch** | **Not on `main`** (see `legacy/ebieot-nn`) |
| MNIST classification | **Not on this branch** | `notebooks/classification/` |
| Toy / Optuna runners | `run_swiss_roll.py`, `run_optuna_search_swiss_roll.py` | `scripts/run_*.py` |

### Layout vs `main`

| | `legacy/ebieot-gmm-alae` | `main` |
|--|--------------------------|--------|
| Config | `configs/` (Pydantic) | `conf/` (Hydra) |
| Models | `src/models/` (`gmm_based.py`, `light_sb.py`, `light_sbm.py`) | `src/ebieot/` |
| ALAE notebooks | `GMMEOT_ALAE_*.ipynb` | `ebieot_gmm_alae*.ipynb` |
| Tests | `tests/test_gmm_based.py` | Migrated under `main` layout |
| Logging | wandb in ALAE notebooks | Comet via Hydra `logger=comet` |

### Related branches

- **`legacy/ebieot-nn`** — pre-Hydra EgEOT / energy-based line (formerly `EBM`). See README on that branch.
- **`legacy/ebm-alae`** — merged EBM + ALAE snapshot. See README on that branch for the combined map.
- Tag **`legacy/alae`** — ALAE tip before this rename/docs commit.

### Verification

```bash
uv run pytest tests/test_gmm_based.py -q
```

Full notebook runs require datasets and long training and were not re-executed for this documentation. Re-run the notebook for your experiment after checkout if you rely on this branch.
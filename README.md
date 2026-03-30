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
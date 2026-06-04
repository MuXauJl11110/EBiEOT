# Inverse Entropic Optimal Transport Solves Semi-supervised Learning via Data Likelihood Maximization

**Active development is on [`main`](https://github.com/MuXauJl11110/EBiEOT/tree/main).**
See the [README on `main`](https://github.com/MuXauJl11110/EBiEOT/blob/main/README.md) for installation (uv/Hydra), datasets, CLI training in `scripts/`, and citations.

## Branch `legacy/ebieot-gmm-alae`

Frozen pre-Hydra **GMM / ALAE** layout (`configs/`, `src/models/`, notebooks).
Use this branch only to reproduce experiments from the old layout.
Former remotes: `ALAE`, `ebieot-gmm-alae` → `git fetch origin && git checkout legacy/ebieot-gmm-alae`.

Experiment maps and comparison with `main`: [LEGACY.md](LEGACY.md).

## How to run

Experiments are driven from Jupyter notebooks under `notebooks/`.
Supporting code lives in `src/` (models, costs, potentials, samplers, plotting) and `configs/` (Pydantic training presets).

## Setup

From the repository root ([uv](https://docs.astral.sh/uv/)):

```bash
uv venv --allow-existing
uv sync --group dev
./scripts/run_tests_uv.sh
```

Python `>=3.10,<3.13` per `pyproject.toml`. Locked versions are in `uv.lock`.

Notebooks under `notebooks/` usually add the repo root to `sys.path` (for example `sys.path.append("..")` one level down). Run Jupyter with `torch` installed and a working directory consistent with that path setup.

Optional: some notebooks import `comet_ml` (not in `pyproject.toml`)—install in the same environment if you enable logging.

Alternative: `pip install -r requirements.txt` then `pytest tests/test_gmm_based.py -v`.

## Layout

- `notebooks/` — ALAE, swiss_roll, weather, colored_mnist, AFHQ experiments
- `src/` — core library (`models/`, `costs/`, `potentials/`, `samplers/`, `utils/`, `plotting/`)
- `configs/` — Pydantic configs for GMM-based and energy-based training
- `tests/` — pytest (e.g. `test_gmm_based.py`)
- `scripts/` — `run_tests_uv.sh` and helpers
- `toy_baselines/` — standalone baseline scripts (CNF, CGAN, regression)
- `run_swiss_roll.py`, `run_optuna_search_swiss_roll.py`, `run_ebm.py` — Swiss roll / Optuna / energy-based entry points
- `checkpoints/`, `plots/` — runtime outputs

Preservation tags: `legacy/alae` (pre-docs ALAE tip), `legacy/ebieot-gmm-alae` (documented tip at this branch).

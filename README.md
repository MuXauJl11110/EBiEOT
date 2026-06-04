# Inverse Entropic Optimal Transport Solves Semi-supervised Learning via Data Likelihood Maximization

**Active development is on [`main`](https://github.com/MuXauJl11110/EBiEOT/tree/main).**
See the [README on `main`](https://github.com/MuXauJl11110/EBiEOT/blob/main/README.md) for installation (uv/Hydra), datasets, CLI training in `scripts/`, and citations.

## Branch `legacy/ebieot-nn`

Frozen pre-Hydra **EgEOT / energy-based IOT** layout (`configs/`, `src/models/`, flat notebooks).
Use this branch only to reproduce experiments from the old layout.
Former remote: `EBM` → `git fetch origin && git checkout legacy/ebieot-nn`.

Experiment maps and comparison with `main`: [LEGACY.md](LEGACY.md).

## How to run

The main code is in `.ipynb` files under `notebooks/`.
Supporting code is in `src/` (models, costs, potentials, samplers) and `configs/` (Pydantic presets).
Scripts: `run_ebm.py`, `run_swiss_roll.py`; toy baselines in `toy_baselines/`.

## Setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

On **`main`**, dependencies are managed with [uv](https://github.com/astral-sh/uv): see the [main README](https://github.com/MuXauJl11110/EBiEOT/blob/main/README.md).

Weather notebooks may use Comet (`COMET_API_KEY`).

## Layout

- `notebooks/` — weather, swiss roll, colored MNIST (flat and nested paths)
- `src/` — core library (`models/`, `costs/`, `potentials/`, `samplers/`, `utils/`, `plotting/`)
- `configs/` — Pydantic configs for GMM-based and energy-based training
- `mnist2to3/` — MNIST 2→3 (EBM anatomy; not on `main`)
- `toy_baselines/` — standalone baseline scripts
- `run_ebm.py`, `run_swiss_roll.py` — energy-based and Swiss roll entry points
- `checkpoints/`, `plots/` — runtime outputs

Preservation tags: `legacy/ebm` (pre-docs EBM tip), `legacy/ebieot-nn` (documented tip at this branch).

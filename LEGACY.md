# Legacy experiments (`ebieot-gmm-alae`)

Frozen **pre-Hydra GMM / ALAE latent IOT** line (formerly branch `ALAE`). Use **`main`** for active development (`conf/`, `src/ebieot/`, Hydra notebooks). Check out this branch only when you need to reproduce an experiment from the old layout.

If you previously tracked `origin/ALAE`, run `git fetch origin && git checkout ebieot-gmm-alae`.

## Checkout

```bash
git checkout ebieot-gmm-alae
uv sync --group dev
```

On **`main`**, the same uv workflow applies (`uv sync`).

Preservation tag for the pre-docs code tip: `legacy/alae`. Documented tip: `legacy/ebieot-gmm-alae`.

## Experiment map

| Experiment | `ebieot-gmm-alae` | `main` |
|------------|-------------------|--------|
| FFHQ / ALAE GMM | `notebooks/ALAE/GMMEOT_ALAE*.ipynb`, `LightSB_alae.ipynb` (wandb) | `notebooks/ALAE/ebieot_gmm_alae.ipynb`, `ebieot_gmm_alae_eval.ipynb` (Hydra + Comet) |
| AFHQ | `notebooks/GMMEOT_AFHQ.ipynb` | `conf/experiment/gmm_afhq.yaml` + Hydra layout (no flat `GMMEOT_AFHQ.ipynb`) |
| Colored MNIST | `notebooks/colored_mnist/EgEOT_colored_mnist_plotting.ipynb` | `notebooks/colored_mnist/ebieot_colored_mnist.ipynb` |
| Swiss roll GMM | `notebooks/swiss_roll/GMMEOT_swiss_roll*.ipynb` | `notebooks/swiss_roll/ebieot_gmm.ipynb` |
| Swiss roll EgEOT | `notebooks/swiss_roll/EgEOT_swiss_roll*.ipynb` | `notebooks/swiss_roll/ebieot_nn.ipynb` |
| Weather + baselines | `notebooks/weather/*.ipynb` (older copies) | Same paths; Hydra via `conf/` |
| MNIST 2→3 / EBM anatomy | **Not on this branch** | **Not on `main`** (see `legacy/ebieot-nn`) |
| MNIST classification | **Not on this branch** | `notebooks/classification/` |
| Toy / Optuna runners | `run_swiss_roll.py`, `run_optuna_search_swiss_roll.py` | `scripts/run_*.py` |

## Layout vs `main`

| | `ebieot-gmm-alae` | `main` |
|--|-------------------|--------|
| Config | `configs/` (Pydantic) | `conf/` (Hydra) |
| Models | `src/models/` (`gmm_based.py`, `light_sb.py`, `light_sbm.py`) | `src/ebieot/` |
| ALAE notebooks | `GMMEOT_ALAE_*.ipynb` | `ebieot_gmm_alae*.ipynb` |
| Tests | `tests/test_gmm_based.py` | Migrated under `main` layout |
| Logging | wandb in ALAE notebooks | Comet via Hydra `logger=comet` |

## Related branches

- **`legacy/ebieot-nn`** — pre-Hydra EgEOT / energy-based line (formerly `EBM`). See its `LEGACY.md`.
- **`legacy/ebm-alae`** — merged EBM + ALAE snapshot. See its `LEGACY.md` for the combined map.
- Tag **`legacy/alae`** — ALAE tip before this rename/docs commit.
- Tag **`legacy/ebieot-gmm-alae`** — this branch tip (includes this file).

## Verification

```bash
uv run pytest tests/test_gmm_based.py -q
```

Full notebook runs require datasets and long training and were not re-executed for this documentation. Re-run the notebook for your experiment after checkout if you rely on this branch.

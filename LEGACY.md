# Legacy experiments (`legacy/ebieot-gmm-alae`)

Frozen **pre-Hydra GMM / ALAE latent IOT** line (formerly branches `ALAE`, `ebieot-gmm-alae`).

**Active development is on [`main`](https://github.com/MuXauJl11110/EBiEOT/tree/main)** (`conf/`, `src/ebieot/`, Hydra notebooks).
See the [README on `main`](https://github.com/MuXauJl11110/EBiEOT/blob/main/README.md) for the current workflow.
Check out this branch only when you need to reproduce an experiment from the old layout.

If you previously tracked `origin/ALAE` or `origin/ebieot-gmm-alae`, run:

```bash
git fetch origin && git checkout legacy/ebieot-gmm-alae
```

## Checkout

```bash
git checkout legacy/ebieot-gmm-alae
uv sync --group dev
```

On **`main`**, use `uv sync` from the [main README](https://github.com/MuXauJl11110/EBiEOT/blob/main/README.md).

Preservation tag for the pre-docs code tip: `legacy/alae`.
Documented tip: tag `legacy/ebieot-gmm-alae` at this branch tip.

## Experiment map

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

## Layout vs `main`

| | `legacy/ebieot-gmm-alae` | `main` |
|--|--------------------------|--------|
| Config | `configs/` (Pydantic) | `conf/` (Hydra) |
| Models | `src/models/` (`gmm_based.py`, `light_sb.py`, `light_sbm.py`) | `src/ebieot/` |
| ALAE notebooks | `GMMEOT_ALAE_*.ipynb` | `ebieot_gmm_alae*.ipynb` |
| Tests | `tests/test_gmm_based.py` | Migrated under `main` layout |
| Logging | wandb in ALAE notebooks | Comet via Hydra `logger=comet` |

## Related branches

- **`legacy/ebieot-nn`** — pre-Hydra EgEOT / energy-based line (formerly `EBM`). See README and `LEGACY.md` on that branch.
- **`legacy/ebm-alae`** — merged EBM + ALAE snapshot. See `LEGACY.md` on that branch for the combined map.
- Tag **`legacy/alae`** — ALAE tip before this rename/docs commit.

## Verification

```bash
uv run pytest tests/test_gmm_based.py -q
```

Full notebook runs require datasets and long training and were not re-executed for this documentation.
Re-run the notebook for your experiment after checkout if you rely on this branch.

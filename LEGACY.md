# Legacy experiments (`legacy/ebieot-nn`)

Frozen **pre-Hydra EgEOT / energy-based IOT** line (formerly branch `EBM`).

**Active development is on [`main`](https://github.com/MuXauJl11110/EBiEOT/tree/main)** (`conf/`, `src/ebieot/`, Hydra notebooks).
See the [README on `main`](https://github.com/MuXauJl11110/EBiEOT/blob/main/README.md) for the current workflow.
Check out this branch only when you need to reproduce an experiment from the old layout.

If you previously tracked `origin/EBM`, run:

```bash
git fetch origin && git checkout legacy/ebieot-nn
```

## Checkout

```bash
git checkout legacy/ebieot-nn
pip install -r requirements.txt
```

On **`main`**, use `uv sync` from the [main README](https://github.com/MuXauJl11110/EBiEOT/blob/main/README.md).

Preservation tag for the pre-docs code tip: `legacy/ebm`.
Documented tip: `legacy/ebieot-nn`.

## Experiment map

| Experiment | `legacy/ebieot-nn` | `main` |
|------------|-------------------|--------|
| Weather EBiEOT-GMM + baselines | `notebooks/weather/*.ipynb` (Comet, `COMET_API_KEY`) | Same paths; Hydra configs under `conf/` |
| Swiss roll GMM | `notebooks/GMMEOT_swiss_roll.ipynb` (flat) | `notebooks/swiss_roll/ebieot_gmm.ipynb` |
| Swiss roll EgEOT / NN | `notebooks/EgEOT_swiss_roll.ipynb`, `run_ebm.py` | `notebooks/swiss_roll/ebieot_nn.ipynb`, `scripts/run_neural_swiss_roll.py` |
| Colored MNIST | `notebooks/EgEOT_colored_mnist.ipynb` (flat) | `notebooks/colored_mnist/ebieot_colored_mnist.ipynb` |
| MNIST 2→3 (EBM anatomy) | `mnist2to3/` | **Not on `main`** |
| ALAE / FFHQ | **Not on this branch** | `notebooks/ALAE/` |
| MNIST classification | **Not on this branch** | `notebooks/classification/` |
| Toy baselines | `toy_baselines/` | `scripts/train_baseline.py` + `conf/baseline/` |

## Layout vs `main`

| | `legacy/ebieot-nn` | `main` |
|--|-------------------|--------|
| Config | `configs/` (Pydantic) | `conf/` (Hydra) |
| Models | `src/models/` (`energy_based.py`, `gmm_based.py`) | `src/ebieot/` |
| Training entry | Notebooks, `run_ebm.py`, `run_swiss_roll.py` | `scripts/train.py`, `scripts/train_baseline.py` |
| Logging | Comet (weather); older wandb in some notebooks | Comet via Hydra `logger=comet` |

## Related branches

- **`legacy/ebieot-gmm-alae`** — pre-Hydra GMM / ALAE line. See README and `LEGACY.md` on that branch.
- **`legacy/ebm-alae`** — merged EBM + ALAE snapshot (includes ALAE notebooks and `light_sb` models). See its `LEGACY.md` for the combined experiment map.
- Tag **`legacy/ebm`** — EBM tip before this rename/docs commit.
- Tag **`legacy/ebieot-nn`** — this branch tip (includes this file).

## Verification

Full notebook runs require datasets and long training and were not re-executed for this documentation.
After checkout, re-run the notebook or script for your experiment.

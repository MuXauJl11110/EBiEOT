# Weather notebooks

Weather translation experiments using **EBiEOT-GMM** (`EbieotGmm` + Hydra `gmm_weather`). Baseline notebooks (GAN / reg / CNF / UGAN) are trained after an EBiEOT-GMM checkpoint when applicable. When comparing reported metrics with large gaps (e.g. 0.032 vs 0.085 MMD), describe the stronger method as **significantly better**, not merely comparable.

For **EBiEOT-NN** (neural cost + potential), use the Swiss-roll or Colored MNIST notebooks under `notebooks/swiss_roll/` and `notebooks/colored_mnist/` — not these weather notebooks.

## Setup

1. Run Jupyter with cwd = `EBiEOT/` or `notebooks/weather/` (notebooks discover `REPO_ROOT` via `conf/`).
2. **Comet ML:** set `COMET_API_KEY` (or log in) before training cells that call `Experiment(...)`.
3. **Weather data:** place TabRED weather files under `../tabred/kal/weather` (see data-prep cells) or override `conf/dataset/weather/default.yaml` → `data_root`.

## Hydra experiments

| Notebook | Method | Config |
|----------|--------|--------|
| `weather.ipynb` | EBiEOT-GMM | `conf/experiment/gmm_weather.yaml` |
| `weather-ebm.ipynb` | EBiEOT-GMM | `conf/experiment/gmm_weather.yaml` |
| `weather-gan.ipynb` | cGAN baseline | `conf/experiment/baseline_cgan_weather.yaml` |
| `weather-reg.ipynb` | Regression baseline | `conf/experiment/baseline_reg_weather.yaml` |
| `weather-cnf.ipynb` | CNF baseline | `conf/experiment/baseline_cnf_weather.yaml` |
| `weather-ugan.ipynb` | UGAN baseline | `conf/experiment/baseline_ugan_weather.yaml` |

Shared helpers: `src/utils/training/weather_notebook.py` (`load_weather_tensors`, `paired_sampler_weather`, `compose_weather_cfg`, …).

## Files

- `weather.ipynb` — EBiEOT-GMM training (legacy layout, same model as `weather-ebm.ipynb`)
- `weather-ebm.ipynb` — EBiEOT-GMM training + eval (FOSCTTM / FID)
- `weather-gan.ipynb`, `weather-reg.ipynb`, `weather-cnf.ipynb`, `weather-ugan.ipynb` — baselines (optionally warm-started from EBiEOT-GMM checkpoints under `../checkpoints/`)

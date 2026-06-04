import sys
from pathlib import Path

import optuna
from optuna.exceptions import TrialPruned

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = Path(__file__).resolve().parent
_PLOOMBER_DIR = _REPO_ROOT / "ploomber_notebooks"
_NOTEBOOK = _REPO_ROOT / "notebooks/swiss_roll/ebieot_gmm.ipynb"

if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

try:
    from scripts.run_swiss_roll import grid_point_to_notebook_params
    from scripts.notebook_sweep_utils import run_notebook_and_collect_metric
except ImportError:
    from run_swiss_roll import grid_point_to_notebook_params
    from notebook_sweep_utils import run_notebook_and_collect_metric


def objective(trial: optuna.Trial) -> float:
    n_potentials = trial.suggest_int("nPotentials", 10, 150)
    m_potentials = trial.suggest_int("mPotentials", 5, n_potentials)
    grid_point = {
        "EXPERIMENT": "gmm-swiss-roll",
        "nPotentials": n_potentials,
        "mPotentials": m_potentials,
        "qXUnpairedSamples": trial.suggest_categorical("qXUnpairedSamples", [0, 1024]),
        "rYUnpairedSamples": trial.suggest_categorical("rYUnpairedSamples", [0, 1024]),
        "pXyPairedSamples": trial.suggest_categorical("pXyPairedSamples", [128]),
        "lrPaired": trial.suggest_float("lrPaired", 1e-5, 1e-2, log=True),
        "lrUnpaired": trial.suggest_float("lrUnpaired", 1e-5, 1e-2, log=True),
        "maxSteps": 10000,
        "yDim": 2,
    }
    parameters = grid_point_to_notebook_params(grid_point)

    output_name = f"trial_{trial.number}_n={n_potentials}_m={m_potentials}"
    output_path = _PLOOMBER_DIR / f"{output_name}.ipynb"

    result = run_notebook_and_collect_metric(
        notebook_input_path=str(_NOTEBOOK),
        notebook_output_path=str(output_path),
        parameters=parameters,
        metric_name="target_metric",
        progress_bar=False,
    )
    if result.status != "ok" or result.metric_value is None:
        print(f"[Trial {trial.number}] Notebook execution failed: {result.error}")
        raise TrialPruned() from RuntimeError(result.error or "Unknown notebook error")

    return float(result.metric_value)


if __name__ == "__main__":
    _PLOOMBER_DIR.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(study_name="EBiEOT_GMM_Swiss_Roll_Optimization", direction="minimize")

    print("Starting parallel Optuna optimization...")
    study.optimize(objective, n_trials=50, n_jobs=4)

    print("\nOptimization Finished!")
    print(f"Best Trial: {study.best_trial.number}")
    print(f"Best Metric Value: {study.best_trial.value}")
    print("Best Hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")

import os

import optuna
import papermill as pm
import scrapbook as sb
from optuna.exceptions import TrialPruned


def objective(trial):
    # 1. Define the Hyperparameter Search Space
    # Suggesting integers and dynamically enforcing the N >= M constraint
    n_potentials = trial.suggest_int("N_POTENTIALS", 10, 150)
    m_potentials = trial.suggest_int("M_POTENTIALS", 5, n_potentials)

    # Categoricals for data ablation as defined in your setup
    q_x_unpaired = trial.suggest_categorical("Q_X_UNPAIRED_SAMPLES", [0, 1024])
    r_y_unpaired = trial.suggest_categorical("R_Y_UNPAIRED_SAMPLES", [0, 1024])
    p_xy_paired = trial.suggest_categorical("P_XY_PAIRED_SAMPLES", [128])

    # Suggesting log-uniform learning rates
    lr_paired = trial.suggest_float("LR_PAIRED", 1e-5, 1e-2, log=True)
    lr_unpaired = trial.suggest_float("LR_UNPAIRED", 1e-5, 1e-2, log=True)

    parameters = {
        "N_POTENTIALS": n_potentials,
        "M_POTENTIALS": m_potentials,
        "Q_X_UNPAIRED_SAMPLES": q_x_unpaired,
        "R_Y_UNPAIRED_SAMPLES": r_y_unpaired,
        "P_XY_PAIRED_SAMPLES": p_xy_paired,
        "LR_PAIRED": lr_paired,
        "LR_UNPAIRED": lr_unpaired,
        "MAX_STEPS": 10000,
    }

    # 2. Setup output paths
    output_name = f"trial_{trial.number}_N={n_potentials}_M={m_potentials}"
    output_path = f"./ploomber_notebooks/{output_name}.ipynb"

    # 3. Execute the Notebook via Papermill
    try:
        pm.execute_notebook(
            input_path="./notebooks/swiss_roll/GMMEOT_swiss_roll_another_plan.ipynb",
            output_path=output_path,
            parameters=parameters,
            progress_bar=False,  # Disable progress bars to keep logs clean in parallel
        )
    except pm.exceptions.PapermillExecutionError as e:
        print(f"[Trial {trial.number}] Notebook execution failed: {e}")
        raise TrialPruned()  # Tell Optuna to ignore this failed run and move on

    # 4. Read the target metric back from the executed notebook
    try:
        nb = sb.read_notebook(output_path)
        # Fetching the 'target_metric' we glued at the end of the notebook
        metric_value = nb.scraps["target_metric"].data
    except Exception as e:
        print(f"[Trial {trial.number}] Failed to read metric via Scrapbook: {e}")
        raise TrialPruned()

    return metric_value


if __name__ == "__main__":
    os.makedirs("./ploomber_notebooks", exist_ok=True)

    # Set direction="minimize" for metrics like Wasserstein/MMD, or "maximize" if it's an accuracy score
    study = optuna.create_study(study_name="GMMEOT_Swiss_Roll_Optimization", direction="minimize")

    # Optuna natively handles multiprocessing via the n_jobs parameter
    print("Starting parallel Optuna optimization...")
    study.optimize(objective, n_trials=50, n_jobs=4)

    # Print results
    print("\nOptimization Finished!")
    print(f"Best Trial: {study.best_trial.number}")
    print(f"Best Metric Value: {study.best_trial.value}")
    print("Best Hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
        print(f"    {key}: {value}")

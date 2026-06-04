import multiprocessing
from itertools import product
from pathlib import Path
from typing import Any

try:
    from sklearn.model_selection import ParameterGrid
except ImportError:  # pragma: no cover - exercised only in minimal environments
    def ParameterGrid(param_grid: dict[str, list[Any]]):  # type: ignore[misc]
        keys = list(param_grid.keys())
        for values in product(*(param_grid[k] for k in keys)):
            yield dict(zip(keys, values))

try:
    from scripts.notebook_sweep_utils import execute_notebook
except ImportError:
    from notebook_sweep_utils import execute_notebook

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PAPEROUTPUT_DIR = _REPO_ROOT / "papermill-notebooks"


def run_notebook(
    notebook_input_path: str, notebook_output_path: str, parameters: None | dict[str, Any] = None
):
    """
    Executes a Jupyter notebook with Papermill.

    Args:
    - notebook_input_path (str): The path to the input notebook.
    - notebook_output_path (str): The path to save the executed notebook.
    - parameters (dict, optional): A dictionary of parameters to pass to the notebook.

    Returns:
    - notebook_output_path (str): The path to the output notebook.
    """
    execute_notebook(
        notebook_input_path=notebook_input_path,
        notebook_output_path=notebook_output_path,
        parameters=parameters,
        progress_bar=False,
    )
    return notebook_output_path


def display_experiment(hydra_name: str) -> str:
    return hydra_name.replace("_", "-")


def hydra_experiment(display_name: str) -> str:
    aliases = {
        "gmm-swiss-roll": "gmm_swiss_roll",
        "gmm-swiss-roll-shared": "gmm_swiss_roll_shared",
    }
    return aliases.get(display_name, display_name.replace("-", "_"))


def grid_point_to_notebook_params(grid_point: dict[str, Any]) -> dict[str, Any]:
    """Map search-grid keys to Hydra EXPERIMENT + OVERRIDES for ebieot_gmm_swiss_roll."""
    overrides: list[str] = list(grid_point.get("OVERRIDES") or [])

    n_potentials = grid_point.get("nPotentials")
    m_potentials = grid_point.get("mPotentials")
    y_dim = int(grid_point.get("yDim", 2))

    if n_potentials is not None:
        overrides.append(f"ebieot.model.n_potentials={n_potentials}")
    if m_potentials is not None:
        overrides.append(f"ebieot.cost.m_potentials={m_potentials}")
    if (v := grid_point.get("pXyPairedSamples")) is not None:
        overrides.append(f"dataset.P_XY_paired={v}")
    if (v := grid_point.get("qXUnpairedSamples")) is not None:
        overrides.append(f"dataset.Q_X_unpaired={v}")
    if (v := grid_point.get("rYUnpairedSamples")) is not None:
        overrides.append(f"dataset.R_Y_unpaired={v}")
    if (v := grid_point.get("lrPaired")) is not None:
        overrides.append(f"train.optimizer.paired.lr={v}")
    if (v := grid_point.get("lrUnpaired")) is not None:
        overrides.append(f"train.optimizer.unpaired.lr={v}")
    if (v := grid_point.get("maxSteps")) is not None:
        overrides.append(f"train.steps_to={v}")
    if (v := grid_point.get("seed")) is not None:
        overrides = [o for o in overrides if not o.startswith("train.seed=")]
        overrides.append(f"train.seed={int(v)}")

    experiment = grid_point.get("EXPERIMENT")
    if experiment is None:
        p_xy = grid_point.get("pXyPairedSamples")
        if p_xy == 16000:
            experiment = "gmm-swiss-roll-shared"
        else:
            experiment = "gmm-swiss-roll"
    elif "_" in str(experiment):
        experiment = display_experiment(str(experiment))

    return {"EXPERIMENT": experiment, "OVERRIDES": overrides}


def run_notebook_in_parallel(max_processes: int, notebook_name: str, param_grid: dict):
    """
    Executes multiple Jupyter notebooks in parallel.
    """
    print(f"max_processes: {max_processes}")
    print(f"notebook_name: {notebook_name}")

    with multiprocessing.Pool(processes=max_processes) as pool:
        processes_list: list[tuple] = []

        for params in ParameterGrid(param_grid):
            n_potentials = params.get("nPotentials")
            m_potentials = params.get("mPotentials")
            if n_potentials is not None and m_potentials is not None and n_potentials < m_potentials:
                continue

            nb_params = grid_point_to_notebook_params(params)
            output_name = "-".join(
                f"{key}={val}"
                for (key, val) in params.items()
                if key
                in {
                    "EXPERIMENT",
                    "nPotentials",
                    "mPotentials",
                    "pXyPairedSamples",
                    "qXUnpairedSamples",
                    "rYUnpairedSamples",
                    "maxSteps",
                }
            )
            processes_list.append(
                (
                    notebook_name,
                    str(_PAPEROUTPUT_DIR / f"{output_name}.ipynb"),
                    nb_params,
                )
            )
        results = pool.starmap(run_notebook, processes_list)

    return results


if __name__ == "__main__":
    max_processes = 4
    notebook_name = str(_REPO_ROOT / "notebooks/swiss_roll/ebieot_gmm.ipynb")
    param_grid = {
        "nPotentials": [50, 100],
        "mPotentials": [10],
        "qXUnpairedSamples": [0, 1024],
        "rYUnpairedSamples": [0, 1024],
        "pXyPairedSamples": [128],
        "lrPaired": [3e-4],
        "lrUnpaired": [1e-3],
        "maxSteps": [100000],
        "yDim": [2],
    }

    results = run_notebook_in_parallel(max_processes, notebook_name, param_grid)

    for result in results:
        print(f"Completed: {result}")

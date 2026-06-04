import multiprocessing
from pathlib import Path
from typing import Any

import papermill as pm
from sklearn.model_selection import ParameterGrid

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
    pm.execute_notebook(notebook_input_path, notebook_output_path, parameters=parameters)
    return notebook_output_path


def display_experiment(hydra_name: str) -> str:
    return hydra_name.replace("_", "-")


def hydra_experiment(display_name: str) -> str:
    aliases = {
        "egeot-swiss-roll": "egeot_swiss_roll",
        "egeot-swiss-roll-128": "egeot_swiss_roll_128",
        "egeot-swiss-roll-16k": "egeot_swiss_roll_16k",
    }
    return aliases.get(display_name, display_name.replace("-", "_"))


def grid_point_to_notebook_params(grid_point: dict[str, Any]) -> dict[str, Any]:
    """Map search-grid keys to Hydra EXPERIMENT + OVERRIDES for ebieot_neural_swiss_roll."""
    overrides: list[str] = list(grid_point.get("OVERRIDES") or [])

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
    if (v := grid_point.get("samplingNumIter")) is not None:
        overrides.append(f"ebieot.model.num_iterations={v}")
    if (v := grid_point.get("maxSteps")) is not None:
        overrides.append(f"train.steps_to={v}")
    if (v := grid_point.get("hiddenLayers")) is not None:
        overrides.append(f"ebieot.cost.hidden_layers={v}")
    if (v := grid_point.get("potentialHiddenLayers")) is not None:
        overrides.append(f"ebieot.potential.hidden_layers={v}")

    cost_fn = grid_point.get("costFunction")
    if cost_fn is not None:
        cost_key = str(cost_fn).lower()
        if cost_key == "mlp":
            overrides.append("ebieot/costs/nn@ebieot.cost=mlp")
        elif cost_key in {"mlplse", "mlp-lse"}:
            overrides.append("ebieot/costs/lse@ebieot.cost=mlp_lse")
            if (v := grid_point.get("mPotentials")) is not None:
                overrides.append(f"ebieot.cost.m_potentials={v}")
            if (v := grid_point.get("logVmHiddenChannels")) is not None:
                overrides.append(f"ebieot.cost.log_v_m_hidden_channels={v}")
            if (v := grid_point.get("bMHiddenChannels")) is not None:
                overrides.append(f"ebieot.cost.b_m_hidden_channels={v}")
        elif cost_key in {"mlpl2", "mlp-l2"}:
            overrides.append("ebieot/costs/nn@ebieot.cost=mlp_l2")
        else:
            raise ValueError(f"Unknown costFunction: {cost_fn}")

    experiment = grid_point.get("EXPERIMENT")
    if experiment is None:
        p_xy = grid_point.get("pXyPairedSamples")
        if p_xy == 16000:
            experiment = "egeot-swiss-roll-16k"
        elif p_xy == 128 and grid_point.get("maxSteps") == 3000:
            experiment = "egeot-swiss-roll-128"
        else:
            experiment = "egeot-swiss-roll"
    elif "_" in str(experiment):
        experiment = display_experiment(str(experiment))

    return {"EXPERIMENT": experiment, "OVERRIDES": overrides}


def run_notebook_in_parallel(max_processes: int, notebook_name: str, param_grid: dict):
    """
    Executes multiple Jupyter notebooks in parallel.

    Args:
    - notebooks (list of dict): A list of dictionaries, where each dictionary contains
                                'input' (str): path to the input notebook,
                                'output' (str): path to save the executed notebook,
                                'parameters' (dict): parameters to pass to the notebook.
    """
    print(f"max_processes: {max_processes}")
    print(f"notebook_name: {notebook_name}")

    with multiprocessing.Pool(processes=max_processes) as pool:
        processes_list: list[tuple] = []

        for params in ParameterGrid(param_grid):
            nb_params = grid_point_to_notebook_params(params)
            output_name = "-".join(
                f"{key}={val}"
                for (key, val) in params.items()
                if key
                in {
                    "EXPERIMENT",
                    "pXyPairedSamples",
                    "qXUnpairedSamples",
                    "rYUnpairedSamples",
                    "hiddenLayers",
                    "potentialHiddenLayers",
                    "maxSteps",
                    "costFunction",
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
    max_processes = 3
    notebook_name = str(_REPO_ROOT / "notebooks/swiss_roll/ebieot_nn.ipynb")
    param_grid = {
        "hiddenLayers": [[128, 128], [256]],
        "potentialHiddenLayers": [[256, 256, 256], [128, 128, 128]],
        "pXyPairedSamples": [128],
        "qXUnpairedSamples": [1024],
        "rYUnpairedSamples": [1024],
        "lrPaired": [1e-4],
        "lrUnpaired": [1e-4],
        "samplingNumIter": [100],
        "maxSteps": [3000],
        "costFunction": ["MLP"],
    }
    results = run_notebook_in_parallel(max_processes, notebook_name, param_grid)

    for result in results:
        print(f"Completed: {result}")

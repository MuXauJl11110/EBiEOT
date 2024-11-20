import multiprocessing

import papermill as pm
from sklearn.model_selection import ParameterGrid


def run_notebook(
    notebook_input_path: str, notebook_output_path: str, parameters: None | dict[str, list[float]] = None
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
            output_name = "_".join(
                f"{key}={val}"
                for (key, val) in params.items()
                if key in {"P_XY_PAIRED_SAMPLES", "Q_X_UNPAIRED_SAMPLES", "R_Y_UNPAIRED_SAMPLES"}
            )
            processes_list.append((notebook_name, f"./papermill_notebooks/{output_name}.ipynb", params))
        results = pool.starmap(run_notebook, processes_list)

    return results


if __name__ == "__main__":
    max_processes = 3
    notebook_name = "./notebooks/EgEOT_swiss_roll.ipynb"
    # TODO: add config for hidden layers for cost
    param_grid = {
        "M_POTENTIALS": [2],
        "LOG_V_M_HIDDEN_CHANNELS": [[128, 128], [256]],
        "B_M_HIDDEN_CHANNELS": [[128, 128], [256, 256]],
        "HIDDEN_LAYERS": [[512]],
        "POTENTIAL_HIDDEN_LAYERS": [[256, 256, 256], [128, 128, 128]],
        "P_XY_PAIRED_SAMPLES": [128],
        "Q_X_UNPAIRED_SAMPLES": [1024],
        "R_Y_UNPAIRED_SAMPLES": [1024],
        "LR_PAIRED": [1e-4],
        "LR_UNPAIRED": [1e-4],
        "SAMPLING_NUM_ITER": [100],
        "MAX_STEPS": [3000],
        "COST_FUNCTION": ["MLP"],
    }
    # Run the notebooks in parallel
    results = run_notebook_in_parallel(max_processes, notebook_name, param_grid)

    for result in results:
        print(f"Completed: {result}")

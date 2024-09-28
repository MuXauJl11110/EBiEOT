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
            if params["N_POTENTIALS"] >= params["M_POTENTIALS"]:
                output_name = "_".join(f"{key}={val}" for (key, val) in params.items())
                processes_list.append((notebook_name, f"./ploomber_notebooks/{output_name}.ipynb", params))
        results = pool.starmap(run_notebook, processes_list)

    return results


if __name__ == "__main__":
    max_processes = 4
    notebook_name = "./notebooks/LightGCOT_swiss_roll.ipynb"
    param_grid = {
        "N_POTENTIALS": [50, 100],
        "M_POTENTIALS": [10],
        "M_X_UNPAIRED_SAMPLES": [0, 1024],
        "N_Y_UNPAIRED_SAMPLES": [0, 1024],
        "L_PAIRED_SAMPLES": [128],
        "D_LR_PAIRED": [3e-4],
        "D_LR_UNPAIRED": [1e-3],
        "MAX_STEPS": [100000],
        "EXP_COST": ["MLP_deep"],
    }

    # Run the notebooks in parallel
    results = run_notebook_in_parallel(max_processes, notebook_name, param_grid)

    for result in results:
        print(f"Completed: {result}")

"""Run Swiss-roll baseline training jobs over a parameter grid."""


import multiprocessing
import subprocess
import sys
from pathlib import Path
from typing import Any

from sklearn.model_selection import ParameterGrid

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TRAIN_SCRIPT = _REPO_ROOT / "scripts" / "train_baseline.py"


def grid_point_to_overrides(grid_point: dict[str, Any]) -> list[str]:
    """Map search-grid keys to Hydra CLI overrides for ``train_baseline.py``."""
    overrides: list[str] = list(grid_point.get("overrides") or [])

    experiment = grid_point.get("experiment")
    if experiment is not None:
        overrides.insert(0, f"experiment={experiment}")

    if (v := grid_point.get("p_xy_paired")) is not None:
        overrides.append(f"dataset.P_XY_paired={v}")
    if (v := grid_point.get("q_x_unpaired")) is not None:
        overrides.append(f"dataset.Q_X_unpaired={v}")
    if (v := grid_point.get("r_y_unpaired")) is not None:
        overrides.append(f"dataset.R_Y_unpaired={v}")
    if (v := grid_point.get("steps_to")) is not None:
        overrides.append(f"train.steps_to={v}")
    if (v := grid_point.get("save_every")) is not None:
        overrides.append(f"train.save_every={v}")
    if (v := grid_point.get("paired_cache_enabled")) is not None:
        overrides.append(f"dataset.paired_cache.enabled={v}")
    if (v := grid_point.get("lr_g")) is not None:
        overrides.append(f"train.lr_g={v}")
    if (v := grid_point.get("lr_d")) is not None:
        overrides.append(f"train.lr_d={v}")
    if (v := grid_point.get("lr")) is not None:
        overrides.append(f"train.lr={v}")

    return overrides


def run_train(overrides: list[str]) -> str:
    cmd = [sys.executable, str(_TRAIN_SCRIPT), *overrides]
    subprocess.run(cmd, check=True, cwd=_REPO_ROOT)
    return " ".join(cmd)


def run_in_parallel(max_processes: int, param_grid: dict[str, list[Any]]) -> list[str]:
    with multiprocessing.Pool(processes=max_processes) as pool:
        jobs: list[list[str]] = []
        for params in ParameterGrid(param_grid):
            overrides = grid_point_to_overrides(params)
            jobs.append(overrides)
        return pool.map(run_train, jobs)


if __name__ == "__main__":
    max_processes = 2
    param_grid = {
        "experiment": [
            "baseline_reg_swiss_roll_16k",
            "baseline_ugan_swiss_roll_16k",
        ],
        "p_xy_paired": [128],
        "q_x_unpaired": [128],
        "r_y_unpaired": [0],
        "steps_to": [2],
        "paired_cache_enabled": [False],
        "overrides": [["debug=default", "extras.print_config=false"]],
    }

    results = run_in_parallel(max_processes, param_grid)
    for result in results:
        print(f"Completed: {result}")

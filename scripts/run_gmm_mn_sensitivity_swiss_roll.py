"""Run M/N sensitivity analysis for Swiss-roll EBiEOT-GMM notebook.

Sweeps ``n_potentials``, ``m_potentials``, and ``seed`` (default three seeds). Each run
is stored as one CSV row; use ``plot_gmm_mn_sensitivity.py`` to average over seeds.
Dataset sizes are fixed to P=128 paired, Q=1024 unpaired X, R=1024 unpaired Y
(see ``gmm_swiss_roll.yaml``).

This runner forces an explicit Jupyter kernel to avoid stale notebook metadata
that may point to a deleted Python executable.
"""


import json
import multiprocessing
import os
import subprocess
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
    from scripts.notebook_sweep_utils import write_results_csv
    from scripts.run_swiss_roll import grid_point_to_notebook_params
except ImportError:
    from notebook_sweep_utils import write_results_csv
    from run_swiss_roll import grid_point_to_notebook_params

_REPO_ROOT = Path(__file__).resolve().parents[1]
_NOTEBOOK = _REPO_ROOT / "notebooks" / "swiss_roll" / "ebieot_gmm.ipynb"
_PAPEROUTPUT_DIR = _REPO_ROOT / "papermill-notebooks"
_METRICS_DIR = _PAPEROUTPUT_DIR / "metrics"

# Fixed dataset sizes (P, Q, R) — not swept; matches conf/experiment/gmm_swiss_roll.yaml
P_XY_PAIRED_SAMPLES = 128
Q_X_UNPAIRED_SAMPLES = 1024
R_Y_UNPAIRED_SAMPLES = 1024
SENSITIVITY_SEEDS: tuple[int, ...] = (0, 1, 2)

_OUTPUT_KEYS = (
    "experiment",
    "n_potentials",
    "m_potentials",
    "seed",
    "p_xy_paired_samples",
    "q_x_unpaired_samples",
    "r_y_unpaired_samples",
    "max_steps",
)

_NOTEBOOK_KEY_ALIASES = {
    "experiment": "EXPERIMENT",
    "n_potentials": "nPotentials",
    "m_potentials": "mPotentials",
    "p_xy_paired_samples": "pXyPairedSamples",
    "q_x_unpaired_samples": "qXUnpairedSamples",
    "r_y_unpaired_samples": "rYUnpairedSamples",
    "lr_paired": "lrPaired",
    "lr_unpaired": "lrUnpaired",
    "max_steps": "maxSteps",
    "y_dim": "yDim",
}


def to_notebook_grid_point(grid_point: dict[str, Any]) -> dict[str, Any]:
    return {
        _NOTEBOOK_KEY_ALIASES.get(key, key): value for key, value in grid_point.items()
    }


def is_valid_mn_point(grid_point: dict[str, Any]) -> bool:
    n_potentials = grid_point.get("n_potentials")
    m_potentials = grid_point.get("m_potentials")
    if n_potentials is None or m_potentials is None:
        return True
    return int(n_potentials) >= int(m_potentials)


def build_output_name(grid_point: dict[str, Any]) -> str:
    parts = [f"{key}={grid_point[key]}" for key in _OUTPUT_KEYS if key in grid_point]
    return "-".join(parts)


def build_job(grid_point: dict[str, Any]) -> dict[str, Any]:
    notebook_params = grid_point_to_notebook_params(to_notebook_grid_point(grid_point))
    seed = grid_point.get("seed")
    if seed is not None:
        notebook_params = _notebook_params_for_seed(notebook_params, int(seed))
    output_name = build_output_name(grid_point)
    output_path = str(_PAPEROUTPUT_DIR / f"mn_sensitivity-{output_name}.ipynb")
    return {
        "grid_point": grid_point,
        "notebook_path": str(_NOTEBOOK),
        "output_path": output_path,
        "notebook_params": notebook_params,
    }


def _notebook_params_for_seed(
    notebook_params: dict[str, Any], seed: int
) -> dict[str, Any]:
    overrides = [
        item
        for item in notebook_params.get("OVERRIDES", [])
        if not str(item).startswith("train.seed=")
    ]
    overrides.append(f"train.seed={int(seed)}")
    return {**notebook_params, "OVERRIDES": overrides}


def _execute_notebook_with_kernel(
    notebook_path: str,
    output_path: str,
    parameters: dict[str, Any],
    kernel_name: str,
) -> None:
    import papermill as pm

    pm.execute_notebook(
        input_path=notebook_path,
        output_path=output_path,
        parameters=parameters,
        progress_bar=False,
        kernel_name=kernel_name,
    )


def _scrap_float(scraps: Any, *names: str) -> float | None:
    for name in names:
        if name in scraps:
            return float(scraps[name].data)
    return None


def _read_notebook_scraps(output_path: str) -> dict[str, float | int | None]:
    import scrapbook as sb  # type: ignore[import-not-found]

    notebook = sb.read_notebook(output_path)
    scraps = notebook.scraps

    mmd = _scrap_float(scraps, "mmd_metric", "target_metric")
    if mmd is None:
        raise KeyError(
            "Metric 'mmd_metric' or 'target_metric' was not found in notebook scraps."
        )
    sinkhorn = _scrap_float(scraps, "sinkhorn_metric")
    if sinkhorn is None:
        raise KeyError("Metric 'sinkhorn_metric' was not found in notebook scraps.")

    conditional_mmd = _scrap_float(scraps, "conditional_mmd_metric")
    conditional_sinkhorn = _scrap_float(scraps, "conditional_sinkhorn_metric")

    if "effective_n_potentials" not in scraps:
        raise KeyError(
            "Metric 'effective_n_potentials' was not found in notebook scraps."
        )
    if "effective_m_potentials" not in scraps:
        raise KeyError(
            "Metric 'effective_m_potentials' was not found in notebook scraps."
        )

    return {
        "mmd": mmd,
        "sinkhorn": sinkhorn,
        "conditional_mmd": conditional_mmd,
        "conditional_sinkhorn": conditional_sinkhorn,
        "effective_n": int(scraps["effective_n_potentials"].data),
        "effective_m": int(scraps["effective_m_potentials"].data),
    }


def _extract_override_int(overrides: list[str], key: str) -> int | None:
    prefix = f"{key}="
    for item in overrides:
        if item.startswith(prefix):
            return int(item[len(prefix) :])
    return None


def _extract_effective_nm(
    notebook_params: dict[str, Any],
) -> tuple[int | None, int | None]:
    overrides = list(notebook_params.get("OVERRIDES", []))
    effective_n = _extract_override_int(overrides, "ebieot.model.n_potentials")
    effective_m = _extract_override_int(overrides, "ebieot.cost.m_potentials")
    return effective_n, effective_m


def _run_job(job: dict[str, Any]) -> dict[str, Any]:
    status = "ok"
    mmd: float | None = None
    sinkhorn: float | None = None
    conditional_mmd: float | None = None
    conditional_sinkhorn: float | None = None
    error: str | None = None
    effective_n: int | None = None
    effective_m: int | None = None

    try:
        _execute_notebook_with_kernel(
            notebook_path=job["notebook_path"],
            output_path=job["output_path"],
            parameters=job["notebook_params"],
            kernel_name=job["kernel_name"],
        )
        scraps = _read_notebook_scraps(job["output_path"])
        mmd = float(scraps["mmd"])  # type: ignore[arg-type]
        sinkhorn = float(scraps["sinkhorn"])  # type: ignore[arg-type]
        conditional_mmd = scraps["conditional_mmd"]
        conditional_sinkhorn = scraps["conditional_sinkhorn"]
        effective_n = int(scraps["effective_n"])  # type: ignore[arg-type]
        effective_m = int(scraps["effective_m"])  # type: ignore[arg-type]
        if conditional_mmd is None or conditional_sinkhorn is None:
            status = "failed"
            error = "conditional metrics missing from notebook scraps"
    except Exception as exc:  # noqa: BLE001
        status = "failed"
        error = str(exc)

    intended_n = job["grid_point"].get("n_potentials")
    intended_m = job["grid_point"].get("m_potentials")
    if effective_n is None or effective_m is None:
        fallback_n, fallback_m = _extract_effective_nm(job["notebook_params"])
        effective_n = fallback_n if effective_n is None else effective_n
        effective_m = fallback_m if effective_m is None else effective_m
    n_applied = intended_n is None or effective_n == int(intended_n)
    m_applied = intended_m is None or effective_m == int(intended_m)
    override_mismatch = not (n_applied and m_applied)
    if status == "ok" and override_mismatch:
        status = "failed"
        error = (
            "override mismatch: "
            f"requested(n={intended_n}, m={intended_m}) "
            f"effective(n={effective_n}, m={effective_m})"
        )

    record = {
        **job["grid_point"],
        "output_path": job["output_path"],
        "kernel_name": job["kernel_name"],
        "metric_name": "target_metric",
        "metric_value": mmd,
        "mmd": mmd,
        "sinkhorn": sinkhorn,
        "conditional_mmd": conditional_mmd,
        "conditional_sinkhorn": conditional_sinkhorn,
        "status": status,
        "error": error,
        "EXPERIMENT": job["notebook_params"]["EXPERIMENT"],
        "intended_n_potentials": intended_n,
        "intended_m_potentials": intended_m,
        "effective_n_potentials": effective_n,
        "effective_m_potentials": effective_m,
        "n_override_applied": n_applied,
        "m_override_applied": m_applied,
        "override_mismatch": override_mismatch,
    }
    return record


def default_mn_sensitivity_param_grid() -> dict[str, list[Any]]:
    """Parameter grid over N, M, and seed; P/Q/R are fixed."""
    return {
        "n_potentials": [8, 16, 32, 64],
        "m_potentials": [8, 16, 32, 64],
        "seed": list(SENSITIVITY_SEEDS),
        "p_xy_paired_samples": [P_XY_PAIRED_SAMPLES],
        "q_x_unpaired_samples": [Q_X_UNPAIRED_SAMPLES],
        "r_y_unpaired_samples": [R_Y_UNPAIRED_SAMPLES],
        "lr_paired": [3e-4],
        "lr_unpaired": [1e-3],
        "max_steps": [10000],
        "y_dim": [2],
    }


def run_sensitivity(
    param_grid: dict[str, list[Any]], max_processes: int, kernel_name: str
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for point in ParameterGrid(param_grid):
        job = build_job(point)
        job["kernel_name"] = kernel_name
        jobs.append(job)

    with multiprocessing.Pool(processes=max_processes) as pool:
        records = pool.map(_run_job, jobs)
    return records


def save_records(
    records: list[dict[str, Any]], stem: str = "gmm_mn_sensitivity"
) -> tuple[Path, Path]:
    _METRICS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = _METRICS_DIR / f"{stem}.json"
    csv_path = _METRICS_DIR / f"{stem}.csv"

    json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    write_results_csv(records, output_csv_path=csv_path)
    return json_path, csv_path


def resolve_kernel_name(explicit_kernel_name: str | None) -> str:
    if explicit_kernel_name:
        return explicit_kernel_name
    return (
        os.environ.get("PAPERMILL_KERNEL")
        or os.environ.get("JUPYTER_KERNEL_NAME")
        or "python3"
    )


def list_available_kernels() -> list[str]:
    try:
        result = subprocess.run(
            ["jupyter", "kernelspec", "list", "--json"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:  # noqa: BLE001
        return []
    try:
        payload = json.loads(result.stdout)
        kernelspecs = payload.get("kernelspecs", {})
        return sorted(kernelspecs.keys())
    except Exception:  # noqa: BLE001
        return []


def parse_args() -> tuple[int, str]:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-processes",
        type=int,
        default=4,
        help="Number of parallel notebook workers.",
    )
    parser.add_argument(
        "--kernel-name",
        type=str,
        default=None,
        help=(
            "Jupyter kernel to use for papermill (overrides notebook metadata). "
            "Defaults to PAPERMILL_KERNEL / JUPYTER_KERNEL_NAME / python3."
        ),
    )
    args = parser.parse_args()
    return int(args.max_processes), resolve_kernel_name(args.kernel_name)


if __name__ == "__main__":
    max_processes, kernel_name = parse_args()
    param_grid = default_mn_sensitivity_param_grid()

    available_kernels = list_available_kernels()
    if available_kernels and kernel_name not in available_kernels:
        print(
            f"Requested kernel '{kernel_name}' not found. "
            f"Available: {', '.join(available_kernels)}"
        )
        raise SystemExit(2)

    print(f"Using kernel: {kernel_name}")
    results = run_sensitivity(
        param_grid=param_grid,
        max_processes=max_processes,
        kernel_name=kernel_name,
    )
    json_path, csv_path = save_records(results)

    print(f"Saved JSON metrics to: {json_path}")
    print(f"Saved CSV metrics to: {csv_path}")
    for row in results:
        print(row)

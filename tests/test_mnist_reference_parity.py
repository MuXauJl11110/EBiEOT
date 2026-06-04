"""End-to-end parity vs Downloads/mnist_energy_grid.py (when present)."""


import csv
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
REF_SCRIPT = Path("/Users/michael/Downloads/mnist_energy_grid.py")


@pytest.mark.skipif(not REF_SCRIPT.exists(), reason="reference script not on machine")
@pytest.mark.parametrize("arch", ["cnn_embed", "mlp_embed"])
def test_grid_matches_reference(arch: str):
    slug = arch.replace("_embed", "")
    run_ref = f"parity_ref_{slug}"
    run_repo = f"parity_repo_{slug}"
    cli_tail = [
        "--paired-grid",
        "20",
        "--unpaired-grid",
        "0",
        "--epochs-max",
        "2",
        "--patience",
        "10",
        "--seed",
        "42",
        "--cpu",
        "--arch",
        arch,
    ]

    uv = "uv"
    subprocess.run(
        [uv, "run", "python", str(REF_SCRIPT), "--run-name", run_ref, *cli_tail],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            uv,
            "run",
            "python",
            str(REPO / "scripts" / "run_mnist_energy_grid.py"),
            "--run-name",
            run_repo,
            *cli_tail,
        ],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )

    ref_path = REPO / "results" / f"grid_results_seed_42_{run_ref}_{arch}.csv"
    repo_path = REPO / "results" / f"grid_results_seed_42_{run_repo}_{slug}.csv"
    def _read_row(path: Path) -> dict[str, float]:
        with path.open(newline="") as f:
            row = next(csv.DictReader(f))
        return {
            "test_acc": float(row["test_acc"]),
            "best_val_acc": float(row["best_val_acc"]),
            "test_loss": float(row["test_loss"]),
        }

    ref_row = _read_row(ref_path)
    repo_row = _read_row(repo_path)

    for key in ("test_acc", "best_val_acc", "test_loss"):
        assert ref_row[key] == pytest.approx(repo_row[key], rel=0, abs=1e-4)

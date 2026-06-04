
from pathlib import Path

import pytest

from scripts.run_gmm_mn_sensitivity_swiss_roll import (
    P_XY_PAIRED_SAMPLES,
    Q_X_UNPAIRED_SAMPLES,
    R_Y_UNPAIRED_SAMPLES,
    SENSITIVITY_SEEDS,
    _extract_effective_nm,
    _notebook_params_for_seed,
    _read_notebook_scraps,
    _run_job,
    build_output_name,
    build_job,
    default_mn_sensitivity_param_grid,
    is_valid_mn_point,
    to_notebook_grid_point,
)
from scripts.run_swiss_roll import grid_point_to_notebook_params

_REPO_ROOT = Path(__file__).resolve().parents[1]
_GMM_NOTEBOOK = _REPO_ROOT / "notebooks" / "swiss_roll" / "ebieot_gmm.ipynb"


def test_grid_point_to_notebook_params_maps_mn_to_hydra_overrides():
    grid_point = {
        "EXPERIMENT": "gmm-swiss-roll",
        "nPotentials": 50,
        "mPotentials": 10,
        "qXUnpairedSamples": 1024,
        "rYUnpairedSamples": 1024,
        "pXyPairedSamples": 128,
        "lrPaired": 3e-4,
        "lrUnpaired": 1e-3,
        "maxSteps": 10000,
        "yDim": 2,
    }

    mapped = grid_point_to_notebook_params(grid_point)

    assert mapped["EXPERIMENT"] == "gmm-swiss-roll"
    assert "ebieot.model.n_potentials=50" in mapped["OVERRIDES"]
    assert "ebieot.cost.m_potentials=10" in mapped["OVERRIDES"]
    assert not any("log_v_m_hidden_channels" in item for item in mapped["OVERRIDES"])
    assert not any("b_m_hidden_channels" in item for item in mapped["OVERRIDES"])
    assert "dataset.P_XY_paired=128" in mapped["OVERRIDES"]
    assert "dataset.Q_X_unpaired=1024" in mapped["OVERRIDES"]
    assert "dataset.R_Y_unpaired=1024" in mapped["OVERRIDES"]


def test_is_valid_mn_point_requires_n_geq_m():
    assert is_valid_mn_point({"n_potentials": 10, "m_potentials": 10})
    assert is_valid_mn_point({"n_potentials": 50, "m_potentials": 10})
    assert not is_valid_mn_point({"n_potentials": 5, "m_potentials": 10})


def test_build_output_name_includes_sensitivity_axes():
    point = {
        "experiment": "gmm-swiss-roll",
        "n_potentials": 50,
        "m_potentials": 10,
        "p_xy_paired_samples": 128,
        "max_steps": 10000,
    }
    output_name = build_output_name(point)
    assert "n_potentials=50" in output_name
    assert "m_potentials=10" in output_name
    assert "p_xy_paired_samples=128" in output_name


def test_gmm_notebook_exposes_target_metric_scrap():
    notebook_text = _GMM_NOTEBOOK.read_text(encoding="utf-8")
    assert 'sb.glue(\\"target_metric\\", target_metric)' in notebook_text
    assert 'sb.glue(\\"mmd_metric\\", mmd_metric)' in notebook_text
    assert 'sb.glue(\\"sinkhorn_metric\\", sinkhorn_metric)' in notebook_text
    assert 'sb.glue(\\"conditional_mmd_metric\\", conditional_mmd_metric)' in notebook_text
    assert 'sb.glue(\\"conditional_sinkhorn_metric\\", conditional_sinkhorn_metric)' in notebook_text
    assert 'sb.glue(\\"effective_n_potentials\\", effective_n_potentials)' in notebook_text
    assert 'sb.glue(\\"effective_m_potentials\\", effective_m_potentials)' in notebook_text
    assert '\\"sensitivity.override_match\\": bool(override_match)' in notebook_text
    assert 'print(f\\"override_match: {override_match}\\")' in notebook_text
    assert 'Papermill injects values in a new cell *after* this one.' in notebook_text
    assert 'EXPERIMENT = \\"gmm-swiss-roll\\"' in notebook_text
    assert 'cfg, EXPERIMENT_KEY, seed = compose_swiss_roll_cfg(' in notebook_text
    assert 'print(f\\"OVERRIDES input: {OVERRIDES}\\")' in notebook_text


def test_default_mn_sensitivity_param_grid_fixes_pqr_and_sweeps_nm():
    import scripts.run_gmm_mn_sensitivity_swiss_roll as runner

    grid = default_mn_sensitivity_param_grid()
    points = list(runner.ParameterGrid(grid))
    assert len(points) == 48
    for point in points:
        assert point["p_xy_paired_samples"] == P_XY_PAIRED_SAMPLES
        assert point["q_x_unpaired_samples"] == Q_X_UNPAIRED_SAMPLES
        assert point["r_y_unpaired_samples"] == R_Y_UNPAIRED_SAMPLES
        assert point["seed"] in SENSITIVITY_SEEDS
    nm_pairs = {(p["n_potentials"], p["m_potentials"]) for p in points}
    assert len(nm_pairs) == 16


def test_build_job_contains_n_m_overrides_for_snake_case_input():
    snake_case_point = {
        "experiment": "gmm-swiss-roll",
        "n_potentials": 64,
        "m_potentials": 32,
        "q_x_unpaired_samples": 1024,
        "r_y_unpaired_samples": 0,
        "p_xy_paired_samples": 128,
        "max_steps": 10000,
        "y_dim": 2,
    }
    job = build_job(snake_case_point)
    overrides = job["notebook_params"]["OVERRIDES"]
    assert "ebieot.model.n_potentials=64" in overrides
    assert "ebieot.cost.m_potentials=32" in overrides


def test_extract_effective_nm_from_overrides():
    notebook_params = {
        "EXPERIMENT": "gmm-swiss-roll",
        "OVERRIDES": [
            "ebieot.model.n_potentials=16",
            "ebieot.cost.m_potentials=8",
        ],
    }
    effective_n, effective_m = _extract_effective_nm(notebook_params)
    assert effective_n == 16
    assert effective_m == 8


def test_mlplse_cost_empty_hidden_channels():
    import torch

    from src.ebieot.costs.lse import MLPLSECost

    cost = MLPLSECost(
        log_v_m_hidden_channels=[],
        b_m_hidden_channels=[],
        m_potentials=5,
        x_dim=2,
        y_dim=2,
    )
    x = torch.randn(2)
    assert cost.compute_log_v_m(x).shape == (5,)
    assert cost.compute_b_m(x).shape == (5, 2)


def test_mlplse_cost_hidden_width_independent_of_m():
    import torch

    from src.ebieot.costs.lse import MLPLSECost

    cost_small_m = MLPLSECost(
        log_v_m_hidden_channels=[32, 32],
        b_m_hidden_channels=[32, 32],
        m_potentials=8,
        x_dim=2,
        y_dim=2,
    )
    cost_large_m = MLPLSECost(
        log_v_m_hidden_channels=[32, 32],
        b_m_hidden_channels=[32, 32],
        m_potentials=64,
        x_dim=2,
        y_dim=2,
    )
    x = torch.randn(2)
    assert cost_small_m.compute_log_v_m(x).shape == (8,)
    assert cost_large_m.compute_log_v_m(x).shape == (64,)
    assert cost_small_m.compute_b_m(x).shape == (8, 2)
    assert cost_large_m.compute_b_m(x).shape == (64, 2)
    # Backbone parameter counts should match (only heads differ).
    small_backbone_params = sum(
        p.numel() for p in cost_small_m._log_v_backbone.parameters()
    ) + sum(p.numel() for p in cost_small_m._b_backbone.parameters())
    large_backbone_params = sum(
        p.numel() for p in cost_large_m._log_v_backbone.parameters()
    ) + sum(p.numel() for p in cost_large_m._b_backbone.parameters())
    assert small_backbone_params == large_backbone_params


def test_to_notebook_grid_point_aliases_snake_case_keys():
    aliased = to_notebook_grid_point(
        {
            "n_potentials": 20,
            "m_potentials": 10,
            "q_x_unpaired_samples": 1024,
            "r_y_unpaired_samples": 0,
            "max_steps": 500,
            "y_dim": 2,
        }
    )
    assert aliased["nPotentials"] == 20
    assert aliased["mPotentials"] == 10
    assert aliased["qXUnpairedSamples"] == 1024
    assert aliased["rYUnpairedSamples"] == 0
    assert aliased["maxSteps"] == 500
    assert aliased["yDim"] == 2


def test_read_notebook_scraps_requires_effective_values(monkeypatch):
    class _FakeNB:
        scraps = {"target_metric": type("Scrap", (), {"data": 0.1})()}

    class _FakeSB:
        @staticmethod
        def read_notebook(_path):
            return _FakeNB()

    monkeypatch.setitem(__import__("sys").modules, "scrapbook", _FakeSB)
    with pytest.raises(KeyError):
        _read_notebook_scraps("dummy.ipynb")


def _mock_run_job_scraps(**overrides):
    scraps = {
        "mmd": 0.1,
        "sinkhorn": 0.2,
        "conditional_mmd": 0.15,
        "conditional_sinkhorn": 0.25,
        "effective_n": 8,
        "effective_m": 8,
    }
    scraps.update(overrides)
    return scraps


def _mock_run_job(
    monkeypatch,
    *,
    grid_point=None,
    overrides=None,
    scraps=None,
) -> dict:
    monkeypatch.setattr(
        "scripts.run_gmm_mn_sensitivity_swiss_roll._execute_notebook_with_kernel",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        "scripts.run_gmm_mn_sensitivity_swiss_roll._read_notebook_scraps",
        lambda _path: scraps or _mock_run_job_scraps(),
    )
    grid_point = grid_point or {
        "n_potentials": 8,
        "m_potentials": 8,
        "seed": 0,
    }
    job = {
        "grid_point": grid_point,
        "notebook_path": "dummy.ipynb",
        "output_path": "dummy_out.ipynb",
        "notebook_params": {
            "EXPERIMENT": "gmm-swiss-roll",
            "OVERRIDES": overrides
            or [
                "ebieot.model.n_potentials=8",
                "ebieot.cost.m_potentials=8",
            ],
        },
        "kernel_name": "python3",
    }
    return _run_job(job)


def test_build_job_applies_seed_override():
    job = build_job(
        {
            "n_potentials": 8,
            "m_potentials": 8,
            "seed": 1,
            "max_steps": 10000,
        }
    )
    assert "train.seed=1" in job["notebook_params"]["OVERRIDES"]
    assert "seed=1" in job["output_path"]


def test_notebook_params_for_seed_sets_train_seed_override():
    params = _notebook_params_for_seed(
        {
            "EXPERIMENT": "gmm-swiss-roll",
            "OVERRIDES": ["ebieot.model.n_potentials=8", "train.seed=99"],
        },
        1,
    )
    assert "train.seed=1" in params["OVERRIDES"]
    assert not any(item == "train.seed=99" for item in params["OVERRIDES"])


def test_build_job_uses_ebieot_gmm_notebook():
    job = build_job(
        {
            "experiment": "gmm-swiss-roll",
            "n_potentials": 8,
            "m_potentials": 8,
            "p_xy_paired_samples": 128,
            "max_steps": 10000,
        }
    )
    assert job["notebook_path"].endswith("notebooks/swiss_roll/ebieot_gmm.ipynb")


def test_run_job_ok_exports_all_four_metrics(monkeypatch):
    record = _mock_run_job(monkeypatch)
    assert record["status"] == "ok"
    assert record["mmd"] == 0.1
    assert record["sinkhorn"] == 0.2
    assert record["conditional_mmd"] == 0.15
    assert record["conditional_sinkhorn"] == 0.25
    assert record["override_mismatch"] is False


def test_run_job_fails_when_conditional_missing(monkeypatch):
    record = _mock_run_job(
        monkeypatch,
        scraps=_mock_run_job_scraps(conditional_mmd=None, conditional_sinkhorn=None),
    )
    assert record["status"] == "failed"
    assert "conditional metrics missing" in str(record["error"])


def test_save_records_writes_conditional_columns(tmp_path, monkeypatch):
    from scripts.run_gmm_mn_sensitivity_swiss_roll import save_records

    metrics_dir = tmp_path / "metrics"
    monkeypatch.setattr(
        "scripts.run_gmm_mn_sensitivity_swiss_roll._METRICS_DIR",
        metrics_dir,
    )
    records = [
        {
            "n_potentials": 8,
            "m_potentials": 8,
            "status": "ok",
            "mmd": 0.1,
            "sinkhorn": 0.2,
            "conditional_mmd": 0.15,
            "conditional_sinkhorn": 0.25,
        }
    ]
    _, csv_path = save_records(records, stem="test_metrics")
    csv_text = csv_path.read_text(encoding="utf-8")
    assert "conditional_mmd" in csv_text
    assert "conditional_sinkhorn" in csv_text
    assert "0.15" in csv_text
    assert "0.25" in csv_text


def test_run_job_fails_when_override_mismatch(monkeypatch):
    record = _mock_run_job(
        monkeypatch,
        overrides=[],
        scraps=_mock_run_job_scraps(effective_n=16, effective_m=8),
    )
    assert record["override_mismatch"] is True
    assert record["status"] == "failed"
    assert "override mismatch" in str(record["error"])


def test_plot_aggregate_records_averages_seeds(tmp_path):
    from scripts.plot_gmm_mn_sensitivity import aggregate_records_by_nm, load_raw_records

    csv_path = tmp_path / "metrics.csv"
    csv_path.write_text(
        "status,n_potentials,m_potentials,seed,mmd,sinkhorn,conditional_mmd,conditional_sinkhorn\n"
        "ok,8,8,0,0.10,0.20,0.30,0.40\n"
        "ok,8,8,1,0.20,0.30,0.40,0.50\n"
        "ok,8,8,2,0.30,0.40,0.50,0.60\n",
        encoding="utf-8",
    )
    raw = load_raw_records(csv_path)
    assert len(raw) == 3
    records = aggregate_records_by_nm(raw)
    assert len(records) == 1
    assert records[0]["n_seeds"] == 3
    assert records[0]["unconditional_mmd"] == pytest.approx(0.2)
    assert records[0]["unconditional_mmd_std"] == pytest.approx(0.1)


def test_plot_load_records_reads_unconditional_and_conditional(tmp_path):
    from scripts.plot_gmm_mn_sensitivity import available_metric_specs, load_records

    csv_path = tmp_path / "metrics.csv"
    csv_path.write_text(
        "status,n_potentials,m_potentials,seed,mmd,sinkhorn,conditional_mmd,conditional_sinkhorn\n"
        "ok,8,16,0,0.11,0.22,0.31,0.42\n"
        "ok,16,8,0,0.33,0.44,0.51,0.62\n",
        encoding="utf-8",
    )
    records = load_records(csv_path)
    assert len(records) == 2
    assert records[0]["unconditional_mmd"] == 0.11
    assert records[0]["unconditional_sinkhorn"] == 0.22
    assert records[0]["conditional_mmd"] == 0.31
    assert records[0]["conditional_sinkhorn"] == 0.42
    specs = available_metric_specs(records)
    assert ("conditional", "mmd", "conditional_mmd") in specs
    assert ("unconditional", "mmd", "unconditional_mmd") in specs


def test_plot_paired_metric_specs(tmp_path):
    from scripts.plot_gmm_mn_sensitivity import paired_metric_specs, load_records

    csv_path = tmp_path / "metrics.csv"
    csv_path.write_text(
        "status,n_potentials,m_potentials,seed,mmd,sinkhorn,conditional_mmd,conditional_sinkhorn\n"
        "ok,8,16,0,0.11,0.22,0.31,0.42\n"
        "ok,16,8,0,0.33,0.44,0.51,0.62\n",
        encoding="utf-8",
    )
    records = load_records(csv_path)
    pairs = paired_metric_specs(records)
    assert ("mmd", "unconditional_mmd", "conditional_mmd") in pairs
    assert ("sinkhorn", "unconditional_sinkhorn", "conditional_sinkhorn") in pairs


def test_plot_main_writes_unconditional_and_conditional_plots(tmp_path):
    import matplotlib

    matplotlib.use("Agg")

    from scripts.plot_gmm_mn_sensitivity import main

    csv_path = tmp_path / "metrics.csv"
    csv_path.write_text(
        "status,n_potentials,m_potentials,seed,mmd,sinkhorn,conditional_mmd,conditional_sinkhorn\n"
        "ok,8,16,0,0.11,0.22,0.31,0.42\n"
        "ok,16,8,0,0.33,0.44,0.51,0.62\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "plots"

    import sys
    from unittest.mock import patch

    with patch.object(
        sys,
        "argv",
        [
            "plot_gmm_mn_sensitivity.py",
            "--metrics-csv",
            str(csv_path),
            "--output-dir",
            str(output_dir),
        ],
    ):
        main()

    expected_outputs = [
        "mn_heatmap_unconditional_mmd.png",
        "mn_heatmap_std_unconditional_mmd.png",
        "mn_heatmap_unconditional_sinkhorn.png",
        "mn_heatmap_std_unconditional_sinkhorn.png",
        "mn_heatmap_conditional_mmd.png",
        "mn_heatmap_std_conditional_mmd.png",
        "mn_heatmap_conditional_sinkhorn.png",
        "mn_heatmap_std_conditional_sinkhorn.png",
        "metric_vs_n_unconditional_mmd.png",
        "metric_vs_n_unconditional_sinkhorn.png",
        "metric_vs_n_conditional_mmd.png",
        "metric_vs_n_conditional_sinkhorn.png",
        "metric_vs_m_unconditional_mmd.png",
        "metric_vs_m_unconditional_sinkhorn.png",
        "metric_vs_m_conditional_mmd.png",
        "metric_vs_m_conditional_sinkhorn.png",
        "mn_heatmap_paired_mmd.png",
        "mn_heatmap_paired_sinkhorn.png",
        "metric_vs_n_paired_mmd.png",
        "metric_vs_n_paired_sinkhorn.png",
        "metric_vs_m_paired_mmd.png",
        "metric_vs_m_paired_sinkhorn.png",
    ]
    for filename in expected_outputs:
        assert (output_dir / filename).exists(), f"missing plot: {filename}"

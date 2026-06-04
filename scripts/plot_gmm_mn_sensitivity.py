"""Plot aggregated M/N sensitivity metrics for Swiss-roll EBiEOT-GMM.

Reads per-seed rows from the sensitivity CSV, averages over seeds at plot time,
and can show standard deviations on line plots and std heatmaps.
"""


import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_METRICS_CSV = (
    _REPO_ROOT / "papermill-notebooks" / "metrics" / "gmm_mn_sensitivity.csv"
)
_DEFAULT_OUTPUT_DIR = _REPO_ROOT / "plots" / "Swiss_Roll" / "gmm_mn_sensitivity"

# (regime label, metric name, CSV column keys to try)
METRIC_SPECS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("unconditional", "mmd", ("mmd", "mmd_metric", "metric_value")),
    ("unconditional", "sinkhorn", ("sinkhorn", "sinkhorn_metric")),
    ("conditional", "mmd", ("conditional_mmd", "conditional_mmd_metric")),
    (
        "conditional",
        "sinkhorn",
        ("conditional_sinkhorn", "conditional_sinkhorn_metric"),
    ),
)


def _parse_int(record: dict[str, Any], *keys: str) -> int:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return int(record[key])
    raise KeyError(f"None of keys {keys} found in record")


def _parse_float(row: dict[str, Any], *keys: str) -> float:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return float(value)
    raise KeyError(f"None of keys {keys} found in record")


def _parse_optional_float(row: dict[str, Any], *keys: str) -> float | None:
    try:
        return _parse_float(row, *keys)
    except KeyError:
        return None


def load_raw_records(csv_path: Path) -> list[dict[str, Any]]:
    """Load one dict per successful CSV row (including per-seed runs)."""
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    records: list[dict[str, Any]] = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        entry: dict[str, Any] = {
            "n_potentials": _parse_int(row, "n_potentials", "nPotentials"),
            "m_potentials": _parse_int(row, "m_potentials", "mPotentials"),
        }
        if row.get("seed") not in (None, ""):
            entry["seed"] = _parse_int(row, "seed")
        for _regime, metric_name, column_keys in METRIC_SPECS:
            value = _parse_optional_float(row, *column_keys)
            if value is not None:
                entry[f"{_regime}_{metric_name}"] = value
        if not any(key in entry for key in ("unconditional_mmd", "conditional_mmd")):
            continue
        records.append(entry)
    if not records:
        raise ValueError(
            "No successful records with unconditional or conditional MMD found in metrics CSV."
        )
    return records


def aggregate_records_by_nm(
    raw_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Average metrics over seeds for each (N, M); attach ``{metric}_std`` fields."""
    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for rec in raw_records:
        groups[(rec["n_potentials"], rec["m_potentials"])].append(rec)

    aggregated: list[dict[str, Any]] = []
    for (n_potentials, m_potentials), group in sorted(groups.items()):
        entry: dict[str, Any] = {
            "n_potentials": n_potentials,
            "m_potentials": m_potentials,
            "n_seeds": len(group),
        }
        for regime, metric_name, _column_keys in METRIC_SPECS:
            record_key = f"{regime}_{metric_name}"
            values = [float(rec[record_key]) for rec in group if record_key in rec]
            if not values:
                continue
            entry[record_key] = float(np.mean(values))
            entry[f"{record_key}_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            )
        if not any(key in entry for key in ("unconditional_mmd", "conditional_mmd")):
            continue
        aggregated.append(entry)
    if not aggregated:
        raise ValueError(
            "No (N, M) groups with plottable metrics after seed aggregation."
        )
    return aggregated


def load_records(csv_path: Path) -> list[dict[str, Any]]:
    """Load CSV and return seed-averaged records for plotting."""
    return aggregate_records_by_nm(load_raw_records(csv_path))


def _std_record_key(record_key: str) -> str:
    return f"{record_key}_std"


def available_metric_specs(records: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    """Return (regime, metric_name, record_key) specs present in loaded records."""
    available: list[tuple[str, str, str]] = []
    for regime, metric_name, _column_keys in METRIC_SPECS:
        record_key = f"{regime}_{metric_name}"
        if any(record_key in rec for rec in records):
            available.append((regime, metric_name, record_key))
    return available


def paired_metric_specs(
    records: list[dict[str, Any]],
) -> list[tuple[str, str, str]]:
    """Return (metric_name, uncond_key, cond_key) when both regimes are present."""
    available = available_metric_specs(records)
    by_metric: dict[str, dict[str, str]] = {}
    for regime, metric_name, record_key in available:
        by_metric.setdefault(metric_name, {})[regime] = record_key

    pairs: list[tuple[str, str, str]] = []
    for metric_name, regimes in by_metric.items():
        uncond_key = regimes.get("unconditional")
        cond_key = regimes.get("conditional")
        if uncond_key and cond_key:
            pairs.append((metric_name, uncond_key, cond_key))
    return pairs


def _full_grid_axes(records: list[dict[str, Any]]) -> tuple[list[int], list[int]]:
    n_values = sorted({rec["n_potentials"] for rec in records})
    m_values = sorted({rec["m_potentials"] for rec in records})
    return n_values, m_values


def _build_grid(
    records: list[dict[str, Any]],
    record_key: str,
    n_values: list[int],
    m_values: list[int],
) -> np.ndarray:
    grid = np.full((len(m_values), len(n_values)), np.nan, dtype=float)
    n_index = {n: idx for idx, n in enumerate(n_values)}
    m_index = {m: idx for idx, m in enumerate(m_values)}
    for rec in records:
        if record_key not in rec:
            continue
        grid[m_index[rec["m_potentials"]], n_index[rec["n_potentials"]]] = rec[
            record_key
        ]
    return grid


def _records_by_nm(
    records: list[dict[str, Any]],
) -> dict[tuple[int, int], dict[str, Any]]:
    return {(rec["m_potentials"], rec["n_potentials"]): rec for rec in records}


def plot_heatmap(
    records: list[dict[str, Any]],
    regime: str,
    metric_name: str,
    record_key: str,
    output_dir: Path,
) -> Path:
    n_values, m_values = _full_grid_axes(records)
    grid = _build_grid(records, record_key, n_values, m_values)

    fig, ax = plt.subplots(figsize=(8, 6))
    masked = np.ma.masked_invalid(grid)
    image = ax.imshow(masked, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(n_values)), [str(n) for n in n_values])
    ax.set_yticks(range(len(m_values)), [str(m) for m in m_values])
    ax.set_xlabel("N")
    ax.set_ylabel("M")
    metric_to_plot = {"sinkhorn": r"$\mathcal{W_2$", "mmd": "MMD"}
    regime_to_plot = {"unconditional": "Unconditional", "conditional": "Conditional"}
    # ax.set_title(f"EBiEOT-GMM Sensitivity — {regime_to_plot[regime]} {metric_to_plot[metric_name]} (lower is better)")
    ax.set_title(
        f"{regime_to_plot[regime]} {metric_to_plot[metric_name]} (lower is better)"
    )
    # fig.colorbar(image, ax=ax, label=f"{regime}{metric_name}")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()

    output_path = output_dir / f"mn_heatmap_{regime}_{metric_name}.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_heatmap_std(
    records: list[dict[str, Any]],
    regime: str,
    metric_name: str,
    record_key: str,
    output_dir: Path,
) -> Path | None:
    std_key = _std_record_key(record_key)
    if not any(std_key in rec for rec in records):
        return None

    n_values, m_values = _full_grid_axes(records)
    grid = _build_grid(records, std_key, n_values, m_values)

    fig, ax = plt.subplots(figsize=(8, 6))
    masked = np.ma.masked_invalid(grid)
    image = ax.imshow(masked, aspect="auto", origin="lower", cmap="magma")
    ax.set_xticks(range(len(n_values)), [str(n) for n in n_values])
    ax.set_yticks(range(len(m_values)), [str(m) for m in m_values])
    ax.set_xlabel("N")
    ax.set_ylabel("M")
    ax.set_title(f"EBiEOT-GMM Sensitivity — {regime} {metric_name} std across seeds")
    fig.colorbar(image, ax=ax, label=std_key)
    fig.tight_layout()

    output_path = output_dir / f"mn_heatmap_std_{regime}_{metric_name}.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_metric_vs_n(
    records: list[dict[str, Any]],
    regime: str,
    metric_name: str,
    record_key: str,
    output_dir: Path,
) -> Path:
    n_values, m_values = _full_grid_axes(records)
    by_m_then_n = _records_by_nm(records)
    std_key = _std_record_key(record_key)

    fig, ax = plt.subplots(figsize=(8, 5))
    for m in m_values:
        xs = [
            n
            for n in n_values
            if (m, n) in by_m_then_n and record_key in by_m_then_n[(m, n)]
        ]
        if not xs:
            continue
        ys = [by_m_then_n[(m, n)][record_key] for n in xs]
        yerr = [by_m_then_n[(m, n)].get(std_key, 0.0) for n in xs]
        ax.errorbar(xs, ys, yerr=yerr, marker="o", capsize=3, label=f"M={m}")

    ax.set_xlabel("N")
    ax.set_ylabel(f"{regime} {metric_name}")
    ax.set_title(f"Sensitivity by N — {regime} {metric_name} (mean ± std over seeds)")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()

    output_path = output_dir / f"metric_vs_n_{regime}_{metric_name}.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_metric_vs_m(
    records: list[dict[str, Any]],
    regime: str,
    metric_name: str,
    record_key: str,
    output_dir: Path,
) -> Path:
    n_values, m_values = _full_grid_axes(records)
    by_n_then_m = {(rec["n_potentials"], rec["m_potentials"]): rec for rec in records}
    std_key = _std_record_key(record_key)

    fig, ax = plt.subplots(figsize=(8, 5))
    for n in n_values:
        xs = [
            m
            for m in m_values
            if (n, m) in by_n_then_m and record_key in by_n_then_m[(n, m)]
        ]
        if not xs:
            continue
        ys = [by_n_then_m[(n, m)][record_key] for m in xs]
        yerr = [by_n_then_m[(n, m)].get(std_key, 0.0) for m in xs]
        ax.errorbar(xs, ys, yerr=yerr, marker="o", capsize=3, label=f"N={n}")

    ax.set_xlabel("M")
    ax.set_ylabel(f"{regime} {metric_name}")
    ax.set_title(f"Sensitivity by M — {regime} {metric_name} (mean ± std over seeds)")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()

    output_path = output_dir / f"metric_vs_m_{regime}_{metric_name}.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_heatmap_paired(
    records: list[dict[str, Any]],
    metric_name: str,
    uncond_key: str,
    cond_key: str,
    output_dir: Path,
) -> Path:
    n_values, m_values = _full_grid_axes(records)
    uncond_grid = _build_grid(records, uncond_key, n_values, m_values)
    cond_grid = _build_grid(records, cond_key, n_values, m_values)

    combined = np.concatenate([uncond_grid.ravel(), cond_grid.ravel()])
    finite = combined[np.isfinite(combined)]
    vmin = float(finite.min()) if finite.size else None
    vmax = float(finite.max()) if finite.size else None

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, grid, regime in (
        (axes[0], uncond_grid, "unconditional"),
        (axes[1], cond_grid, "conditional"),
    ):
        masked = np.ma.masked_invalid(grid)
        image = ax.imshow(
            masked,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xticks(range(len(n_values)), [str(n) for n in n_values])
        ax.set_yticks(range(len(m_values)), [str(m) for m in m_values])
        ax.set_xlabel("N")
        ax.set_ylabel("M")
        ax.set_title(f"{regime} {metric_name}")
        fig.colorbar(image, ax=ax, label=f"{regime}_{metric_name}")

    fig.suptitle(
        f"EBiEOT-GMM Sensitivity — {metric_name} (lower is better)",
        y=1.02,
    )
    fig.tight_layout()

    output_path = output_dir / f"mn_heatmap_paired_{metric_name}.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_metric_vs_n_paired(
    records: list[dict[str, Any]],
    metric_name: str,
    uncond_key: str,
    cond_key: str,
    output_dir: Path,
) -> Path:
    n_values, m_values = _full_grid_axes(records)
    by_m_then_n = _records_by_nm(records)
    uncond_std = _std_record_key(uncond_key)
    cond_std = _std_record_key(cond_key)

    fig, ax = plt.subplots(figsize=(9, 5))
    for m in m_values:
        xs = [n for n in n_values if (m, n) in by_m_then_n]
        if not xs:
            continue
        uncond_ys = [by_m_then_n[(m, n)].get(uncond_key, np.nan) for n in xs]
        cond_ys = [by_m_then_n[(m, n)].get(cond_key, np.nan) for n in xs]
        uncond_err = [by_m_then_n[(m, n)].get(uncond_std, 0.0) for n in xs]
        cond_err = [by_m_then_n[(m, n)].get(cond_std, 0.0) for n in xs]
        ax.errorbar(
            xs,
            uncond_ys,
            yerr=uncond_err,
            marker="o",
            linestyle="-",
            capsize=3,
            label=f"M={m} unconditional",
        )
        ax.errorbar(
            xs,
            cond_ys,
            yerr=cond_err,
            marker="s",
            linestyle="--",
            capsize=3,
            label=f"M={m} conditional",
        )

    ax.set_xlabel("N")
    ax.set_ylabel(metric_name)
    ax.set_title(
        f"Sensitivity by N — unconditional vs conditional {metric_name} (mean ± std)"
    )
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()

    output_path = output_dir / f"metric_vs_n_paired_{metric_name}.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_metric_vs_m_paired(
    records: list[dict[str, Any]],
    metric_name: str,
    uncond_key: str,
    cond_key: str,
    output_dir: Path,
) -> Path:
    n_values, m_values = _full_grid_axes(records)
    by_n_then_m = {(rec["n_potentials"], rec["m_potentials"]): rec for rec in records}
    uncond_std = _std_record_key(uncond_key)
    cond_std = _std_record_key(cond_key)

    fig, ax = plt.subplots(figsize=(9, 5))
    for n in n_values:
        xs = [m for m in m_values if (n, m) in by_n_then_m]
        if not xs:
            continue
        uncond_ys = [by_n_then_m[(n, m)].get(uncond_key, np.nan) for m in xs]
        cond_ys = [by_n_then_m[(n, m)].get(cond_key, np.nan) for m in xs]
        uncond_err = [by_n_then_m[(n, m)].get(uncond_std, 0.0) for m in xs]
        cond_err = [by_n_then_m[(n, m)].get(cond_std, 0.0) for m in xs]
        ax.errorbar(
            xs,
            uncond_ys,
            yerr=uncond_err,
            marker="o",
            linestyle="-",
            capsize=3,
            label=f"N={n} unconditional",
        )
        ax.errorbar(
            xs,
            cond_ys,
            yerr=cond_err,
            marker="s",
            linestyle="--",
            capsize=3,
            label=f"N={n} conditional",
        )

    ax.set_xlabel("M")
    ax.set_ylabel(metric_name)
    ax.set_title(
        f"Sensitivity by M — unconditional vs conditional {metric_name} (mean ± std)"
    )
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()

    output_path = output_dir / f"metric_vs_m_paired_{metric_name}.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=_DEFAULT_METRICS_CSV,
        help="Path to gmm_mn_sensitivity.csv file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help="Directory where PNG plots are saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.metrics_csv.exists():
        raise FileNotFoundError(f"Metrics CSV does not exist: {args.metrics_csv}")

    records = load_records(args.metrics_csv)
    specs = available_metric_specs(records)
    if not specs:
        raise ValueError("No plottable metric columns found in metrics CSV.")

    outputs: list[Path] = []
    for regime, metric_name, record_key in specs:
        outputs.append(
            plot_heatmap(records, regime, metric_name, record_key, output_dir)
        )
        std_heatmap = plot_heatmap_std(
            records, regime, metric_name, record_key, output_dir
        )
        if std_heatmap is not None:
            outputs.append(std_heatmap)
        outputs.extend(
            [
                plot_metric_vs_n(records, regime, metric_name, record_key, output_dir),
                plot_metric_vs_m(records, regime, metric_name, record_key, output_dir),
            ]
        )

    for metric_name, uncond_key, cond_key in paired_metric_specs(records):
        outputs.extend(
            [
                plot_heatmap_paired(
                    records, metric_name, uncond_key, cond_key, output_dir
                ),
                plot_metric_vs_n_paired(
                    records, metric_name, uncond_key, cond_key, output_dir
                ),
                plot_metric_vs_m_paired(
                    records, metric_name, uncond_key, cond_key, output_dir
                ),
            ]
        )

    print("Saved plots:")
    for path in outputs:
        print(f"- {path}")


if __name__ == "__main__":
    main()


from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class NotebookRunResult:
    output_path: str
    metric_name: str
    metric_value: float | None
    status: str
    error: str | None = None
    params: dict[str, Any] | None = None

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


def execute_notebook(
    notebook_input_path: str,
    notebook_output_path: str,
    parameters: None | dict[str, Any] = None,
    progress_bar: bool = False,
) -> str:
    import papermill as pm

    pm.execute_notebook(
        input_path=notebook_input_path,
        output_path=notebook_output_path,
        parameters=parameters,
        progress_bar=progress_bar,
    )
    return notebook_output_path


def read_metric_from_notebook(notebook_path: str, metric_name: str = "target_metric") -> float:
    import scrapbook as sb

    notebook = sb.read_notebook(notebook_path)
    if metric_name not in notebook.scraps:
        raise KeyError(f"Metric '{metric_name}' was not found in notebook scraps.")
    return float(notebook.scraps[metric_name].data)


def run_notebook_and_collect_metric(
    notebook_input_path: str,
    notebook_output_path: str,
    parameters: None | dict[str, Any] = None,
    metric_name: str = "target_metric",
    progress_bar: bool = False,
) -> NotebookRunResult:
    try:
        execute_notebook(
            notebook_input_path=notebook_input_path,
            notebook_output_path=notebook_output_path,
            parameters=parameters,
            progress_bar=progress_bar,
        )
        metric_value = read_metric_from_notebook(notebook_output_path, metric_name=metric_name)
    except Exception as exc:  # noqa: BLE001
        return NotebookRunResult(
            output_path=notebook_output_path,
            metric_name=metric_name,
            metric_value=None,
            status="failed",
            error=str(exc),
            params=parameters,
        )

    return NotebookRunResult(
        output_path=notebook_output_path,
        metric_name=metric_name,
        metric_value=metric_value,
        status="ok",
        error=None,
        params=parameters,
    )


def write_results_csv(records: list[dict[str, Any]], output_csv_path: Path) -> None:
    import csv

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        output_csv_path.write_text("", encoding="utf-8")
        return

    field_names = sorted({key for record in records for key in record.keys()})
    with output_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(records)

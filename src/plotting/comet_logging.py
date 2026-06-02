from __future__ import annotations

from typing import Any

from matplotlib import pyplot as plt


def log_matplotlib_figure(experiment: Any | None, figure_name: str, fig: Any, step: int | None = None) -> None:
    if experiment is None:
        return
    experiment.log_figure(figure_name=figure_name, figure=fig, step=step)
    plt.close(fig)


def log_plotly_figure(experiment: Any | None, figure_name: str, fig: Any, step: int | None = None) -> None:
    if experiment is None:
        return
    experiment.log_figure(figure_name=figure_name, figure=fig, step=step)

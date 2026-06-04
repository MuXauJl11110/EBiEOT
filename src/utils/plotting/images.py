"""Image-grid plotting for colored MNIST notebooks."""

from collections.abc import Sequence
from typing import Any, Protocol

import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils

from src.utils.datasets.colored_mnist import unflatten_images


class FigureLogger(Protocol):
    def log_figure(
        self,
        figure_name: str,
        figure: Any,
        *,
        step: int | None = None,
    ) -> None: ...


def to_image_batch(
    images: torch.Tensor,
    *,
    channels: int = 3,
    img_size: int = 32,
) -> torch.Tensor:
    if images.dim() == 2:
        return unflatten_images(images, channels=channels, img_size=img_size)
    return images


def to_display(
    images: torch.Tensor,
    *,
    value_range: tuple[float, float],
) -> torch.Tensor:
    """Map a ``(N, C, H, W)`` batch to ``[0, 1]`` for matplotlib."""
    low, high = value_range
    if high <= low:
        raise ValueError(f"value_range must satisfy low < high, got {value_range}")

    batch = images.detach().float().cpu()
    if batch.dim() == 3:
        batch = batch.unsqueeze(0)
    if batch.shape[1] == 1:
        batch = batch.repeat(1, 3, 1, 1)
    return batch.clamp(low, high).sub(low).div(high - low)


def _finalize_figure(
    fig: plt.Figure,
    *,
    figure_name: str,
    experiment: FigureLogger | None,
    step: int | None,
) -> None:
    if experiment is not None:
        kwargs: dict[str, Any] = {"figure_name": figure_name, "figure": fig}
        if step is not None:
            kwargs["step"] = step
        experiment.log_figure(**kwargs)
        plt.close(fig)
    else:
        plt.show()


def plot_image_grid(
    images: torch.Tensor,
    *,
    nrow: int = 8,
    title: str = "",
    channels: int = 3,
    img_size: int = 32,
    value_range: tuple[float, float] = (0.0, 1.0),
    experiment: FigureLogger | None = None,
    step: int | None = None,
) -> None:
    """Plot a grid of images; ``value_range=None`` infers scaling from the batch."""
    batch = to_image_batch(images, channels=channels, img_size=img_size)
    display = to_display(batch, value_range=value_range)
    grid = vutils.make_grid(display, nrow=nrow, normalize=False, pad_value=0.0)
    nrows = max(1, (display.shape[0] + nrow - 1) // nrow)
    fig, ax = plt.subplots(figsize=(min(16, nrow * 1.5), max(3, nrows * 1.5)))
    ax.imshow(grid.permute(1, 2, 0).numpy(), vmin=0.0, vmax=1.0)
    ax.axis("off")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    _finalize_figure(
        fig,
        figure_name=title or "image_grid",
        experiment=experiment,
        step=step,
    )


def plot_image_grids_vertical(
    images_list: Sequence[torch.Tensor],
    titles: Sequence[str] | None = None,
    *,
    nrow: int = 8,
    channels: int = 3,
    img_size: int = 32,
    value_range: tuple[float, float] = (0.0, 1.0),
    experiment: FigureLogger | None = None,
    step: int | None = None,
) -> None:
    """Stack several image grids vertically in one figure."""
    if not images_list:
        raise ValueError("images_list must be non-empty")

    n_panels = len(images_list)
    if titles is None:
        titles = [""] * n_panels
    elif len(titles) != n_panels:
        raise ValueError(f"titles length ({len(titles)}) must match images_list length ({n_panels})")

    batches = [to_image_batch(images, channels=channels, img_size=img_size) for images in images_list]

    width = min(16, nrow * 1.5)
    height_per_panel = 3.0
    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(width, height_per_panel * n_panels),
        squeeze=False,
    )

    for ax, batch, panel_title in zip(axes.flatten(), batches, titles, strict=True):
        display = to_display(batch, value_range=value_range)
        grid = vutils.make_grid(display, nrow=nrow, normalize=False, pad_value=0.0)
        ax.imshow(grid.permute(1, 2, 0).numpy(), vmin=0.0, vmax=1.0)
        ax.axis("off")
        if panel_title:
            ax.set_title(panel_title)

    fig.tight_layout()
    figure_name = titles[0] if titles and titles[0] else "image_grids_vertical"
    _finalize_figure(fig, figure_name=figure_name, experiment=experiment, step=step)


def plot_transport_pairs(
    source: torch.Tensor,
    target: torch.Tensor,
    moved: torch.Tensor,
    *,
    num_show: int = 6,
    channels: int = 3,
    img_size: int = 32,
    value_range: tuple[float, float] = (0.0, 1.0),
    experiment: FigureLogger | None = None,
    step: int | None = None,
) -> None:
    """Rows of source | model output | target for qualitative transport."""
    n = min(num_show, source.shape[0])
    src = to_image_batch(source[:n], channels=channels, img_size=img_size)
    tgt = to_image_batch(target[:n], channels=channels, img_size=img_size)
    mov = to_image_batch(moved[:n], channels=channels, img_size=img_size)
    triplets = torch.cat([src, mov, tgt], dim=0)
    plot_image_grid(
        triplets,
        nrow=n,
        title="source | transport | target",
        channels=channels,
        img_size=img_size,
        value_range=value_range,
        experiment=experiment,
        step=step,
    )

import os
import random

import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn as nn
from comet_ml import Experiment
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

from src.utils.samplers.base import Sampler


def plot_gaussians(
    model: nn.Module,
    X_sampler: Sampler,
    Y_sampler: Sampler,
    X_paired: torch.Tensor,
    Y_paired: torch.Tensor,
    num_samples: int = 256,
    experiment: Experiment | None = None,
) -> None:
    num_gaussians = len(X_sampler.mu)
    colors = cm.rainbow(np.linspace(0, 1, num_gaussians))
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=200)

    for ax in axes:
        ax.grid(zorder=-20)

    x_samples = X_sampler.sample(num_samples)
    y_samples = Y_sampler.sample(num_samples)

    y_pred = model(x_samples).cpu().numpy()
    for i in range(num_gaussians):
        indices = np.arange(i, len(x_samples), num_gaussians)
        # First plot
        axes[0].scatter(
            x_samples[indices, 0].cpu().numpy(),
            x_samples[indices, 1].cpu().numpy(),
            alpha=0.3,
            color=colors[i],
            s=32,
            edgecolors="black",
        )
        # Second plot
        axes[1].scatter(
            y_samples[indices, 0].cpu().numpy(),
            y_samples[indices, 1].cpu().numpy(),
            alpha=0.3,
            color=colors[i],
            s=32,
            edgecolors="black",
        )
        # Third plot
        axes[2].scatter(
            y_pred[indices, 0],
            y_pred[indices, 1],
            alpha=0.3,
            color=colors[i],
            s=32,
            edgecolors="black",
        )
    axes[0].set_title(label=r"Input distribution $p_0$")
    axes[1].set_title(label=r"Target distribution $p_1$")
    axes[2].set_title(label=r"Fitted distribution")

    pair_colors = cm.rainbow(np.linspace(0, 1, len(X_paired)))
    for color, x, y in zip(pair_colors, X_paired.cpu().numpy(), Y_paired.cpu().numpy()):
        axes[3].scatter(x[0], x[1], color=color, s=32, edgecolors="black")
        axes[3].scatter(y[0], y[1], color=color, s=32, edgecolors="black")
        axes[3].arrow(x[0], x[1], y[0] - x[0], y[1] - x[1], color=color)
    axes[3].set_title(label=r"Pairs")

    fig.tight_layout(pad=0.1)

    if experiment is not None:
        experiment.log_figure(figure_name="Distribution", figure=fig)
        plt.close(fig)
    else:
        plt.show()


def plot_PCA(
    model: nn.Module,
    source_data: torch.Tensor,
    target_data: torch.Tensor,
    paired_source_data: torch.Tensor,
    paired_target_data: torch.Tensor,
    lims: tuple[tuple] = ((-25, 50), (-25, 30)),
    experiment: Experiment | None = None,
) -> None:
    fig, axes = plt.subplots(
        1, 3, figsize=(12, 4), squeeze=True, sharex=True, sharey=True
    )
    pca = PCA(n_components=2).fit(target_data.cpu().numpy())

    source_data_pca = pca.transform(source_data.cpu().numpy())
    target_data_pca = pca.transform(target_data.cpu().numpy())

    # First plot
    axes[0].scatter(
        source_data_pca[:, 0],
        source_data_pca[:, 1],
        c="g",
        edgecolor="black",
        label=r"$x\sim P_0(x)$",
        s=30,
    )
    # Second plot
    axes[1].scatter(
        target_data_pca[:, 0],
        target_data_pca[:, 1],
        c="orange",
        edgecolor="black",
        label=r"$x\sim P_1(x)$",
        s=30,
    )

    paired_source_data_pca = pca.transform(paired_source_data.cpu().numpy())
    paired_target_data_pca = pca.transform(paired_target_data.cpu().numpy())
    pred_data = model(paired_source_data).cpu().numpy()
    pred_data_pca = pca.transform(pred_data)
    axes[2].scatter(
        paired_source_data_pca[:, 0],
        paired_source_data_pca[:, 1],
        c="g",
        edgecolor="black",
        label=r"$x\sim P_0(x)$",
        s=30,
    )
    axes[2].scatter(
        paired_target_data_pca[:, 0],
        paired_target_data_pca[:, 1],
        c="orange",
        edgecolor="black",
        label=r"$x\sim P_1(x)$",
        s=30,
    )
    axes[2].scatter(
        pred_data_pca[:, 0],
        pred_data_pca[:, 1],
        c="yellow",
        edgecolor="black",
        label=r"$x\sim T(x)$",
        s=30,
    )
    for source_point, target_point, pred_point in zip(
        paired_source_data_pca, paired_target_data_pca, pred_data_pca
    ):
        axes[2].arrow(
            source_point[0],
            source_point[1],
            target_point[0] - source_point[0],
            target_point[1] - source_point[1],
            edgecolor="g",
        )
        axes[2].arrow(
            source_point[0],
            source_point[1],
            pred_point[0] - source_point[0],
            pred_point[1] - source_point[1],
            edgecolor="r",
        )

    for i in range(3):
        axes[i].grid()
        axes[i].set_xlim(lims[0])
        axes[i].set_ylim(lims[1])
        axes[i].legend()

    fig.tight_layout(pad=0.5)

    if experiment is not None:
        experiment.log_figure(figure_name="PCA samples", figure=fig)
        plt.close(fig)
    else:
        plt.show()


def plot_swiss_roll(
    models_dict: dict[str, torch.nn.Module],
    X_sampler: Sampler,
    Y_sampler: Sampler,
    X_paired: torch.Tensor,
    Y_paired: torch.Tensor,
    starting_points: torch.Tensor,
    gt_Y_points: torch.Tensor,
    num_ending_points: int = 64,
    num_samples: int = 1024,
    x_lim: tuple[float, float] = (-2.5, 2.5),
    y_lim: tuple[float, float] = (-2.5, 2.5),
    arrows_num: int = 8,
    experiment: Experiment | None = None,
    save_dir: str | None = None,
) -> None:
    num_starting_points = len(starting_points)
    colors = cm.rainbow(np.linspace(0.1, 0.9, num_starting_points))
    num_subplots = 3 + len(models_dict)
    fig, axes = plt.subplots(
        1, num_subplots, figsize=(3.75 * num_subplots, 3.75), dpi=200
    )
    save_filenames = []

    for ax in axes:
        ax.grid(zorder=-20)

    x_samples = X_sampler.sample(num_samples)
    y_samples = Y_sampler.sample(num_samples)

    # First plot
    for x, y in zip(X_paired.cpu().numpy(), Y_paired.cpu().numpy()):
        axes[1].arrow(x[0], x[1], y[0] - x[0], y[1] - x[1], color="black")
    axes[0].scatter(
        x_samples[:, 0].cpu().numpy(),
        x_samples[:, 1].cpu().numpy(),
        # alpha=0.3,
        c="g",
        s=32,
        edgecolors="black",
        label=r"Input distribution $\pi^*_x$",
    )
    axes[0].scatter(
        y_samples[:, 0].cpu().numpy(),
        y_samples[:, 1].cpu().numpy(),
        c="orange",
        s=32,
        edgecolors="black",
        label=r"Target distribution $\pi^*_y$",
    )
    save_filenames.append("source_target")
    # Second plot
    axes[1].scatter(
        X_paired[:, 0].cpu().numpy(),
        X_paired[:, 1].cpu().numpy(),
        # alpha=0.3,
        c="g",
        s=32,
        edgecolors="black",
        zorder=2,
        label=r"Input paired samples $x \sim \pi^*_x$",
    )
    axes[1].scatter(
        Y_paired[:, 0].cpu().numpy(),
        Y_paired[:, 1].cpu().numpy(),
        c="orange",
        s=32,
        edgecolors="black",
        zorder=2,
        label=r"Target paired samples $y \sim \pi^*_y$",
    )
    save_filenames.append("paired_data")

    # Third plot
    axes[2].scatter(
        y_samples[:, 0].cpu().numpy(),
        y_samples[:, 1].cpu().numpy(),
        c="orange",
        s=32,
        edgecolors="black",
    )
    default_legend = Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="orange",
        markeredgecolor="black",
        markersize=8,
    )
    legend_start, legend_end = [], []
    for color, point, gt_point in zip(colors, starting_points, gt_Y_points):
        label = f"{point.cpu().numpy()}"
        axes[2].scatter(
            point[0].item(),
            point[1].item(),
            color=color,
            label=label,
            s=32,
            zorder=3,
            edgecolors="black",
            marker="s",
        )
        axes[2].scatter(
            gt_point[:, 0].cpu().numpy(),
            gt_point[:, 1].cpu().numpy(),
            color=color,
            s=32,
            zorder=3,
            edgecolors="black",
            marker="d",
        )
        indices = random.choices(range(gt_point.shape[0]), k=arrows_num)
        for y in gt_point[indices]:
            axes[2].arrow(
                point[0].item(),
                point[1].item(),
                y[0].item() - point[0].item(),
                y[1].item() - point[1].item(),
                color="black",
            )
        legend_start.append(
            Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor=color,
                markeredgecolor="black",
                markersize=8,
            )
        )
        legend_end.append(
            Line2D(
                [0],
                [0],
                marker="d",
                color="w",
                markerfacecolor=color,
                markeredgecolor="black",
                markersize=8,
            )
        )
    axes[2].legend(
        [default_legend, tuple(legend_start), tuple(legend_end)],
        [
            r"Target distribution $\pi^*_y$",
            r"Source samples $x\sim \pi^*_x$",
            r"Ground-truth samples $y \sim \pi^\star(\cdot\vert x)$",
        ],
        handler_map={tuple: HandlerTuple(ndivide=None, pad=1)},
        loc="lower left",
        prop={"size": 9},
    )
    axes[2].set_xlim(x_lim)
    axes[2].set_ylim(y_lim)
    save_filenames.append("gt_mapping")

    # Last plots
    for i, (title, model) in enumerate(models_dict.items()):
        ax_index = 3 + i

        y_pred = model(x_samples).cpu().numpy()
        axes[ax_index].scatter(
            y_pred[:, 0], y_pred[:, 1], c="yellow", s=32, edgecolors="black"
        )
        default_legend = Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="yellow",
            markeredgecolor="black",
            markersize=8,
        )

        legend_start, legend_end = [], []
        for color, point in zip(colors, starting_points):
            axes[ax_index].scatter(
                point[0].item(),
                point[1].item(),
                color=color,
                s=32,
                zorder=3,
                edgecolors="black",
                marker="s",
            )

            repeated_starting_points = point[None, :].repeat(num_ending_points, 1)
            point_pred = model(repeated_starting_points).cpu().numpy()
            axes[ax_index].scatter(
                point_pred[:, 0],
                point_pred[:, 1],
                color=color,
                s=32,
                zorder=3,
                edgecolors="black",
                marker="d",
            )

            indices = random.choices(range(num_ending_points), k=arrows_num)
            for y in point_pred[indices]:
                axes[ax_index].arrow(
                    point[0].item(),
                    point[1].item(),
                    y[0] - point[0].item(),
                    y[1] - point[1].item(),
                    color="black",
                    width=0.003,
                )
            legend_start.append(
                Line2D(
                    [0],
                    [0],
                    marker="s",
                    color="w",
                    markerfacecolor=color,
                    markeredgecolor="black",
                    markersize=8,
                )
            )
            legend_end.append(
                Line2D(
                    [0],
                    [0],
                    marker="d",
                    color="w",
                    markerfacecolor=color,
                    markeredgecolor="black",
                    markersize=8,
                )
            )

        axes[ax_index].legend(
            [default_legend, tuple(legend_start), tuple(legend_end)],
            [
                r"Fitted distribution $\pi^\theta_y$",
                r"Source samples $x\sim \pi^*_x$",
                r"Conditional samples $y \sim \pi^\theta(\cdot\vert x)$",
            ],
            handler_map={tuple: HandlerTuple(ndivide=None, pad=1)},
            loc="lower left",
            prop={"size": 9},
        )
        axes[ax_index].set_xlim(x_lim)
        axes[ax_index].set_ylim(y_lim)
        save_filenames.append("Light-IOT_" + title)

    for _, ax in enumerate(axes[:2]):
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.legend(loc="lower left")

    fig.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        # Save each subplot
        for filename, ax in zip(save_filenames, axes):
            extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            filename = os.path.join(save_dir, f"{filename}.png")
            fig.savefig(filename, bbox_inches=extent.expanded(1.2, 1.2))
            print(f"Saved {filename}")

    if experiment is not None:
        experiment.log_figure(figure_name="Distribution", figure=fig)
        plt.close(fig)
    else:
        plt.show()


def pca(input: torch.Tensor, k: int = 2) -> torch.Tensor:
    input = input.flatten(1)
    *_, V = torch.pca_lowrank(input, q=k)
    return input @ V[:, :k]


@torch.no_grad()
def get_transport_plot_pca(
    source_samples: torch.Tensor,
    target_samples: torch.Tensor,
    moved_samples: torch.Tensor,
    *,
    colors=None,
    experiment: Experiment | None = None,
    **figure_kwargs,
):
    if source_samples.size(1) != 2:
        source_samples = pca(source_samples, 2)

    if moved_samples.size(1) != 2:
        moved_samples, target_samples = pca(
            torch.cat([moved_samples, target_samples]), 2
        ).chunk(2)

    figure = plt.figure(**figure_kwargs)

    if colors is None:
        colors = source_samples[:, 1].cpu()

    source_axis = figure.add_subplot(1, 2, 1)
    source_axis.scatter(
        *source_samples.cpu().T, c=colors, label="Source samples", alpha=0.5
    )
    source_axis.set_title("Source space")

    target_axis = figure.add_subplot(1, 2, 2)
    target_axis.scatter(
        *target_samples.cpu().T, c="black", label="Target samples", alpha=0.5
    )
    target_axis.scatter(
        *moved_samples.cpu().T, c=colors, label="Moved samples", alpha=0.5
    )
    target_axis.set_title("Target space")
    target_axis.legend()

    if experiment is not None:
        experiment.log_figure(figure_name="Mapping", figure=figure)
        plt.close(figure)
    else:
        plt.show()
        return

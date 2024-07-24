import matplotlib.cm as cm
import numpy as np
import torch
from matplotlib import pyplot as plt

import wandb
from src.models.light_gcot import LightGCOT
from src.samplers.primary import GridGaussiansSampler, Sampler


def plot_A_parameters(model: LightGCOT, log: bool = False) -> dict[str, wandb.Image] | None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=200)
    color = cm.rainbow(np.linspace(0.1, 0.9, 1))

    log_w_n = model.compute_log_w_n()
    a_n = model.compute_a_n()
    A_n = model.compute_A_n()
    axes[0].scatter(np.arange(model.n_potentials), log_w_n.cpu().detach().numpy(), color=color)
    axes[0].set_xlabel("N")
    axes[0].set_ylabel("value")
    axes[0].set_title(r"$\log{w_n}$")
    axes[0].grid(zorder=-20)

    axes[1].scatter(a_n[:, 0].cpu().detach().numpy(), a_n[:, 1].cpu().detach().numpy(), color=color)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_title(r"$a_n$")
    axes[1].grid(zorder=-20)

    axes[2].scatter(
        A_n[:, 0].cpu().detach().numpy(),
        A_n[:, 1].cpu().detach().numpy(),
        color=color,
    )
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    axes[2].set_title(r"$A_n$")
    axes[2].grid(zorder=-20)

    if log:
        A_dict = {"A parameters": wandb.Image(fig)}
        plt.close(fig)
        return A_dict
    else:
        plt.show()


def plot_B_parameters(
    model: LightGCOT, starting_points: torch.Tensor, log: bool = False
) -> dict[str, wandb.Image] | None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=200)

    colors = cm.rainbow(np.linspace(0.1, 0.9, len(starting_points)))
    log_v_m = model.compute_log_v_m(starting_points)
    b_m = model.compute_b_m(starting_points)
    for i, (color, point) in enumerate(zip(colors, starting_points)):
        label = f"{point.cpu().numpy()}"

        axes[0].scatter(np.arange(model.m_potentials), log_v_m[i].cpu().detach().numpy(), label=label, color=color)
        axes[0].set_xlabel("M")
        axes[0].set_ylabel("value")
        axes[0].set_title(r"$\log{v_m}$")
        axes[0].grid(zorder=-20)

        axes[1].scatter(
            b_m[i, :, 0].cpu().detach().numpy(),
            b_m[i, :, 1].cpu().detach().numpy(),
            label=label,
            color=color,
        )
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        axes[1].set_title(r"$b_m$")
        axes[1].grid(zorder=-20)

    for _, ax in enumerate(axes):
        ax.legend(loc="lower right")

    fig.tight_layout(pad=0.1)
    if log:
        B_dict = {"B parameters": wandb.Image(fig)}
        plt.close(fig)
        return B_dict
    else:
        plt.show()


def plot_gaussians(
    model: LightGCOT,
    X_sampler: GridGaussiansSampler,
    Y_sampler: GridGaussiansSampler,
    X_paired: torch.Tensor,
    Y_paired: torch.Tensor,
    num_samples: int = 256,
    log: bool = False,
) -> dict[str, wandb.Image] | None:
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

    if log:
        distr_dict = {"Distribution": wandb.Image(fig)}
        plt.close(fig)
        return distr_dict
    else:
        plt.show()


def plot_distributions(
    model: LightGCOT,
    X_sampler: Sampler,
    Y_sampler: Sampler,
    X_paired: torch.Tensor,
    Y_paired: torch.Tensor,
    starting_points: torch.Tensor,
    num_ending_points: int = 256,
    num_samples: int = 1024,
    log: bool = False,
) -> dict[str, wandb.Image] | None:
    colors = cm.rainbow(np.linspace(0.1, 0.9, len(starting_points)))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=200)

    for ax in axes:
        ax.grid(zorder=-20)

    x_samples = X_sampler.sample(num_samples)
    y_samples = Y_sampler.sample(num_samples)

    # First plot
    axes[0].scatter(
        x_samples[:, 0].cpu().numpy(),
        x_samples[:, 1].cpu().numpy(),
        alpha=0.3,
        c="g",
        s=32,
        edgecolors="black",
        label=r"Input distirubtion $p_0$",
    )
    axes[0].scatter(
        y_samples[:, 0].cpu().numpy(),
        y_samples[:, 1].cpu().numpy(),
        c="orange",
        s=32,
        edgecolors="black",
        label=r"Target distribution $p_1$",
    )

    # Second plot
    y_pred = model(x_samples).cpu().numpy()
    axes[1].scatter(
        y_pred[:, 0], y_pred[:, 1], c="yellow", s=32, edgecolors="black", label="Fitted distribution", zorder=1
    )

    for color, point in zip(colors, starting_points):
        label = f"{point.cpu().numpy()}"
        repeated_starting_points = point[None, :].repeat(num_ending_points, 1)
        point_pred = model(repeated_starting_points).cpu().numpy()
        axes[1].scatter(
            point[0].item(),
            point[1].item(),
            color=color,
            label=label,
            s=48,
            zorder=3,
            edgecolors="black",
            marker="s",
        )
        axes[1].scatter(
            point_pred[:, 0],
            point_pred[:, 1],
            color=color,
            s=32,
            zorder=3,
            edgecolors="black",
        )

    pair_colors = cm.rainbow(np.linspace(0, 1, len(X_paired)))
    for color, x, y in zip(pair_colors, X_paired.cpu().numpy(), Y_paired.cpu().numpy()):
        axes[2].scatter(x[0], x[1], color=color, s=32, edgecolors="black")
        axes[2].scatter(y[0], y[1], color=color, s=32, edgecolors="black")
        axes[2].arrow(x[0], x[1], y[0] - x[0], y[1] - x[1], color=color)

    for _, ax in enumerate(axes):
        ax.set_xlim([-3.5, 3.5])
        ax.set_ylim([-3.5, 3.5])
        ax.legend(loc="lower right")

    fig.tight_layout(pad=0.1)

    if log:
        distr_dict = {"Distribution": wandb.Image(fig)}
        plt.close(fig)
        return distr_dict
    else:
        plt.show()

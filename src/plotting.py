import matplotlib.cm as cm
import numpy as np
import torch
from matplotlib import pyplot as plt

import wandb
from src.distributions import Sampler
from src.light_gcot import LightGCOT


def plot_A_parameters(
    model: LightGCOT, starting_points: torch.Tensor, log: bool = False
) -> dict[str, wandb.Image] | None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=200)
    color = cm.rainbow(np.linspace(0.1, 0.9, 1))
    colors_chosen = cm.rainbow(np.linspace(0.1, 0.9, len(starting_points)))

    axes[0].scatter(np.arange(model.n_potentials), model.log_w_n.cpu().detach().numpy(), color=color)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title(r"$\log{w_n}$")
    axes[0].grid(zorder=-20)

    axes[1].scatter(model.a_n[:, 0].cpu().detach().numpy(), model.a_n[:, 1].cpu().detach().numpy(), color=color)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_title(r"$a_n$")
    axes[1].grid(zorder=-20)

    B_m = model.compute_B_m(starting_points)
    A_n = model.compute_A_n(B_m)
    for i, (color, point) in enumerate(zip(colors_chosen, starting_points)):
        label = f"{point.cpu().numpy()}"
        axes[2].scatter(
            A_n[i, :, 0].cpu().detach().numpy(),
            A_n[i, :, 1].cpu().detach().numpy(),
            label=label,
            color=color,
        )
        axes[2].set_xlabel("x")
        axes[2].set_ylabel("y")
        axes[2].set_title(r"$A_n$")
        axes[2].grid(zorder=-20)
    axes[2].legend(loc="lower right")

    if log:
        A_dict = {"A parameters": wandb.Image(fig)}
        plt.close(fig)
        return A_dict
    else:
        plt.show()


def plot_B_parameters(
    model: LightGCOT, starting_points: torch.Tensor, log: bool = False
) -> dict[str, wandb.Image] | None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=200)

    colors_chosen = cm.rainbow(np.linspace(0.1, 0.9, len(starting_points)))
    log_v_m = model.compute_log_v_m(starting_points)
    b_m = model.compute_b_m(starting_points)
    B_m = model.compute_B_m(starting_points)
    for i, (color, point) in enumerate(zip(colors_chosen, starting_points)):
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

        axes[2].scatter(
            B_m[i, :, 0].cpu().detach().numpy(),
            B_m[i, :, 1].cpu().detach().numpy(),
            label=label,
            color=color,
        )
        axes[2].set_xlabel("x (log. scale)")
        axes[2].set_ylabel("y (log. scale)")
        axes[2].set_xscale("log")
        axes[2].set_yscale("log")
        axes[2].set_title(r"$B_m$")
        axes[2].grid(zorder=-20)

    for _, ax in enumerate(axes):
        ax.legend(loc="lower right")

    fig.tight_layout(pad=0.1)
    if log:
        B_dict = {"B parameters": wandb.Image(fig)}
        plt.close(fig)
        return B_dict
    else:
        plt.show()


def plot_G_parameters(
    model: LightGCOT, starting_points: torch.Tensor, log: bool = False
) -> dict[str, wandb.Image] | None:
    fig, axes = plt.subplots(1, 5, figsize=(25, 5), dpi=200)
    colors_chosen = cm.rainbow(np.linspace(0.1, 0.9, len(starting_points)))

    log_v_m = model.compute_log_v_m(starting_points)
    b_m = model.compute_b_m(starting_points)
    B_m = model.compute_B_m(starting_points)

    A_n = model.compute_A_n(B_m)

    c_nm = model.compute_c_nm(b_m, B_m, A_n)
    b_nm = model.compute_b_nm(b_m, B_m, A_n)

    G_nm, G_inv_nm = model.compute_G_nm(B_m, A_n)
    g_nm = model.compute_g_nm(b_nm, G_inv_nm)

    log_alpha_nm = model.compute_log_alpha_nm(log_v_m, B_m, G_nm, c_nm)
    log_Z_nm = model.compute_log_Z_nm(log_alpha_nm, G_inv_nm, b_nm)
    scale = torch.sqrt(model.epsilon * G_inv_nm).view(
        len(starting_points), model.n_potentials * model.m_potentials, model.y_dim
    )
    for i, (color, point) in enumerate(zip(colors_chosen, starting_points)):
        label = f"{point.cpu().numpy()}"

        axes[0].scatter(
            b_nm[i].view(model.m_potentials * model.n_potentials, model.y_dim)[:, 0].cpu().detach().numpy(),
            b_nm[i].view(model.m_potentials * model.n_potentials, model.y_dim)[:, 1].cpu().detach().numpy(),
            label=label,
            color=color,
        )
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        axes[0].set_title(r"$b_{nm}$")
        axes[0].grid(zorder=-20)

        axes[1].scatter(
            np.arange(model.m_potentials * model.n_potentials),
            c_nm[i].view(model.m_potentials * model.n_potentials).cpu().detach().numpy(),
            label=label,
            color=color,
        )
        axes[1].set_xlabel("N * M")
        axes[1].set_ylabel("value")
        axes[1].set_title(r"$c_{nm}$")
        axes[1].grid(zorder=-20)

        axes[2].scatter(
            np.arange(model.m_potentials * model.n_potentials),
            log_Z_nm[i].view(model.m_potentials * model.n_potentials).cpu().detach().numpy(),
            label=label,
            color=color,
        )
        axes[2].set_xlabel("N * M")
        axes[2].set_ylabel("value")
        axes[2].set_title(r"$\log{Z_{nm}}$")
        axes[2].grid(zorder=-20)

        axes[3].scatter(
            g_nm[i].view(model.m_potentials * model.n_potentials, model.y_dim)[:, 0].cpu().detach().numpy(),
            g_nm[i].view(model.m_potentials * model.n_potentials, model.y_dim)[:, 1].cpu().detach().numpy(),
            label=label,
            color=color,
        )
        axes[3].set_xlabel("x")
        axes[3].set_ylabel("y")
        axes[3].set_title(r"$g_{nm}$")
        axes[3].grid(zorder=-20)

        axes[4].scatter(
            scale[i, :, 0].cpu().detach().numpy(),
            scale[i, :, 1].cpu().detach().numpy(),
            label=label,
            color=color,
        )
        axes[4].set_xlabel("x (log. scale)")
        axes[4].set_ylabel("y (log. scale)")
        axes[4].set_xscale("log")
        axes[4].set_yscale("log")
        axes[4].set_title(r"$\varepsilon\sqrt{G_{nm}^{-1}}$")
        axes[4].grid(zorder=-20)

    for _, ax in enumerate(axes):
        ax.legend(loc="lower right")

    fig.tight_layout(pad=0.1)
    if log:
        G_dict = {"G parameters": wandb.Image(fig)}
        plt.close(fig)
        return G_dict
    else:
        plt.show()


def plot_distributions(
    model: LightGCOT,
    X_sampler: Sampler,
    Y_sampler: Sampler,
    starting_points: torch.Tensor,
    num_ending_points: int = 256,
    num_samples: int = 1024,
    log: bool = False,
) -> dict[str, wandb.Image] | None:
    colors_chosen = cm.rainbow(np.linspace(0.1, 0.9, len(starting_points)))
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

    for color, point in zip(colors_chosen, starting_points):
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
        )
        axes[1].scatter(
            point_pred[:, 0],
            point_pred[:, 1],
            color=color,
            s=32,
            zorder=3,
            edgecolors="black",
        )

        # Third plot
        log_v_m = model.compute_log_v_m(repeated_starting_points)
        b_m = model.compute_b_m(repeated_starting_points)
        B_m = model.compute_B_m(repeated_starting_points)
        y_cost_sample = model.compute_cost(b_m, B_m, log_v_m, sample=True).cpu().numpy()
        axes[2].scatter(
            y_cost_sample[:, 0],
            y_cost_sample[:, 1],
            color=color,
            s=32,
            edgecolors="black",
            label=label,
        )

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
        plt.show()

import matplotlib.cm as cm
import numpy as np
import torch
from matplotlib import pyplot as plt

import wandb
from src.models.light_gcot import LightGCOT


def plot_A_parameters(model: LightGCOT, log: bool = False) -> dict[str, wandb.Image] | None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=200)
    color = cm.rainbow(np.linspace(0.1, 0.9, 1))

    log_w_n = model.compute_log_w_n().cpu().detach().numpy()
    a_n = model.compute_a_n().cpu().detach().numpy()
    A_n = model.compute_A_n().cpu().detach().numpy()

    coeffs = np.exp(log_w_n)
    alphas = coeffs / np.sum(coeffs)

    axes[0].scatter(np.arange(model.n_potentials), log_w_n, alpha=alphas, color=color)
    axes[0].set_xlabel("N")
    axes[0].set_ylabel("value")
    axes[0].set_title(r"$\log{w_n}$")
    axes[0].grid(zorder=-20)

    axes[1].scatter(a_n[:, 0], a_n[:, 1], alpha=alphas, color=color)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_title(r"$a_n$")
    axes[1].grid(zorder=-20)

    axes[2].scatter(A_n[:, 0], A_n[:, 1], alpha=alphas, color=color)
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
    num_subplots = 2
    fig, axes = plt.subplots(1, num_subplots, figsize=(5 * num_subplots, 5), dpi=200)

    num_starting_points = len(starting_points)
    colors = cm.rainbow(np.linspace(0.1, 0.9, num_starting_points))
    log_v_m = model.compute_log_v_m(starting_points).cpu().detach().numpy()
    b_m = model.compute_b_m(starting_points).cpu().detach().numpy()

    coeffs = np.exp(log_v_m)
    alphas = coeffs / np.sum(coeffs)

    for i, (color, point) in enumerate(zip(colors, starting_points)):
        label = f"{point.cpu().numpy()}"

        axes[0].scatter(np.arange(model.m_potentials), log_v_m[i], alpha=alphas, label=label, color=color)
        axes[0].set_xlabel("M")
        axes[0].set_ylabel("value")
        axes[0].set_title(r"$\log{v_m(x)}$")
        axes[0].grid(zorder=-20)

        axes[1].scatter(b_m[i, :, 0], b_m[i, :, 1], alpha=alphas, label=label, color=color)
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        axes[1].set_title(r"$b_m(x)$")
        axes[1].grid(zorder=-20)

    for i, ax in enumerate(axes[:2]):
        ax.legend(loc="lower right")

    fig.tight_layout(pad=0.1)
    if log:
        B_dict = {"B parameters": wandb.Image(fig)}
        plt.close(fig)
        return B_dict
    else:
        plt.show()


def plot_Z_parameters(
    model: LightGCOT,
    starting_points: torch.Tensor,
    X_paired: torch.Tensor | None = None,
    Y_paired: torch.Tensor | None = None,
    log: bool = False,
) -> dict[str, wandb.Image] | None:
    if X_paired is not None and Y_paired is not None:
        num_subplots = 5
    elif X_paired is None and Y_paired is None:
        num_subplots = 2
    else:
        raise ValueError("X_paired and Y_paired must be None or not None simultaneously!")
    fig, axes = plt.subplots(1, num_subplots, figsize=(5 * num_subplots, 5), dpi=200)

    num_starting_points = len(starting_points)  # nsp
    colors = cm.rainbow(np.linspace(0.1, 0.9, num_starting_points))

    log_v_m = model.compute_log_v_m(starting_points)
    b_m = model.compute_b_m(starting_points)

    log_w_n = model.compute_log_w_n()
    a_n = model.compute_a_n()
    A_n = model.compute_A_n()

    r_nm = (
        (a_n[None, :, None, :] + A_n[None, :, None, :] * b_m[:, None, :, :])
        .reshape(num_starting_points, model.n_potentials * model.m_potentials, model.y_dim)
        .cpu()
        .detach()
        .numpy()
    )  # view([1 x N x 1 x y_dim] + [1 x N x 1 x y_dim] * [nsp x 1 x M x y_dim] = [nsp x N x M x y_dim]) = [nsp x N * M]
    log_Z_nm = (
        model.compute_log_Z_nm(log_w_n, a_n, A_n, log_v_m, b_m)
        .reshape(num_starting_points, model.n_potentials * model.m_potentials)
        .cpu()
        .detach()
        .numpy()
    )  # [nsp x N * M]

    for i, (color, point) in enumerate(zip(colors, starting_points)):
        label = f"{point.cpu().numpy()}"

        coeffs = np.exp(log_Z_nm[i])
        alphas = coeffs / np.sum(coeffs)
        axes[0].scatter(
            r_nm[i, :, 0],
            r_nm[i, :, 1],
            alpha=alphas,
            label=label,
            color=color,
        )
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        axes[0].set_title(r"$r_{nm}(x)$")
        axes[0].grid(zorder=-20)

        axes[1].scatter(
            np.arange(model.n_potentials * model.m_potentials),
            log_Z_nm[i],
            alpha=alphas,
            label=label,
            color=color,
        )
        axes[1].set_xlabel("N * M")
        axes[1].set_ylabel("value")
        axes[1].set_title(r"$\log{Z_{nm}(x)}$")
        axes[1].grid(zorder=-20)

    if X_paired is not None and Y_paired is not None:
        num_starting_paired_points = len(X_paired)
        colors_paired = cm.rainbow(np.linspace(0.1, 0.9, num_starting_paired_points))

        log_v_m_cost = model.compute_log_v_m(X_paired).cpu().detach().numpy()
        b_m_cost = model.compute_b_m(X_paired)
        scalar_product_m = (
            torch.sum(b_m_cost * Y_paired[:, None, :], dim=2).cpu().detach().numpy()
        )  # sum([bs x M x y_dim] * [bs x 1 x y_dim], dim=(1, 2)) = [bs x M]

        for i, (color, point) in enumerate(zip(colors_paired, X_paired.cpu().numpy())):
            label = f"[{point[0]:.2f}, {point[1]:.2f}]"

            coeffs = np.exp(log_v_m_cost[i])
            alphas = coeffs / np.sum(coeffs)
            axes[2].scatter(
                np.arange(model.m_potentials),
                log_v_m_cost[i],
                alpha=alphas,
                label=label,
                color=color,
            )
            axes[2].set_xlabel("M")
            axes[2].set_ylabel("value")
            axes[2].set_title(r"$\log{v_m}(x_{paired})$")
            axes[2].grid(zorder=-20)

            axes[3].scatter(
                np.arange(model.m_potentials),
                scalar_product_m[i],
                alpha=alphas,
                label=label,
                color=color,
            )
            axes[3].set_xlabel("M")
            axes[3].set_ylabel("value")
            axes[3].set_title(r"$\langle b_m(x_{paired}), y_{paired} \rangle$")
            axes[3].grid(zorder=-20)

            axes[4].scatter(
                np.arange(model.m_potentials),
                -scalar_product_m[i] / model.epsilon.cpu().detach().numpy() - log_v_m_cost[i],
                alpha=alphas,
                label=label,
                color=color,
            )
            axes[4].set_xlabel("M")
            axes[4].set_ylabel("value")
            axes[4].set_title(
                r"$-\log{v_m}(x_{paired})-\dfrac{\langle b_m(x_{paired}), y_{paired} \rangle}{\varepsilon}$"
            )
            axes[4].grid(zorder=-20)

    for _, ax in enumerate(axes):
        ax.legend(loc="lower right")

    fig.tight_layout(pad=0.1)
    if log:
        Z_dict = {"Z parameters": wandb.Image(fig)}
        plt.close(fig)
        return Z_dict
    else:
        plt.show()

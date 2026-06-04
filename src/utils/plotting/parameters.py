import matplotlib.cm as cm
import numpy as np
import torch
from comet_ml import Experiment
from matplotlib import pyplot as plt

from src.ebieot.costs.lse import BaseLSECost
from src.ebieot.ebieot_gmm import EbieotGmm
from src.utils.plotting.distributions import pca


def plot_A_parameters(
    model: EbieotGmm, experiment: Experiment | None = None
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=200)
    color = cm.rainbow(np.linspace(0.1, 0.9, 1))

    log_w_n = model.log_w_n()
    a_n = model.a_n()
    A_n = model.A_n()

    if A_n.size(1) != 2:
        A_n = pca(A_n, 2)
    if a_n.size(1) != 2:
        a_n = pca(a_n, 2)

    a_n = a_n.cpu().detach().numpy()
    A_n = A_n.cpu().detach().numpy()

    log_coeffs = torch.logsumexp(log_w_n, dim=0)
    alphas = 0.1 + torch.exp(log_w_n - log_coeffs).cpu().detach().numpy() * 0.9

    axes[0].scatter(
        np.arange(model.n_potentials),
        log_w_n.cpu().detach().numpy(),
        alpha=alphas,
        color=color,
    )
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

    fig.tight_layout(pad=0.1)

    if experiment is not None:
        experiment.log_figure(figure_name="A parameters", figure=fig)
        plt.close(fig)
    else:
        plt.show()


def plot_B_parameters(
    cost: BaseLSECost, starting_points: torch.Tensor, experiment: Experiment | None
) -> None:
    num_subplots = 2
    fig, axes = plt.subplots(1, num_subplots, figsize=(5 * num_subplots, 5), dpi=200)

    num_starting_points = len(starting_points)
    colors = cm.rainbow(np.linspace(0.1, 0.9, num_starting_points))
    log_v_m = cost.log_v_m(starting_points)
    b_m = cost.b_m(starting_points)  # [nsp x M x y_dim]

    log_coeffs = torch.logsumexp(log_v_m, dim=0)
    alphas = 0.1 + torch.exp(log_v_m - log_coeffs).cpu().detach().numpy() * 0.9
    log_v_m = log_v_m.cpu().detach().numpy()

    for i, (color, point) in enumerate(zip(colors, starting_points)):
        label = f"{point.cpu().numpy()[:2]}"

        axes[0].scatter(
            np.arange(cost.m_potentials),
            log_v_m[i],
            alpha=alphas,
            label=label,
            color=color,
        )
        axes[0].set_xlabel("M")
        axes[0].set_ylabel("value")
        axes[0].set_title(r"$\log{v_m(x)}$")
        axes[0].grid(zorder=-20)

        if cost.y_dim != 2:
            b_m_i = pca(b_m[i], 2).cpu().detach().numpy()
        else:
            b_m_i = b_m[i].cpu().detach().numpy()

        axes[1].scatter(
            b_m_i[:, 0], b_m_i[:, 1], alpha=alphas, label=label, color=color
        )
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        axes[1].set_title(r"$b_m(x)$")
        axes[1].grid(zorder=-20)

    for i, ax in enumerate(axes[:2]):
        ax.legend(loc="lower right")

    fig.tight_layout(pad=0.1)

    if experiment is not None:
        experiment.log_figure(figure_name="B parameters", figure=fig)
        plt.close(fig)
    else:
        plt.show()


def plot_Z_parameters(
    model: EbieotGmm,
    starting_points: torch.Tensor,
    X_paired: torch.Tensor | None = None,
    Y_paired: torch.Tensor | None = None,
    experiment: Experiment | None = None,
) -> None:
    if X_paired is not None and Y_paired is not None:
        fig, axes = plt.subplots(2, 4, figsize=(20, 10), dpi=200)
        axes = axes.reshape(-1)
    elif X_paired is None and Y_paired is None:
        # num_subplots = 5
        fig, axes = plt.subplots(1, 5, figsize=(25, 5), dpi=200, squeeze=False)
    else:
        raise ValueError(
            "X_paired and Y_paired must be None or not None simultaneously!"
        )

    num_starting_points = len(starting_points)  # nsp
    colors = cm.rainbow(np.linspace(0.1, 0.9, num_starting_points))

    log_v_m = model.cost.log_v_m(starting_points)
    b_m = model.cost.b_m(starting_points)

    log_w_n = model.log_w_n()
    a_n = model.a_n()
    A_n = model.A_n()

    r_nm = (a_n[None, :, None, :] + A_n[None, :, None, :] * b_m[:, None, :, :]).reshape(
        num_starting_points, model.n_potentials * model.cost.m_potentials, model.y_dim
    )  # [nsp x N * M x y_dim]
    log_Z_nm = model.log_Z_nm(log_w_n, a_n, A_n, log_v_m, b_m).reshape(
        num_starting_points, model.n_potentials * model.cost.m_potentials
    )  # [nsp x N * M]

    bT_A = b_m[:, None, :, :] * A_n[None, :, None, :]
    # [bs x 1 x M x y_dim] * [1 x N x 1 x y_dim] = [bs x N x M x y_dim]
    bT_A_b = torch.sum(bT_A * b_m[:, None, :, :], dim=3).cpu().detach().numpy()
    aT_b = (
        torch.sum(2 * a_n[None, :, None, :] * b_m[:, None, :, :], dim=3)
        .cpu()
        .detach()
        .numpy()
    )
    correction = 0.5 * (bT_A_b + aT_b) / model.epsilon.cpu().detach().numpy()

    log_coeffs = torch.logsumexp(log_Z_nm, dim=1)
    alphas = (
        0.1 + torch.exp(log_Z_nm - log_coeffs[:, None]).cpu().detach().numpy() * 0.9
    )
    log_Z_nm = log_Z_nm.cpu().detach().numpy()

    for i, (color, point) in enumerate(zip(colors, starting_points)):
        label = f"{point.cpu().numpy()[:2]}"

        alpha = alphas[i]
        if model.y_dim != 2:
            r_nm_i = pca(r_nm[i]).cpu().detach().numpy()
        else:
            r_nm_i = r_nm[i].cpu().detach().numpy()
        axes[0].scatter(
            r_nm_i[:, 0],
            r_nm_i[:, 1],
            alpha=alpha,
            label=label,
            color=color,
        )
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        axes[0].set_title(r"$r_{nm}(x)$")
        axes[0].grid(zorder=-20)

        axes[1].scatter(
            np.arange(model.n_potentials * model.cost.m_potentials),
            log_Z_nm[i],
            alpha=alpha,
            label=label,
            color=color,
        )
        axes[1].set_xlabel("N * M")
        axes[1].set_ylabel("value")
        axes[1].set_title(r"$\log{Z_{nm}(x)}$")
        axes[1].grid(zorder=-20)

        axes[2].scatter(
            np.arange(model.n_potentials * model.cost.m_potentials),
            bT_A_b[i],
            alpha=alpha,
            label=label,
            color=color,
        )
        axes[2].set_xlabel("N * M")
        axes[2].set_ylabel("value")
        axes[2].set_title(r"$b_m(x)^\top A_n b_m(x)$")
        axes[2].grid(zorder=-20)

        axes[3].scatter(
            np.arange(model.n_potentials * model.cost.m_potentials),
            aT_b[i],
            alpha=alpha,
            label=label,
            color=color,
        )
        axes[3].set_xlabel("N * M")
        axes[3].set_ylabel("value")
        axes[3].set_title(r"$2 \cdot b_m(x)^\top a_n$")
        axes[3].grid(zorder=-20)

        axes[4].scatter(
            np.arange(model.n_potentials * model.cost.m_potentials),
            correction[i],
            alpha=alpha,
            label=label,
            color=color,
        )
        axes[4].set_xlabel("N * M")
        axes[4].set_ylabel("value")
        axes[4].set_title(
            r"$\dfrac{b_m(x)^\top A_n b_m(x) + 2 \cdot b_m(x)^\top a_n}{2 \varepsilon}$"
        )
        axes[4].grid(zorder=-20)

    if X_paired is not None and Y_paired is not None:
        num_starting_paired_points = len(X_paired)
        colors_paired = cm.rainbow(np.linspace(0.1, 0.9, num_starting_paired_points))

        log_v_m_cost = model.cost.log_v_m(X_paired)
        b_m_cost = model.cost.b_m(X_paired)
        scalar_product_m = (
            torch.sum(b_m_cost * Y_paired[:, None, :], dim=2).cpu().detach().numpy()
        )  # sum([bs x M x y_dim] * [bs x 1 x y_dim], dim=(1, 2)) = [bs x M]

        log_coeffs = torch.logsumexp(log_v_m_cost, dim=1)
        alphas = (
            0.1
            + torch.exp(log_v_m_cost - log_coeffs[:, None]).cpu().detach().numpy() * 0.9
        )
        log_v_m_cost = log_v_m_cost.cpu().detach().numpy()

        for i, (color, point) in enumerate(zip(colors_paired, X_paired.cpu().numpy())):
            label = f"[{point[0]:.2f}, {point[1]:.2f}]"

            alpha = alphas[i]
            axes[5].scatter(
                np.arange(model.cost.m_potentials),
                log_v_m_cost[i],
                alpha=alpha,
                label=label,
                color=color,
            )
            axes[5].set_xlabel("M")
            axes[5].set_ylabel("value")
            axes[5].set_title(r"$\log{v_m}(x_{paired})$")
            axes[5].grid(zorder=-20)

            axes[6].scatter(
                np.arange(model.cost.m_potentials),
                scalar_product_m[i],
                alpha=alpha,
                label=label,
                color=color,
            )
            axes[6].set_xlabel("M")
            axes[6].set_ylabel("value")
            axes[6].set_title(r"$\langle b_m(x_{paired}), y_{paired} \rangle$")
            axes[6].grid(zorder=-20)

            axes[7].scatter(
                np.arange(model.cost.m_potentials),
                -scalar_product_m[i] / model.epsilon.cpu().detach().numpy()
                - log_v_m_cost[i],
                alpha=alpha,
                label=label,
                color=color,
            )
            axes[7].set_xlabel("M")
            axes[7].set_ylabel("value")
            axes[7].set_title(
                r"$-\log{v_m}(x_{paired})-\dfrac{\langle b_m(x_{paired}), y_{paired} \rangle}{\varepsilon}$"
            )
            axes[7].grid(zorder=-20)

    for _, ax in enumerate(axes):
        ax.legend(loc="lower right")

    fig.tight_layout(pad=0.1)

    if experiment is not None:
        experiment.log_figure(figure_name="Z parameters", figure=fig)
        plt.close(fig)
    else:
        plt.show()

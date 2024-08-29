import matplotlib.cm as cm
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

import wandb
from src.models.light_gcot import LightGCOT
from src.samplers.primary import GridGaussiansSampler, Sampler
from src.utils.discrete_ot import OTPlanSampler


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
    starting_points: list[torch.Tensor],
    gt_Y_points: list[np.ndarray],
    num_ending_points: int = 256,
    num_samples: int = 1024,
    log: bool = False,
) -> dict[str, wandb.Image] | None:
    colors = cm.rainbow(np.linspace(0.1, 0.9, len(starting_points)))
    num_subplots = 4
    fig, axes = plt.subplots(1, num_subplots, figsize=(5 * num_subplots, 5), dpi=200)

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
    axes[1].scatter(
        X_paired[:, 0].cpu().numpy(),
        X_paired[:, 1].cpu().numpy(),
        alpha=0.3,
        c="g",
        s=32,
        edgecolors="black",
        label=r"Input paired samples from distribution $p_0$",
    )
    axes[1].scatter(
        Y_paired[:, 0].cpu().numpy(),
        Y_paired[:, 1].cpu().numpy(),
        c="orange",
        s=32,
        edgecolors="black",
        label=r"Target paired samples from distribution $p_1$",
    )

    for x, y in zip(X_paired.cpu().numpy(), Y_paired.cpu().numpy()):
        axes[1].arrow(x[0], x[1], y[0] - x[0], y[1] - x[1], color="black")

    # Third plot
    axes[2].set_title(f"Ground truth mapping")
    axes[2].scatter(
        y_samples[:, 0].cpu().numpy(),
        y_samples[:, 1].cpu().numpy(),
        c="orange",
        s=32,
        edgecolors="black",
        label=r"Target distribution $p_1$",
    )
    for color, point, gt_point in zip(colors, starting_points, gt_Y_points):
        label = f"{point.cpu().numpy()}"
        axes[2].scatter(
            point[0].item(),
            point[1].item(),
            color=color,
            label=label,
            s=48,
            zorder=3,
            edgecolors="black",
            marker="s",
        )
        axes[2].scatter(
            gt_point[:, 0],
            gt_point[:, 1],
            color=color,
            s=32,
            zorder=3,
            edgecolors="black",
            marker="d",
        )

    # Fourth plot
    y_pred = model(x_samples).cpu().numpy()
    axes[3].scatter(
        y_pred[:, 0], y_pred[:, 1], c="yellow", s=32, edgecolors="black", label="Fitted distribution", zorder=1
    )

    for color, point in zip(colors, starting_points):
        label = f"{point.cpu().numpy()}"
        repeated_starting_points = point[None, :].repeat(num_ending_points, 1)
        point_pred = model(repeated_starting_points).cpu().numpy()
        axes[3].scatter(
            point[0].item(),
            point[1].item(),
            color=color,
            label=label,
            s=48,
            zorder=3,
            edgecolors="black",
            marker="s",
        )
        axes[3].scatter(
            point_pred[:, 0],
            point_pred[:, 1],
            color=color,
            s=32,
            zorder=3,
            edgecolors="black",
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


def plot_PCA(
    model: LightGCOT,
    source_data: torch.Tensor,
    target_data: torch.Tensor,
    paired_source_data: torch.Tensor,
    paired_target_data: torch.Tensor,
    lims: tuple[tuple] = ((-25, 50), (-25, 30)),
    log: bool = False,
) -> dict[str, wandb.Image] | None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), squeeze=True, sharex=True, sharey=True)
    pca = PCA(n_components=2).fit(target_data.cpu().numpy())

    source_data_pca = pca.transform(source_data.cpu().numpy())
    target_data_pca = pca.transform(target_data.cpu().numpy())

    # First plot
    axes[0].scatter(
        source_data_pca[:, 0], source_data_pca[:, 1], c="g", edgecolor="black", label=r"$x\sim P_0(x)$", s=30
    )
    # Second plot
    axes[1].scatter(
        target_data_pca[:, 0], target_data_pca[:, 1], c="orange", edgecolor="black", label=r"$x\sim P_1(x)$", s=30
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
        pred_data_pca[:, 0], pred_data_pca[:, 1], c="yellow", edgecolor="black", label=r"$x\sim T(x)$", s=30
    )
    for source_point, target_point, pred_point in zip(paired_source_data_pca, paired_target_data_pca, pred_data_pca):
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

    if log:
        distr_dict = {"PCA samples": wandb.Image(fig)}
        plt.close(fig)
        return distr_dict
    else:
        plt.show()

# download Oxford Flowers 102, plotting functions, and toy dataset

import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision as tv
from torchvision import transforms

sys.path.append("../..")

import wandb


##################
# ## PLOTTING ## #
##################
def normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor - tensor.min()) / (tensor.max() - tensor.min())


def tensor2image(tensor: torch.Tensor, normalize: bool = True, invert: bool = False) -> torch.Tensor:
    tensor = normalize_tensor(tensor) if normalize else tensor
    assert not invert or normalize
    tensor = 1 - tensor if invert else tensor

    return (tensor * 255).byte()


# def plot_images(
#     image_name: str,
#     source_tensor: torch.Tensor,
#     step: int,
#     target_tensor: torch.Tensor | None = None,
#     nrow: int | None = None,
#     invert: bool = False,
#     clamp: bool = True,
#     normalize: bool = True,
#     use_wandb: bool = False,
#     save_dir: Path | None = None,
# ):
#     if target_tensor is not None:
#         assert source_tensor.shape == target_tensor.shape

#     if nrow is None:
#         nrow = int(source_tensor.shape[0] ** 0.5)
#     im_shape = tuple(source_tensor.shape[1:])  # source_tensor: [B, C, H, W]

#     normalize_transorm = transforms.Normalize(mean=[0.5], std=[0.5])
#     source_tensor = normalize_transorm(source_tensor) if normalize else source_tensor
#     source_tensor = source_tensor.clamp(-1.0, 1.0) if clamp else source_tensor
#     source_images = tensor2image(source_tensor, invert=invert)
#     if target_tensor is not None:
#         target_tensor = normalize_transorm(target_tensor) if normalize else target_tensor
#         target_tensor = target_tensor.clamp(-1.0, 1.0) if clamp else target_tensor
#         target_images = tensor2image(target_tensor, invert=invert)
#         output_images = torch.stack([source_images, target_images], dim=1).view(-1, *im_shape)
#     else:
#         output_images = source_images

#     # pad_value = 1.0 if invert else 0.0
#     pad_value = 0
#     grid = tv.utils.make_grid(output_images, nrow=nrow, pad_value=pad_value)

#     fig = plt.figure()
#     plt.imshow(grid.permute(1, 2, 0).detach().cpu().numpy())
#     plt.axis("off")

#     # Add column subtitles
#     columns = ["Column 1", "Column 2", "Column 3", "Column 4"]
#     for i, col in enumerate(columns):
#         plt.text(
#             i * grid.shape[2] // 4 + grid.shape[2] // 8,
#             -15,
#             col,
#             color="black",
#             ha="center",
#             va="center",
#             fontsize=12,
#             weight="bold",
#         )

#     if save_dir is not None:
#         os.makedirs(save_dir, exist_ok=True)

#         fig.savefig(save_dir / f"{image_name}_{step:>06d}.png")
#         print(f"Saved {image_name} into {save_dir}")

#     if use_wandb:
#         distr_dict = {image_name: wandb.Image(fig)}
#         plt.close(fig)
#         return distr_dict
#     else:
#         plt.show()


def plot_images(
    image_name: str,
    source_tensor: torch.Tensor,
    step: int,
    target_tensor: torch.Tensor | None = None,
    nrow: int | None = None,
    invert: bool = False,
    clamp: bool = True,
    normalize: bool = True,
    experiment=None,  # <-- Comet experiment object
    save_dir: Path | None = None,
):
    if target_tensor is not None:
        assert source_tensor.shape == target_tensor.shape

    if nrow is None:
        nrow = int(source_tensor.shape[0] ** 0.5)
    im_shape = tuple(source_tensor.shape[1:])  # source_tensor: [B, C, H, W]

    normalize_transform = transforms.Normalize(mean=[0.5], std=[0.5])
    source_tensor = normalize_transform(source_tensor) if normalize else source_tensor
    source_tensor = source_tensor.clamp(-1.0, 1.0) if clamp else source_tensor
    source_images = tensor2image(source_tensor, invert=invert)

    if target_tensor is not None:
        target_tensor = normalize_transform(target_tensor) if normalize else target_tensor
        target_tensor = target_tensor.clamp(-1.0, 1.0) if clamp else target_tensor
        target_images = tensor2image(target_tensor, invert=invert)
        output_images = torch.stack([source_images, target_images], dim=1).view(-1, *im_shape)
    else:
        output_images = source_images

    pad_value = 0
    grid = tv.utils.make_grid(output_images, nrow=nrow, pad_value=pad_value)

    fig = plt.figure()
    plt.imshow(grid.permute(1, 2, 0).detach().cpu().numpy())
    plt.axis("off")

    # Optional: Add column subtitles
    # columns = ["Column 1", "Column 2", "Column 3", "Column 4"]
    # for i, col in enumerate(columns):
    #     plt.text(
    #         i * grid.shape[2] // 4 + grid.shape[2] // 8,
    #         -15,
    #         col,
    #         color="black",
    #         ha="center",
    #         va="center",
    #         fontsize=12,
    #         weight="bold",
    #     )

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fig_path = save_dir / f"{image_name}_{step:>06d}.png"
        fig.savefig(fig_path)
        print(f"Saved {image_name} into {save_dir}")

        # Log image to Comet if experiment is provided
        if experiment is not None:
            experiment.log_image(str(fig_path), name=image_name, step=step)

    elif experiment is not None:
        # Log directly from memory (no file saved)
        experiment.log_figure(figure_name=image_name, figure=fig, step=step)

    plt.close(fig)


# plot diagnostics for learning
def plot_diagnostics(batch, en_diffs, grad_mags, exp_dir, fontsize=10):
    # axis tick size
    matplotlib.rc("xtick", labelsize=6)
    matplotlib.rc("ytick", labelsize=6)
    fig = plt.figure()

    def plot_en_diff_and_grad_mag():
        # energy difference
        ax = fig.add_subplot(221)
        ax.plot(en_diffs[0 : (batch + 1)].data.cpu().numpy())
        ax.axhline(y=0, ls="--", c="k")
        ax.set_title("Energy Difference", fontsize=fontsize)
        ax.set_xlabel("batch", fontsize=fontsize)
        ax.set_ylabel("$d_{s_t}$", fontsize=fontsize)
        # mean langevin gradient
        ax = fig.add_subplot(222)
        ax.plot(grad_mags[0 : (batch + 1)].data.cpu().numpy())
        ax.set_title("Average Langevin Gradient Magnitude", fontsize=fontsize)
        ax.set_xlabel("batch", fontsize=fontsize)
        ax.set_ylabel("$r_{s_t}$", fontsize=fontsize)

    def plot_crosscorr_and_autocorr(t_gap_max=2000, max_lag=15, b_w=0.35):
        t_init = max(0, batch + 1 - t_gap_max)
        t_end = batch + 1
        t_gap = t_end - t_init
        max_lag = min(max_lag, t_gap - 1)
        # rescale energy diffs to unit mean square but leave uncentered
        en_rescale = en_diffs[t_init:t_end] / torch.sqrt(
            torch.sum(en_diffs[t_init:t_end] * en_diffs[t_init:t_end]) / (t_gap - 1)
        )
        # normalize gradient magnitudes
        grad_rescale = (grad_mags[t_init:t_end] - torch.mean(grad_mags[t_init:t_end])) / torch.std(
            grad_mags[t_init:t_end]
        )
        # cross-correlation and auto-correlations
        cross_corr = np.correlate(en_rescale.cpu().numpy(), grad_rescale.cpu().numpy(), "full") / (t_gap - 1)
        en_acorr = np.correlate(en_rescale.cpu().numpy(), en_rescale.cpu().numpy(), "full") / (t_gap - 1)
        grad_acorr = np.correlate(grad_rescale.cpu().numpy(), grad_rescale.cpu().numpy(), "full") / (t_gap - 1)
        # x values and indices for plotting
        x_corr = np.linspace(-max_lag, max_lag, 2 * max_lag + 1)
        x_acorr = np.linspace(0, max_lag, max_lag + 1)
        t_0_corr = int((len(cross_corr) - 1) / 2 - max_lag)
        t_0_acorr = int((len(cross_corr) - 1) / 2)

        # plot cross-correlation
        ax = fig.add_subplot(223)
        ax.bar(x_corr, cross_corr[t_0_corr : (t_0_corr + 2 * max_lag + 1)])
        ax.axhline(y=0, ls="--", c="k")
        ax.set_title("Cross Correlation of Energy Difference\nand Gradient Magnitude", fontsize=fontsize)
        ax.set_xlabel("lag", fontsize=fontsize)
        ax.set_ylabel("correlation", fontsize=fontsize)
        # plot auto-correlation
        ax = fig.add_subplot(224)
        ax.bar(x_acorr - b_w / 2, en_acorr[t_0_acorr : (t_0_acorr + max_lag + 1)], b_w, label="en. diff. $d_{s_t}$")
        ax.bar(
            x_acorr + b_w / 2, grad_acorr[t_0_acorr : (t_0_acorr + max_lag + 1)], b_w, label="grad. mag. $r_{s_t}}$"
        )
        ax.axhline(y=0, ls="--", c="k")
        ax.set_title("Auto-Correlation of Energy Difference\nand Gradient Magnitude", fontsize=fontsize)
        ax.set_xlabel("lag", fontsize=fontsize)
        ax.set_ylabel("correlation", fontsize=fontsize)
        ax.legend(loc="upper right", fontsize=fontsize - 4)

    # make diagnostic plots
    plot_en_diff_and_grad_mag()
    plot_crosscorr_and_autocorr()
    # save figure
    plt.subplots_adjust(hspace=0.6, wspace=0.6)
    # plt.savefig(os.path.join(exp_dir, "diagnosis_plot.pdf"), format="pdf")
    plt.close()


def steps_counter(s0, s1, res0=False, res1=True):
    assert res0 != res1
    curr_step = 0
    steps_passed = 0
    res_mapping = [res0, res1]
    while True:
        steps_passed += 1
        if curr_step == 0:
            if steps_passed > s0:
                curr_step = 1
                steps_passed = 1
        elif curr_step == 1:
            if steps_passed > s1:
                curr_step = 0
                steps_passed = 1
        yield res_mapping[curr_step]

# download Oxford Flowers 102, plotting functions, and toy dataset

import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision as tv
import torchvision.transforms as transforms

sys.path.append("../..")

import wandb

##################
# ## PLOTTING ## #
##################


# visualize negative samples synthesized from energy
def plot_ims(p, x, n_step=None, im_name="dummy name", use_wandb=False, nrow=None, invert=False):
    x = torch.clamp(x, -1.0, 1.0)
    if invert:
        x = 1.0 - x
    if nrow is None:
        nrow = int(x.shape[0] ** 0.5)
    pad_value = 1.0 if invert else 0.0
    if not use_wandb:
        tv.utils.save_image(x, p, normalize=True, nrow=nrow, pad_value=pad_value)
    else:
        SB_torch_grid = tv.utils.make_grid(x, nrow=nrow, pad_value=pad_value, normalize=True)
        SB_images = wandb.Image(SB_torch_grid, caption="Xs")
        wandb.log(
            {
                im_name: [
                    SB_images,
                ]
            },
            step=n_step,
        )


def plot_im_pairs(p, x, y, n_step=None, im_name="dummy name", use_wandb=False, nrow=None, invert=False):
    if nrow is None:
        nrow = int(x.shape[0] ** 0.5)
    assert x.shape == y.shape
    im_shape = tuple(x.shape[1:])
    # if invert:
    #     to_draw = torch.clamp(torch.cat([x.unsqueeze(1), y.unsqueeze(1)], 1).view(-1, *im_shape), -1.0, 1.0)
    #     to_draw = 1.0 - to_draw
    # else:
    #     to_draw = torch.cat([x.unsqueeze(1), y.unsqueeze(1)], 1).view(-1, *im_shape)

    to_draw = torch.clamp(torch.cat([x.unsqueeze(1), y.unsqueeze(1)], 1).view(-1, *im_shape), -1.0, 1.0)

    # min_val = y.min()
    # max_val = y.max()
    # y = (y - min_val) / (max_val - min_val)
    # # Rescale to the range [-1, 1]
    # y = y * 2 - 1
    # to_draw = torch.clamp(torch.cat([x.unsqueeze(1), y.unsqueeze(1)], 1).view(-1, *im_shape), -1.0, 1.0)
    if invert:
        to_draw = 1.0 - to_draw
    pad_value = 1.0 if invert else 0.0
    if not use_wandb:
        tv.utils.save_image(to_draw, p, normalize=True, nrow=nrow, pad_value=pad_value)
    else:
        SB_torch_grid = tv.utils.make_grid(to_draw, nrow=nrow, pad_value=pad_value, normalize=False)
        SB_images = wandb.Image(SB_torch_grid, caption="first: X, second: Y")
        wandb.log(
            {im_name: [SB_images]},
            step=n_step,
        )


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

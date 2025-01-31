import torch
from matplotlib import pyplot as plt
from torchvision.transforms.functional import to_pil_image

import wandb


def plot_images(X: torch.Tensor, Y_pred: torch.Tensor, Y: torch.Tensor, log: bool = False):
    imgs = torch.cat([X, Y_pred, Y]).to("cpu").permute(0, 2, 3, 1).mul(0.5).add(0.5).numpy().clip(0, 1)
    fig, axes = plt.subplots(3, 10, figsize=(15, 4.5), dpi=150)
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(imgs[i], cmap="Greys")
        ax.get_xaxis().set_visible(False)
        ax.set_yticks([])

    axes[0, 0].set_ylabel("X", fontsize=24)
    axes[1, 0].set_ylabel("T(X)", fontsize=24)
    axes[2, 0].set_ylabel("Y", fontsize=24)
    fig.tight_layout(pad=0.001)

    if log:
        distr_dict = {"Distribution": wandb.Image(fig)}
        plt.close(fig)
        return distr_dict
    else:
        plt.show()


def plot_samples(
    model: torch.nn.Module,
    source_samples: torch.Tensor,
    num_samples: int = 5,
    title: str = "Source and Target Samples",
    log: bool = False,
):
    """
    Plots `num_samples` source and target samples, one under another.

    Args:
        source_samples (list of Tensors): List of source images (e.g., digit 2).
        target_samples (list of Tensors): List of target images (e.g., digit 3).
        num_samples (int): Number of samples to plot from each category.
        title (str): Title for the plot.
    """
    # Ensure we have enough samples
    num_samples = min(num_samples, len(source_samples))
    target_samples = model(source_samples[:num_samples])

    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 3, 6), dpi=150)

    for i in range(num_samples):
        # Plot source sample
        source_img = to_pil_image((source_samples[i] * 0.5 + 0.5))  # De-normalize to [0, 1]
        axes[0, i].imshow(source_img)

        # Plot target sample
        target_img = to_pil_image((target_samples[i] * 0.5 + 0.5))  # De-normalize to [0, 1]
        axes[1, i].imshow(target_img)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    if log:
        distr_dict = {"Distribution": wandb.Image(fig)}
        plt.close(fig)
        return distr_dict
    else:
        plt.show()

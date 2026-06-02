import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from src.plotting.comet_logging import log_matplotlib_figure


def plot_PCA(
    model: nn.Module,
    source_data: torch.Tensor,
    target_data: torch.Tensor,
    paired_source_data: torch.Tensor,
    paired_target_data: torch.Tensor,
    lims: tuple[tuple] = ((-25, 50), (-25, 30)),
    experiment=None,
    step: int | None = None,
) -> None:
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

    if experiment is not None:
        log_matplotlib_figure(experiment, "PCA samples", fig, step=step)
    else:
        plt.show()

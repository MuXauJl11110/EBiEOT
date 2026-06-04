"""Samplers for colored MNIST EBiEOT notebooks."""

import torch
from omegaconf import DictConfig

from src.utils.datasets.colored_mnist import (
    build_unpaired_colored,
    download_digit_images,
    flatten_images,
    get_paired_digits,
)
from src.utils.samplers.data import DatasetSampler, PairedTensorBatchSampler


def build_colored_mnist_samplers(
    cfg: DictConfig,
    device: torch.device,
) -> tuple[
    DatasetSampler,
    DatasetSampler,
    PairedTensorBatchSampler,
    torch.Tensor,
    torch.Tensor,
]:
    """Build unpaired / paired samplers and image tensors for plotting.

    Returns:
        unpaired source, unpaired target, paired batch sampler,
        paired images ``(N, 3, H, W)`` for source and target.
    """
    ds = cfg.dataset
    img_size = int(ds.img_size)
    channels = int(ds.channels)
    flat_dim = channels * img_size * img_size
    data_root = ds.get("data_root", "./data")

    source_images = download_digit_images(
        str(ds.get("source_dataset", "MNIST")),
        int(ds.source_digit),
        int(ds.max_source_samples) if ds.get("max_source_samples") else None,
        img_size=img_size,
        root=data_root,
    )
    target_images = download_digit_images(
        str(ds.get("target_dataset", "MNIST")),
        int(ds.target_digit),
        int(ds.max_target_samples) if ds.get("max_target_samples") else None,
        img_size=img_size,
        root=data_root,
    )

    mb = ds.get("minibatch", {})
    x_paired_img, y_paired_img = get_paired_digits(
        source_images,
        target_images,
        int(ds.P_XY_paired),
        hue_offset=float(ds.get("hue_offset", 120.0)),
        device=device,
        ot_method=str(mb.get("method", "sinkhorn")),
        ot_reg=float(mb.get("reg", 0.05)),
    )
    q_x_img = build_unpaired_colored(source_images, device=device)
    q_y_img = build_unpaired_colored(target_images, device=device)

    x_paired = flatten_images(x_paired_img)
    y_paired = flatten_images(y_paired_img)
    paired_sampler = PairedTensorBatchSampler(x_paired, y_paired)

    num_unpaired_x = int(ds.Q_X_unpaired)
    num_unpaired_y = int(ds.R_Y_unpaired)
    q_x_flat = flatten_images(q_x_img)
    q_y_flat = flatten_images(q_y_img)

    if num_unpaired_x > 0:
        n_x = min(num_unpaired_x, q_x_flat.shape[0])
        unpaired_x = DatasetSampler(q_x_flat[:n_x], device=str(device))
    else:
        unpaired_x = DatasetSampler(x_paired, device=str(device))

    if num_unpaired_y > 0:
        n_y = min(num_unpaired_y, q_y_flat.shape[0])
        unpaired_y = DatasetSampler(q_y_flat[:n_y], device=str(device))
    else:
        unpaired_y = DatasetSampler(y_paired, device=str(device))

    assert flat_dim == int(ds.x_dim) == int(ds.y_dim), (
        f"dataset x_dim/y_dim ({ds.x_dim}) must equal "
        f"channels*img_size**2 ({flat_dim})"
    )

    return unpaired_x, unpaired_y, paired_sampler, x_paired_img, y_paired_img


def build_colored_mnist_image_samplers(
    cfg: DictConfig,
    device: torch.device,
) -> tuple[
    DatasetSampler,
    DatasetSampler,
    PairedTensorBatchSampler,
    torch.Tensor,
    torch.Tensor,
]:
    """Like ``build_colored_mnist_samplers`` but keeps ``(N, C, H, W)`` tensors."""
    ds = cfg.dataset
    img_size = int(ds.img_size)
    channels = int(ds.channels)
    flat_dim = channels * img_size * img_size
    data_root = ds.get("data_root", "./data")

    source_images = download_digit_images(
        str(ds.get("source_dataset", "MNIST")),
        int(ds.source_digit),
        int(ds.max_source_samples) if ds.get("max_source_samples") else None,
        img_size=img_size,
        root=data_root,
    )
    target_images = download_digit_images(
        str(ds.get("target_dataset", "MNIST")),
        int(ds.target_digit),
        int(ds.max_target_samples) if ds.get("max_target_samples") else None,
        img_size=img_size,
        root=data_root,
    )

    mb = ds.get("minibatch", {})
    x_paired_img, y_paired_img = get_paired_digits(
        source_images,
        target_images,
        int(ds.P_XY_paired),
        hue_offset=float(ds.get("hue_offset", 120.0)),
        device=device,
        ot_method=str(mb.get("method", "sinkhorn")),
        ot_reg=float(mb.get("reg", 0.05)),
    )
    q_x_img = build_unpaired_colored(source_images, device=device)
    q_y_img = build_unpaired_colored(target_images, device=device)

    paired_sampler = PairedTensorBatchSampler(x_paired_img, y_paired_img)

    num_unpaired_x = int(ds.Q_X_unpaired)
    num_unpaired_y = int(ds.R_Y_unpaired)
    if num_unpaired_x > 0:
        n_x = min(num_unpaired_x, q_x_img.shape[0])
        unpaired_x = DatasetSampler(q_x_img[:n_x], device=str(device))
    else:
        unpaired_x = DatasetSampler(x_paired_img, device=str(device))

    if num_unpaired_y > 0:
        n_y = min(num_unpaired_y, q_y_img.shape[0])
        unpaired_y = DatasetSampler(q_y_img[:n_y], device=str(device))
    else:
        unpaired_y = DatasetSampler(y_paired_img, device=str(device))

    assert flat_dim == int(ds.x_dim) == int(ds.y_dim), (
        f"dataset x_dim/y_dim ({ds.x_dim}) must equal "
        f"channels*img_size**2 ({flat_dim})"
    )

    return unpaired_x, unpaired_y, paired_sampler, x_paired_img, y_paired_img

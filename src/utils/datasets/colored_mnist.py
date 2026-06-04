"""Colored MNIST digit transport (e.g. 2 → 3) helpers."""

import random
from pathlib import Path

import torch
import torchvision.transforms as T
from torchvision import datasets

from src.utils.samplers.discrete_ot import OTPlanSampler


def flatten_images(images: torch.Tensor) -> torch.Tensor:
    """(N, C, H, W) → (N, C·H·W)."""
    return images.flatten(1)


def unflatten_images(flat: torch.Tensor, *, channels: int = 3, img_size: int = 32) -> torch.Tensor:
    """(N, C·H·W) → (N, C, H, W)."""
    return flat.view(flat.shape[0], channels, img_size, img_size)


def download_digit_images(
    dataset_name: str, digit: int, max_samples: int | None = None, *, img_size: int = 32, root: str | Path = "./data"
) -> list[torch.Tensor]:
    """Load grayscale MNIST digits as ``(1, H, W)`` tensors in ``[-1, 1]``."""
    transform = T.Compose(
        [
            T.Resize(img_size),
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,)),
        ]
    )
    name = dataset_name.upper()
    if name != "MNIST":
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    data = datasets.MNIST(root=str(root), train=True, download=True, transform=transform)
    indices = [i for i, label in enumerate(data.targets) if int(label) == digit]
    if max_samples is not None:
        indices = indices[:max_samples]
    return [data[i][0] for i in indices]


def apply_random_color(image: torch.Tensor, hue: torch.Tensor | float) -> torch.Tensor:
    if image.dim() == 3 and image.shape[0] == 1:
        image = image.squeeze(0)

    device = image.device
    dtype = image.dtype
    if isinstance(hue, torch.Tensor):
        hue = hue.to(device=device, dtype=dtype)
    else:
        hue = torch.tensor(hue, device=device, dtype=dtype)

    image_min = torch.zeros((), device=device, dtype=dtype)
    frac = (hue % 60) / 60
    image_diff = (image - image_min) * frac
    image_inc = image_diff
    image_dec = image - image_diff
    colored_image = torch.zeros((3, *image.shape), device=device, dtype=dtype)
    H_i = int(torch.round(hue / 60) % 6)

    if H_i == 0:
        colored_image[0] = image
        colored_image[1] = image_inc
        colored_image[2] = image_min
    elif H_i == 1:
        colored_image[0] = image_dec
        colored_image[1] = image
        colored_image[2] = image_min
    elif H_i == 2:
        colored_image[0] = image_min
        colored_image[1] = image
        colored_image[2] = image_inc
    elif H_i == 3:
        colored_image[0] = image_min
        colored_image[1] = image_dec
        colored_image[2] = image
    elif H_i == 4:
        colored_image[0] = image_inc
        colored_image[1] = image_min
        colored_image[2] = image
    elif H_i == 5:
        colored_image[0] = image
        colored_image[1] = image_min
        colored_image[2] = image_dec

    return colored_image


def get_paired_digits(
    source_data: list[torch.Tensor],
    target_data: list[torch.Tensor],
    num_pairs: int,
    hue_offset: int = 120,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates paired digit samples with random color transformations,
    where half are hue-shifted by +offset and half by -offset (modulo 360)."""
    num_pairs = min(num_pairs, len(source_data), len(target_data))

    paired_source_samples = []
    paired_target_samples = []

    # Define signs: half +1, half -1
    signs = torch.tensor([1] * (num_pairs // 2) + [-1] * (num_pairs - num_pairs // 2))
    signs = signs[torch.randperm(num_pairs)]  # Shuffle to randomize order

    for i, (src_data, tgt_data) in enumerate(zip(source_data[:num_pairs], target_data[:num_pairs])):
        src_hue = 360 * torch.rand(1, device=src_data.device)
        sign = signs[i]
        tgt_hue = (src_hue + sign * hue_offset) % 360

        paired_source_samples.append(apply_random_color(src_data, src_hue))
        paired_target_samples.append(apply_random_color(tgt_data, tgt_hue))

    q_x_paired = torch.stack(paired_source_samples).to(device)
    q_y_paired = torch.stack(paired_target_samples).to(device)

    return q_x_paired, q_y_paired


def build_unpaired_colored(images: list[torch.Tensor], *, device: torch.device | str = "cpu") -> torch.Tensor:
    """Randomly color each grayscale digit; returns ``(N, 3, H, W)``."""
    device = torch.device(device)
    colored = [
        apply_random_color(img.to(device), 360.0 * torch.rand((), device=device)) for img in images
    ]
    return torch.stack(colored)

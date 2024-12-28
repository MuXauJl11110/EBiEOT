import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import TensorDataset
from torchvision import datasets

from src.samplers.from_dataset import DatasetSampler


def apply_random_color(image: torch.Tensor, hue: torch.Tensor) -> torch.Tensor:
    image_min = 0
    image_diff = (image - image_min) * (hue % 60) / 60
    image_inc = image_diff
    image_dec = image - image_diff
    colored_image = torch.zeros((3, image.shape[1], image.shape[2]))
    H_i = torch.round(hue / 60) % 6  # type: ignore

    transform_norm = transforms.Normalize([0.5], [0.5])

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

    return transform_norm(colored_image)


def download_digit_images(
    dataset: str,
    digit: int,
    num_digits: int,
    image_size: int = 32,
    root: str = "./data",
    train: bool = True,
) -> list[torch.Tensor]:
    """Downloads and filters digit images from a dataset."""
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    if dataset == "MNIST":
        data = datasets.MNIST(root=root, train=train, transform=transform, download=True)
    elif dataset == "USPS":
        data = datasets.USPS(root=root, train=train, transform=transform, download=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset}!")

    num_digits = min(num_digits, len(data.targets))
    indices = [i for i, label in enumerate(data.targets) if label == digit]

    return [data[i][0] for i in indices][:num_digits]


def get_paired_digits(
    source_data: list[torch.Tensor],
    target_data: list[torch.Tensor],
    num_pairs: int,
    hue_offset: int = 120,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates paired digit samples with random color transformations."""
    num_pairs = min(num_pairs, len(source_data), len(target_data))

    paired_source_samples = []
    paired_target_samples = []

    for src_data, tgt_data in zip(source_data[:num_pairs], target_data[:num_pairs]):
        src_hue = 360 * torch.rand(1)
        tgt_hue = (src_hue + hue_offset) % 360
        paired_source_samples.append(apply_random_color(src_data, src_hue.to(src_data.device)))
        paired_target_samples.append(apply_random_color(tgt_data, tgt_hue.to(tgt_data.device)))

    q_x_paired = torch.stack(paired_source_samples).to(device)
    q_y_paired = torch.stack(paired_target_samples).to(device)

    return q_x_paired, q_y_paired

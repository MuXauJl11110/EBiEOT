import os
import shutil

import numpy as np
import torch as t
from torch.utils.data import TensorDataset
from torchvision import datasets
from torchvision import transforms as tv
from torchvision.utils import save_image


def get_random_colored_images(images, pix_val_range=(-1.0, 1.0), seed=0x000000):
    """
    Apply random coloring to grayscale images based on random hues.

    Args:
        images (Tensor): Grayscale images of shape (N, 1, H, W).
        pix_val_range (tuple): Range for pixel values after transformation.
        seed (int): Seed for random hue generation.

    Returns:
        Tensor: Colored images of shape (N, 3, H, W).
    """
    np.random.seed(seed)

    # Normalize and denormalize pixel values
    normalize = lambda x: (x - pix_val_range[0]) / (pix_val_range[1] - pix_val_range[0])
    denormalize = lambda x: pix_val_range[0] + x * (pix_val_range[1] - pix_val_range[0])

    images = normalize(images)
    num_images = images.shape[0]
    hues = 360 * np.random.rand(num_images)

    colored_images = []

    for V, H in zip(images, hues):
        V_min = 0
        a = (V - V_min) * (H % 60) / 60
        V_inc = a
        V_dec = V - a

        colored_image = t.zeros((3, V.shape[1], V.shape[2]))
        H_i = int(H // 60) % 6

        # Assign RGB channels based on hue segment
        if H_i == 0:
            colored_image[0], colored_image[1], colored_image[2] = V, V_inc, V_min
        elif H_i == 1:
            colored_image[0], colored_image[1], colored_image[2] = V_dec, V, V_min
        elif H_i == 2:
            colored_image[0], colored_image[1], colored_image[2] = V_min, V, V_inc
        elif H_i == 3:
            colored_image[0], colored_image[1], colored_image[2] = V_min, V_dec, V
        elif H_i == 4:
            colored_image[0], colored_image[1], colored_image[2] = V_inc, V_min, V
        elif H_i == 5:
            colored_image[0], colored_image[1], colored_image[2] = V, V_min, V_dec

        colored_images.append(colored_image)

    colored_images = t.stack(colored_images, dim=0)
    colored_images = denormalize(colored_images)

    return colored_images


def load_cmnist_dataset(
    name, path, batch_size=64, shuffle=True, device="cuda", pix_val_range=(-1.0, 1.0), seed=0x000000
):
    """
    Load and preprocess the (colored) MNIST dataset.

    Args:
        name (str): Dataset name, e.g., 'MNIST_colored_0_1_2'.
        path (str): Path to download the dataset.
        batch_size (int): Batch size.
        shuffle (bool): Shuffle data during loading.
        device (str): Device for data processing ('cpu' or 'cuda').
        pix_val_range (tuple): Range for pixel values.
        seed (int): Seed for color generation.

    Returns:
        Tuple[TensorDataset, TensorDataset]: Processed train and test sets.
    """
    assert name.startswith("MNIST"), "Dataset name must start with 'MNIST'."

    # Normalize pixel values
    normalize = lambda x: pix_val_range[0] + x * (pix_val_range[1] - pix_val_range[0])

    transform = tv.Compose([tv.Resize((32, 32)), tv.ToTensor(), tv.Lambda(normalize)])

    dataset_type = name.split("_")[0]
    is_colored = dataset_type.endswith("colored")
    classes = [int(digit) for digit in name.split("_")[1:]] or list(range(10))

    # Load MNIST datasets
    train_set = datasets.MNIST(path, train=True, transform=transform, download=True)
    test_set = datasets.MNIST(path, train=False, transform=transform, download=True)

    def filter_and_stack(dataset):
        data, labels = [], []
        for class_idx, class_label in enumerate(classes):
            class_data = t.stack(
                [dataset[i][0] for i in range(len(dataset)) if dataset.targets[i] == class_label], dim=0
            )
            data.append(class_data)
            labels.extend([class_idx] * len(class_data))

        data = t.cat(data, dim=0).reshape(-1, 1, 32, 32)
        labels = t.tensor(labels)

        if is_colored:
            data = get_random_colored_images(data, pix_val_range, seed=seed)

        return TensorDataset(data, labels)

    return filter_and_stack(train_set), filter_and_stack(test_set)


def download_colored_mnist_data(name):
    """
    Download and save the Colored MNIST dataset as images.

    Args:
        name (str): Name of the dataset.
    """
    dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"dataset/{name}/")
    if os.path.exists(dataset_dir):
        return  # Dataset already exists

    os.makedirs(dataset_dir, exist_ok=True)
    raw_dir = os.path.join(dataset_dir, "__raw")
    ims_dir = os.path.join(dataset_dir, "ims")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(ims_dir, exist_ok=True)

    train_dataset, test_dataset = load_cmnist_dataset(
        name, raw_dir, batch_size=64, shuffle=False, device="cpu", pix_val_range=(0.0, 1.0)
    )

    def save_dataset_images(dataset, start_idx=0):
        for idx, (image, _) in enumerate(dataset):
            save_image(image, os.path.join(ims_dir, f"im_{start_idx + idx:06d}.png"))
        return start_idx + len(dataset)

    save_dataset_images(train_dataset)
    save_dataset_images(test_dataset, start_idx=len(train_dataset))

    shutil.rmtree(raw_dir)

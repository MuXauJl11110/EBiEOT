import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.samplers.base import Sampler
from src.samplers.from_loader import PairedLoaderSampler, PairedWithLabelsLoaderSampler
from src.utils.discrete_ot import OTPlanSampler


def generate_paired_data(
    X_sampler: Sampler,
    Y_sampler: Sampler,
    mini_batch_sampler: OTPlanSampler,
    num_samples: int,
    save_dir: str,
    file_postfix: str,
    mini_batch_size: int = 64,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not os.path.exists(os.path.join(save_dir, f"X_paired_train_{file_postfix}.pt")):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        X_paired_list, Y_paired_list = [], []

        for _ in tqdm(range(2 * num_samples)):  # the first part for train, another for test
            _X_paired, _Y_paired = X_sampler.sample(mini_batch_size), Y_sampler.sample(mini_batch_size)
            _X_paired, _Y_paired = mini_batch_sampler.sample_plan(_X_paired, _Y_paired)
            X_paired_list.append(_X_paired[0])
            Y_paired_list.append(_Y_paired[0])

        X_paired, Y_paired = torch.stack(X_paired_list), torch.stack(Y_paired_list)

        torch.save(X_paired[:num_samples], os.path.join(save_dir, f"X_paired_train_{file_postfix}.pt"))
        torch.save(Y_paired[:num_samples], os.path.join(save_dir, f"Y_paired_train_{file_postfix}.pt"))
        torch.save(X_paired[num_samples:], os.path.join(save_dir, f"X_paired_test_{file_postfix}.pt"))
        torch.save(Y_paired[num_samples:], os.path.join(save_dir, f"Y_paired_test_{file_postfix}.pt"))

        X_paired_train = X_paired[:num_samples]
        Y_paired_train = Y_paired[:num_samples]
        X_paired_test = X_paired[num_samples:]
        Y_paired_test = Y_paired[num_samples:]
    else:
        X_paired_train = torch.load(
            os.path.join(save_dir, f"X_paired_train_{file_postfix}.pt"), map_location=device, weights_only=True
        )
        Y_paired_train = torch.load(
            os.path.join(save_dir, f"Y_paired_train_{file_postfix}.pt"), map_location=device, weights_only=True
        )
        X_paired_test = torch.load(
            os.path.join(save_dir, f"X_paired_test_{file_postfix}.pt"), map_location=device, weights_only=True
        )
        Y_paired_test = torch.load(
            os.path.join(save_dir, f"Y_paired_test_{file_postfix}.pt"), map_location=device, weights_only=True
        )

    return X_paired_train, Y_paired_train, X_paired_test, Y_paired_test


def get_paired_sampler(
    X_paired: torch.Tensor, Y_paired: torch.Tensor, batch_size: int, num_samples: int, device: str = "cuda"
) -> PairedLoaderSampler:
    assert len(X_paired) == len(Y_paired)
    loader_kwargs = {"num_workers": 0, "generator": torch.Generator(device=X_paired.device)}
    ind = random.choices(range(len(X_paired)), k=min(num_samples, len(X_paired)))
    paired_loader = DataLoader(
        TensorDataset(X_paired[ind], Y_paired[ind]),
        batch_size=min(batch_size, num_samples),
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    return PairedLoaderSampler(paired_loader, device=device)


def get_GT_points(
    X_sampler: Sampler,
    Y_sampler: Sampler,
    mini_batch_sampler: OTPlanSampler,
    starting_points: list[torch.Tensor],
    num_ending_points: int = 64,
) -> list[np.ndarray]:
    gt_Y_points = []
    for point in starting_points:
        _gt_points = []
        for _ in tqdm(range(num_ending_points)):
            x_start = torch.cat((point[None, :], X_sampler.sample(num_ending_points - 1)))
            y_end = Y_sampler.sample(num_ending_points)
            p = mini_batch_sampler.get_map(x_start, y_end)
            point_true = y_end[np.argmax(p[0])].cpu().numpy()
            _gt_points.append(point_true)
        gt_Y_points.append(np.array(_gt_points))

    return gt_Y_points


def get_paired_with_labels_sampler(
    X_paired: torch.Tensor,
    X_labels_paired: torch.Tensor,
    Y_paired: torch.Tensor,
    Y_labels_paired: torch.Tensor,
    batch_size: int,
    num_samples: int,
    device: str = "cuda",
) -> PairedLoaderSampler:
    assert len(X_paired) == len(Y_paired)
    loader_kwargs = {
        "num_workers": 0,
        "generator": torch.Generator(device=X_paired.device),
    }
    ind = random.choices(range(len(X_paired)), k=min(num_samples, len(X_paired)))
    paired_loader = DataLoader(
        TensorDataset(X_paired[ind], X_labels_paired[ind], Y_paired[ind], Y_labels_paired[ind]),
        batch_size=min(batch_size, num_samples),
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    return PairedWithLabelsLoaderSampler(paired_loader, device=device)

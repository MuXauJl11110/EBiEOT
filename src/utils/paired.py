import os
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from src.samplers.base import Sampler
from src.samplers.from_loader import PairedLoaderSampler, PairedWithLabelsLoaderSampler
from src.samplers.primary import SwissRollSampler, swiss_roll_transform
from src.utils.discrete_ot import OTPlanSampler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


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
    ind = torch.randperm(len(X_paired), device=X_paired.device)[:num_samples]
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


def match_gaussian_and_swiss_roll(
    Y_sampler: SwissRollSampler,
    starting_points: list[torch.Tensor],
    num_ending_points: int = 64,
    g_func: Callable[[torch.Tensor], torch.Tensor] | None = None,
    noise_std: float = 0.1,
) -> torch.Tensor:
    assert g_func is not None

    gt_Y_points = []

    generator = Y_sampler.generator
    t_min = Y_sampler.t_min
    t_max = Y_sampler.t_max
    t_mid = 0.5 * (t_min + t_max)
    scale = Y_sampler.scale

    for point in tqdm(starting_points):
        n = num_ending_points

        x_batch = point.unsqueeze(0).repeat(n, 1)
        gx = g_func(x_batch)

        x_norm = torch.norm(gx, dim=-1)
        base = torch.tanh(x_norm)
        is_upper = torch.rand(n, device=point.device) > 0.5

        t_lower = t_min + (t_mid - t_min) * base
        t_upper = t_mid + (t_max - t_mid) * base

        t = torch.where(is_upper, t_upper, t_lower)
        t = t + noise_std * torch.randn_like(t)

        y_spiral = swiss_roll_transform(t=t, generator=generator, noise=Y_sampler.noise) / scale

        gt_Y_points.append(y_spiral.cpu().numpy())

    return torch.from_numpy(np.stack(gt_Y_points)).to(Y_sampler.device)


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


def get_point_filename(point: np.ndarray) -> str:
    return f"point_{point[0]:.4f}_{point[1]:.4f}.npz"


def save_gt_points(
    starting_points: torch.Tensor, gt_Y_points: list[np.ndarray], target_dir: str = "./data/sinkhorn_points"
):
    """Saves each ground truth array to a separate file named by its starting point."""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    for point_tensor, gt_Y in zip(starting_points, gt_Y_points):
        point_np = point_tensor.cpu().numpy()
        filename = get_point_filename(point_np)
        target_path = target_dir / filename

        # Save with a fixed key 'data' for easy retrieval
        np.savez_compressed(target_path, data=gt_Y)
        print(f"Successfully saved: {target_path.name}")


def load_or_compute_gt_points(
    starting_points: torch.Tensor,
    X_sampler: Any,
    Y_sampler: Any,
    otp_sampler: Any,
    compute_func: Callable,
    target_dir: str = "./data/sinkhorn_points",
    num_ending_points: int = 1024,
) -> list[np.ndarray]:
    """
    Loads points from disk. If any points are missing, computes them using
    the provided samplers and saves them before returning the full list.
    """
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    results = [None] * len(starting_points)
    indices_to_compute = []
    points_to_compute = []

    # 1. Check disk for existing points
    for i, point_tensor in enumerate(starting_points):
        point_np = point_tensor.cpu().numpy()
        file_path = target_path / get_point_filename(point_np)

        if file_path.exists():
            with np.load(file_path) as loader:
                results[i] = loader["data"]
        else:
            indices_to_compute.append(i)
            points_to_compute.append(point_tensor)

    # 2. Compute missing points if necessary
    if points_to_compute:
        print(f"Missing {len(points_to_compute)} points. Computing...")
        # Convert list of tensors back to a single batch tensor
        batch_to_compute = torch.stack(points_to_compute)

        new_gt_points = compute_func(
            X_sampler, Y_sampler, otp_sampler, batch_to_compute, num_ending_points=num_ending_points
        )

        # 3. Save newly computed points
        save_gt_points(batch_to_compute, new_gt_points, target_path)

        # 4. Fill results list
        for idx, gt_data in zip(indices_to_compute, new_gt_points):
            results[idx] = gt_data

    return results

import typing as tp
from pathlib import Path

import torch
from tqdm import tqdm

from src.utils.samplers.base import Sampler
from src.utils.samplers.discrete_ot import OTPlanSampler
from src.utils.samplers.synthetic import SwissRollSampler, swiss_roll_transform


def get_GT_points(
    X_sampler: Sampler,
    Y_sampler: Sampler,
    mini_batch_sampler: OTPlanSampler,
    starting_points: list[torch.Tensor],
    num_ending_points: int = 64,
) -> list[torch.Tensor]:
    gt_Y_points = []
    for point in starting_points:
        _gt_points = []
        for _ in tqdm(range(num_ending_points)):
            x_start = torch.cat(
                (point[None, :], X_sampler.sample(num_ending_points - 1))
            )
            y_end = Y_sampler.sample(num_ending_points)
            p = mini_batch_sampler.get_map(x_start, y_end)
            point_true = y_end[torch.argmax(p[0])]
            _gt_points.append(point_true)
        gt_Y_points.append(torch.stack(_gt_points))

    return gt_Y_points


def match_gaussian_and_swiss_roll(
    Y_sampler: SwissRollSampler,
    starting_points: list[torch.Tensor],
    num_ending_points: int = 64,
    g_func: tp.Callable[[torch.Tensor], torch.Tensor] | None = None,
    noise_std: float = 0.1,
) -> torch.Tensor:
    assert g_func is not None, "g_func must be provided"

    gt_Y_points = []

    generator = Y_sampler.generator
    t_min = Y_sampler.t_min
    t_max = Y_sampler.t_max
    scale = Y_sampler.scale

    for point in tqdm(starting_points):
        n = num_ending_points

        x_batch = point.unsqueeze(0).repeat(n, 1)
        gx = g_func(x_batch)

        x_norm = torch.norm(gx, dim=-1)
        base = torch.tanh(x_norm)

        t = t_min + (t_max - t_min) * base
        t = t + noise_std * torch.randn_like(t)

        y_spiral = (
            swiss_roll_transform(t=t, generator=generator, noise=Y_sampler.noise)
            / scale
        )

        gt_Y_points.append(y_spiral)

    return torch.stack(gt_Y_points).to(Y_sampler.device)


def save_gt_points(
    starting_points: torch.Tensor,
    gt_Y_points: list[torch.Tensor],
    target_dir: str = "./data/sinkhorn_points",
):
    """Saves each ground truth array to a separate file named by its starting point."""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    for point_tensor, gt_Y in zip(starting_points, gt_Y_points):
        point = point_tensor.detach().cpu()
        filename = f"point_{point[0].item():.4f}_{point[1].item():.4f}.pt"
        target_path = target_dir / filename

        torch.save(gt_Y.detach().cpu(), target_path)
        print(f"Successfully saved: {target_path.name}")


def load_or_compute_gt_points(
    starting_points: torch.Tensor,
    X_sampler: Sampler,
    Y_sampler: Sampler,
    otp_sampler: OTPlanSampler,
    compute_func: tp.Callable[
        [Sampler, Sampler, OTPlanSampler, torch.Tensor, int], torch.Tensor
    ],
    target_dir: str = "./data/sinkhorn_points",
    num_ending_points: int = 1024,
) -> list[torch.Tensor]:
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    results = [None] * len(starting_points)
    indices_to_compute = []
    points_to_compute = []

    # 1. Check disk for existing points
    for i, point_tensor in enumerate(starting_points):
        point = point_tensor.detach().cpu()
        file_path = target_path / f"point_{point[0].item():.4f}_{point[1].item():.4f}.pt"

        if file_path.exists():
            results[i] = torch.load(file_path, map_location=point_tensor.device, weights_only=True)
        else:
            indices_to_compute.append(i)
            points_to_compute.append(point_tensor)

    # 2. Compute missing points if necessary
    if points_to_compute:
        print(f"Missing {len(points_to_compute)} points. Computing...")
        # Convert list of tensors back to a single batch tensor
        batch_to_compute = torch.stack(points_to_compute)

        new_gt_points = compute_func(
            X_sampler,
            Y_sampler,
            otp_sampler,
            batch_to_compute,
            num_ending_points=num_ending_points,
        )

        # 3. Save newly computed points
        save_gt_points(batch_to_compute, new_gt_points, target_path)

        # 4. Fill results list
        for idx, gt_data in zip(indices_to_compute, new_gt_points):
            results[idx] = gt_data

    return results

from typing import Callable, Literal

import numpy as np
import torch


def clip_by_norm(x: torch.Tensor, norm_thresh: float) -> torch.Tensor:
    """
    Clips the tensor x by its norms, ensuring that no tensor's norm exceeds norm_thresh.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, ...) where norms will be computed along all dimensions except batch.
        norm_thresh (float): The maximum allowable norm.

    Returns:
        torch.Tensor: The clipped tensor.
    """
    assert x.ndim > 1, "Input tensor must have at least 2 dimensions."

    # Compute the norms along all dimensions except the first (batch dimension)
    x_norms = torch.norm(x, dim=tuple(range(1, x.ndim)), keepdim=True)

    # Compute the scaling factors where the norm exceeds the threshold
    scaling_factors = torch.clamp(norm_thresh / x_norms, max=1.0)

    # Scale the input tensor by the calculated scaling factors
    return x * scaling_factors


def project_score_parts(
    score: torch.Tensor,
    cost_part: torch.Tensor,
    grad_proj_type: Literal["value", "norm", "none"],
    norm_thresh: float,
    value_thresh: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if grad_proj_type == "none":
        return score, cost_part
    if grad_proj_type == "value":
        score.clamp_(-value_thresh, value_thresh)
        cost_part.clamp_(-value_thresh, value_thresh)
        return score, cost_part
    if grad_proj_type == "norm":
        return clip_by_norm(score, norm_thresh), clip_by_norm(cost_part, norm_thresh)
    raise ValueError(f"unknown grad_proj_type: {grad_proj_type}")


def scale_step_and_noise(
    score: torch.Tensor,
    sampling_step: torch.Tensor,
    sampling_noise: torch.Tensor,
    thresh: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-batch step size and noise scale for the update (noise column shape (B, 1))."""
    if thresh is None:
        return sampling_step, sampling_noise.unsqueeze(-1)
    score_norms = torch.norm(score, dim=1)
    scaling_factors = torch.clamp(thresh / score_norms, max=1.0)
    step = sampling_step * scaling_factors
    noise_vec = (sampling_noise * torch.sqrt(scaling_factors)).unsqueeze(-1)
    return step, noise_vec


def sample_langevin_batch(
    score_function: Callable[
        [torch.Tensor, bool], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ],
    y: torch.Tensor,
    sampler_type: Literal["langevin", "pseudo"] = "langevin",
    step_size: float = 1e-3,
    noise: float = 0.005,
    num_iterations: int = 100,
    decay: float = 1.0,
    thresh: float | None = None,
    grad_proj_type: Literal["value", "norm", "none"] = "none",
    norm_thresh: float = 1.0,
    value_thresh: float = 0.01,
    data_projector: Callable[[torch.Tensor], torch.Tensor] = lambda x: x.clamp_(
        0.0, 1.0
    ),
    compute_stats: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    r"""
    Langevin: Y_{t + 1} = Y_{t} + 0.5 * step * score(Y_{t}) + noise * N(0, 1).

    Pseudo: add fixed per-batch noise to y, then the same score / grad-projection /
    optional thresh-based step scaling as Langevin, but the update has no Gaussian term.

    ``thresh`` and ``grad_proj_type`` apply to both modes; use ``grad_proj_type="none"``
    for classical Langevin without clipping.
    """
    batch_size = y.size(0)
    sampling_step = torch.full((batch_size,), step_size, device=y.device, dtype=y.dtype)
    sampling_noise = torch.full((batch_size,), noise, device=y.device, dtype=y.dtype)

    r_t = torch.zeros(1, device=y.device, dtype=y.dtype)
    cost_r_t = torch.zeros(1, device=y.device, dtype=y.dtype)
    potential_r_t = torch.zeros(1, device=y.device, dtype=y.dtype)
    noise_t = torch.zeros(1, device=y.device, dtype=y.dtype)

    noise_view = sampling_noise.view(batch_size, *([1] * (y.ndim - 1)))

    for _ in range(num_iterations):
        if sampler_type == "pseudo":
            y = y + noise_view

        score, cost_part, potential_part = score_function(y, stats=True)
        score, cost_part = project_score_parts(
            score, cost_part, grad_proj_type, norm_thresh, value_thresh
        )
        step, noise_vec = scale_step_and_noise(
            score, sampling_step, sampling_noise, thresh
        )

        if sampler_type == "langevin":
            z_t = torch.randn_like(y)
            y = y + 0.5 * step[:, None] * score + noise_vec * z_t
            if compute_stats:
                noise_t += (noise_vec.squeeze(-1) * torch.norm(z_t, dim=1)).mean()
        else:
            y = y + 0.5 * step[:, None] * score

        if compute_stats:
            r_t += (0.5 * step * torch.norm(score, dim=1)).mean()
            cost_r_t += (0.5 * step * torch.norm(cost_part, dim=1)).mean()
            potential_r_t += (0.5 * step * torch.norm(potential_part, dim=1)).mean()

        sampling_step.mul_(decay)
        if sampler_type == "langevin":
            sampling_noise.mul_(np.sqrt(decay))

        y = data_projector(y)

    if not compute_stats:
        return y, {}

    n_it = float(num_iterations)
    noise_stat = (
        (noise_t / n_it).item()
        if sampler_type == "langevin"
        else sampling_noise.mean().item()
    )
    return y, {
        "neg_energy_t": (r_t / n_it).item(),
        "cost_t": (cost_r_t / n_it).item(),
        "potential_t": (potential_r_t / n_it).item(),
        "noise": noise_stat,
    }

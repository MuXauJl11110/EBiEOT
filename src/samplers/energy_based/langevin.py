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


def sample_langevin_batch(
    score_function: Callable[[torch.Tensor, bool], tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    y: torch.Tensor,
    step_size: float = 1e-3,
    noise: float = 0.005,
    num_iterations: int = 100,
    decay: float = 1.0,
    thresh: float | None = None,
    data_projector: Callable[[torch.Tensor], torch.Tensor] = lambda x: x.clamp_(0.0, 1.0),
    compute_stats: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor:
    r"""
    Overall, langevin step looks as:
    Y_{t + 1} = Y_{t} + 0.5 * step_size * score(Y_{t}) + noise * N(0, 1)
    """
    # make step and noise to be data-dimensional
    batch_size = y.size(0)
    sampling_step = torch.full((batch_size,), step_size, device=y.device)  # [bs]
    sampling_noise = torch.full((batch_size,), noise, device=y.device)  # [bs]

    # statistics
    r_t = torch.zeros(1).to(y.device)
    cost_r_t = torch.zeros(1).to(y.device)
    score_r_t = torch.zeros(1).to(y.device)
    noise_t = torch.zeros(1).to(y.device)

    # langevin iterations
    for _ in range(num_iterations):
        z_t = torch.randn_like(y)
        score, cost_part, score_part = score_function(y, stats=True)

        # adjusting discretization step
        if thresh is None:
            step = sampling_step
            noise = sampling_noise
        else:
            score_norms = torch.norm(score, dim=1)  # [bs]
            scaling_factors = torch.clamp(thresh / score_norms, max=1.0)
            step = sampling_step * scaling_factors  # [bs]
            noise = sampling_noise * torch.sqrt(scaling_factors)[:, None]  # [bs]

        # Langevin dynamics
        y = y + 0.5 * step[:, None] * score + noise[:, None] * z_t

        # stats calculation
        if compute_stats:
            r_t += (0.5 * step * torch.norm(score, dim=1)).mean()
            cost_r_t += (0.5 * step * torch.norm(cost_part, dim=1)).mean()
            score_r_t += (0.5 * step * torch.norm(score_part, dim=1)).mean()
            noise_t += (noise * torch.norm(z_t, dim=1)).mean()

        sampling_step *= decay
        sampling_noise *= np.sqrt(decay)

        # Project data to images compact
        y = data_projector(y)

    if not compute_stats:
        return y
    return y, r_t / num_iterations, cost_r_t / num_iterations, score_r_t / num_iterations, noise_t / num_iterations


# WARNING: Do we need to scale noise? Where did this code come from?
def sample_pseudo_langevin_batch(
    score_function: Callable[[torch.Tensor, bool], tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    y: torch.Tensor,
    step_size: float = 10.0,
    num_iterations: int = 60,
    noise: float = 0.005,
    decay: float = 1.0,
    grad_proj_type: Literal["value", "norm", "none"] = "value",
    norm_thresh: float = 1.0,
    value_thresh: float = 0.01,
    data_projector: Callable[[torch.Tensor], torch.Tensor] = lambda x: x.clamp_(0.0, 1.0),
    compute_stats: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor:
    # make step and noise to be data-dimensional
    batch_size = y.size(0)
    sampling_step = torch.full((batch_size,), step_size, device=y.device)  # [bs]
    sampling_noise = torch.full((batch_size,), noise, device=y.device)  # [bs]

    # statistics
    r_t = torch.zeros(1).to(y.device)
    cost_r_t = torch.zeros(1).to(y.device)
    score_r_t = torch.zeros(1).to(y.device)

    # langevin iterations
    for _ in range(num_iterations):
        y += sampling_noise
        score, cost_part, score_part = score_function(y, stats=True)

        if grad_proj_type == "none":
            pass
        elif grad_proj_type == "value":
            score.clamp_(-value_thresh, value_thresh)
            cost_part.clamp_(-value_thresh, value_thresh)
        elif grad_proj_type == "norm":
            score = clip_by_norm(score, norm_thresh)
            cost_part = clip_by_norm(cost_part, norm_thresh)
        else:
            raise ValueError("unknown proj_type")

        y = y + 0.5 * sampling_step * score
        sampling_step *= decay

        if compute_stats:
            r_t += (0.5 * sampling_step * torch.norm(score, dim=1)).mean()
            cost_r_t += (0.5 * sampling_step * torch.norm(cost_part, dim=1)).mean()
            score_r_t += (0.5 * sampling_step * torch.norm(score_part, dim=1)).mean()

        # Project data to images compact
        y = data_projector(y)

    if not compute_stats:
        return y

    return (
        y,
        r_t / num_iterations,
        cost_r_t / num_iterations,
        score_r_t / num_iterations,
        sampling_noise.norm(dim=1).mean(),
    )

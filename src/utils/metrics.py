from typing import Callable

import numpy as np
import torch
from comet_ml import Experiment
from geomloss import SamplesLoss
from src.samplers.primary import Sampler


def symmetrize(X: torch.Tensor) -> torch.Tensor:
    """Symmetrizes a square matrix."""
    return (X + X.mT) / 2.0


def sqrtm_psd(A: torch.Tensor) -> torch.Tensor:
    """
    Computes the matrix square root of a symmetric positive semi-definite tensor.
    Using eigendecomposition is stable for covariance matrices.
    """
    L, Q = torch.linalg.eigh(symmetrize(A))
    # Clamp negative eigenvalues that may arise due to numerical instability
    L = torch.clamp(L, min=0.0)
    return Q @ torch.diag_embed(torch.sqrt(L)) @ Q.mT


def compute_BW_UVP(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # x: generated samples
    # y: ground-truth samples
    y_covariance = torch.cov(y.T)
    y_mu = y.mean(dim=0)

    x_covariance = torch.cov(x.T)
    x_mu = x.mean(dim=0)

    x_covariance_sqrt = symmetrize(sqrtm_psd(x_covariance))

    mu_term = 0.5 * torch.sum((y_mu - x_mu) ** 2)

    # Inner matrix: sqrt(Sigma_m) @ Sigma_t @ sqrt(Sigma_m)
    inner_matrix = x_covariance_sqrt @ y_covariance @ x_covariance_sqrt
    inner_sqrt = symmetrize(sqrtm_psd(inner_matrix))

    covariance_term = 0.5 * torch.trace(x_covariance) + 0.5 * torch.trace(y_covariance) - torch.trace(inner_sqrt)

    BW = mu_term + covariance_term
    BW_UVP = 100 * (BW / (0.5 * torch.trace(y_covariance)))
    return BW_UVP


def compute_mmd(x: torch.Tensor, y: torch.Tensor, sigma: float = 10.0, scale: float = 1.0) -> torch.Tensor:
    """
    Memory-efficient MMD implementation in PyTorch using the Gaussian RBF kernel.
    Utilizes torch.cdist for optimized pairwise distance computations.
    """
    gamma = 1.0 / (2 * sigma**2)

    # Compute squared pairwise distances
    xx = torch.cdist(x, x, p=2.0) ** 2
    xy = torch.cdist(x, y, p=2.0) ** 2
    yy = torch.cdist(y, y, p=2.0) ** 2

    k_xx = torch.exp(-gamma * xx)
    k_xy = torch.exp(-gamma * xy)
    k_yy = torch.exp(-gamma * yy)

    # Unbiased/minimum-variance estimate
    MMD = scale * (k_xx.mean() + k_yy.mean() - 2 * k_xy.mean())
    return MMD


def compute_sinkhorn_divergence(xs: torch.Tensor, ys: torch.Tensor, epsilon: float = 1.0) -> torch.Tensor:
    """
    Computes the unbiased Sinkhorn divergence between two point clouds.

    Requires the 'geomloss' package which is the standard PyTorch equivalent
    to ott-jax for optimal transport distances.
    """
    assert xs.ndim == 2 and ys.ndim == 2, "Inputs must be 2D tensors"
    assert xs.shape[1] == ys.shape[1], "Feature dimensions must match"

    # geomloss uses 'blur' as the scaling parameter, which corresponds to the
    # square root of the regularizer epsilon in standard entropy-regularized OT.
    blur_radius = torch.sqrt(torch.tensor(epsilon, device=xs.device))

    loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=blur_radius.item())

    return loss_fn(xs, ys)


def compute_metrics(
    models_dict: dict[str, torch.nn.Module],
    metrics_dict: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    X_sampler: Sampler,
    Y_sampler: Sampler,
    starting_points: torch.Tensor,  # [num_starting points, dim]
    gt_Y_points: list[np.ndarray],
    num_samples: int,
    experiment: Experiment | None = None,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """
    Computes unconditional and conditional evaluation metrics for a set of models.

    Unconditional metrics evaluate marginal distribution matching by drawing
    random samples from the X and Y marginals. Conditional metrics evaluate the
    accuracy of the learned conditional transport plan by comparing model predictions
    for specific starting points against their ground-truth conditional targets.

    Args:
        models_dict: Dictionary mapping model names to PyTorch modules.
        metrics_dict: Dictionary mapping metric names to callable metric functions.
            Functions should take two PyTorch tensors (pred, target) and return a scalar.
        X_sampler: Sampler for the source distribution (marginal X).
        Y_sampler: Sampler for the target distribution (marginal Y).
        starting_points: Tensor of shape (N, dim_x) containing specific points in X
            for conditional evaluation.
        gt_Y_points: List of numpy arrays where the i-th array contains the
            ground-truth translated points for the i-th starting point.
        num_samples: Number of samples to draw for unconditional metric computation.
        experiment: Optional experiment tracking object (e.g., Weights & Biases,
            CometML) that supports a `.log_metrics()` or `.log()` method.

    Returns:
        A tuple containing two dictionaries:
            - unconditional_metrics: Mappings of model names to unconditional metric values.
            - conditional_metrics: Mappings of model names to averaged conditional metric values.
    """
    device = starting_points.device

    # Generate unconditional evaluation batches
    x_samples = X_sampler.sample(num_samples).to(device)
    y_samples = Y_sampler.sample(num_samples).to(device)

    # ---------------------------------------------------------
    # 1. Compute Unconditional Metrics
    # ---------------------------------------------------------
    unconditional_metrics = {}
    for model_name, model in models_dict.items():
        model.eval()
        with torch.no_grad():
            y_pred = model(x_samples)

        log_metrics = {}
        for metric_name, metric in metrics_dict.items():
            metric_val = metric(y_pred, y_samples).item()
            log_metrics[metric_name] = metric_val

        unconditional_metrics[model_name] = log_metrics

        if experiment is not None:
            experiment.log_metrics(log_metrics, prefix=f"Unconditional/{model_name}")

    # ---------------------------------------------------------
    # 2. Compute Conditional Metrics
    # ---------------------------------------------------------
    conditional_metrics = {}
    for model_name, model in models_dict.items():
        model.eval()

        # Accumulate metrics across all starting points to average later
        accumulated_metrics = {metric_name: [] for metric_name in metrics_dict.keys()}

        with torch.no_grad():
            for point, gt_Y_point in zip(starting_points, gt_Y_points):
                num_ending_points = gt_Y_point.shape[0]
                repeated_starting_points = point[None, :].repeat(num_ending_points, 1)

                point_pred = model(repeated_starting_points)
                # Convert GT numpy array to tensor and move to identical device
                gt_tensor = torch.tensor(gt_Y_point, dtype=torch.float64, device=device)

                for metric_name, metric in metrics_dict.items():
                    metric_val = metric(point_pred, gt_tensor).item()
                    accumulated_metrics[metric_name].append(metric_val)

        # Average the metrics across all starting points
        log_metrics = {metric_name: sum(vals) / len(vals) for metric_name, vals in accumulated_metrics.items()}

        conditional_metrics[model_name] = log_metrics

        if experiment is not None:
            experiment.log_metrics(log_metrics, prefix=f"Conditional/{model_name}")

    return unconditional_metrics, conditional_metrics

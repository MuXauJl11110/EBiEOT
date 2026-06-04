import typing as tp

import torch
from comet_ml import Experiment

from src.ebieot.base import BaseEBiEOT
from src.utils.samplers.base import Sampler


def update_average(
    model_tgt: torch.nn.Module, model_src: torch.nn.Module, beta: float = 0.99
) -> None:
    with torch.no_grad():
        param_dict_src = dict(model_src.named_parameters())
        for p_name, p_tgt in model_tgt.named_parameters():
            p_src = param_dict_src[p_name]
            assert p_src is not p_tgt, "Source and target parameters must be different"
            p_tgt.data.copy_(beta * p_tgt.data + (1.0 - beta) * p_src.data)


def compute_loss(
    model: BaseEBiEOT,
    X_unpaired: torch.Tensor,
    Y_unpaired: torch.Tensor,
    X_paired: torch.Tensor,
    Y_paired: torch.Tensor,
) -> float:
    output_unpaired = model.compute_unpaired_loss(X_unpaired, Y_unpaired)
    unpaired_loss = output_unpaired["loss"]

    output_paired = model.compute_paired_loss(X_paired, Y_paired)
    paired_loss = output_paired["loss"]
    return (paired_loss + unpaired_loss).item()


def compute_metrics(
    models_dict: dict[str, torch.nn.Module],
    metrics_dict: dict[str, tp.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    X_sampler: Sampler,
    Y_sampler: Sampler,
    starting_points: torch.Tensor,  # [num_starting points, dim]
    gt_Y_points: list[torch.Tensor],
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
        gt_Y_points: List of tensors where the i-th tensor contains the
            ground-truth translated points for the i-th starting point.
        num_samples: Number of samples to draw for unconditional metric computation.
        experiment: Optional experiment tracking object (e.g., CometML) that supports a `.log_metrics()` or `.log()` method.

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
        log_metrics = {
            metric_name: sum(vals) / len(vals)
            for metric_name, vals in accumulated_metrics.items()
        }

        conditional_metrics[model_name] = log_metrics

        if experiment is not None:
            experiment.log_metrics(log_metrics, prefix=f"Conditional/{model_name}")

    return unconditional_metrics, conditional_metrics

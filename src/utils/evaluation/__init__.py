"""Distribution-matching evaluation metrics."""

from src.utils.evaluation.metrics import (
    compute_mmd,
    compute_sinkhorn_divergence,
)

__all__ = ["compute_mmd", "compute_sinkhorn_divergence"]

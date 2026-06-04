from collections.abc import Callable

import torch
from geomloss import SamplesLoss

Kernel = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def median_heuristic(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    z = torch.cat([x, y], dim=0)

    with torch.no_grad():
        dists = torch.cdist(z, z, p=2)
        dists = dists[dists > 0]
        sigma = torch.median(dists)

    return sigma


def rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float = 10.0) -> torch.Tensor:
    gamma = 1.0 / (2 * sigma**2)

    dist2 = torch.cdist(x, y) ** 2
    return torch.exp(-gamma * dist2)


def mixture_kernel(
    x: torch.Tensor, y: torch.Tensor, basis_kernels: list[Kernel]
) -> torch.Tensor:

    return sum(kernel(x, y) for kernel in basis_kernels) / len(basis_kernels)


def compute_mmd(x: torch.Tensor, y: torch.Tensor, kernel: Kernel) -> torch.Tensor:
    n = x.shape[0]
    m = y.shape[0]

    if n < 2 or m < 2:
        raise ValueError("Need at least 2 samples per set.")

    k_xx = kernel(x, x)
    k_yy = kernel(y, y)
    k_xy = kernel(x, y)

    k_xx.fill_diagonal_(0.0)
    k_yy.fill_diagonal_(0.0)

    mmd_xx = k_xx.sum() / (n * (n - 1))
    mmd_yy = k_yy.sum() / (m * (m - 1))
    mmd_xy = k_xy.mean()

    mmd = mmd_xx + mmd_yy - 2.0 * mmd_xy

    return mmd


def compute_sinkhorn_divergence(
    xs: torch.Tensor, ys: torch.Tensor, epsilon: float = 0.001
) -> torch.Tensor:
    assert xs.ndim == 2 and ys.ndim == 2, "Inputs must be 2D tensors"
    assert xs.shape[1] == ys.shape[1], "Feature dimensions must match"

    # geomloss uses 'blur' as the scaling parameter, which corresponds to the
    # square root of the regularizer epsilon in standard entropy-regularized OT.
    blur_radius = torch.sqrt(torch.tensor(epsilon, device=xs.device))

    loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=blur_radius.item())

    return loss_fn(xs, ys)

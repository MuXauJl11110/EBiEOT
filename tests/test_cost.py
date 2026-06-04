import torch

from src.ebieot.costs.base import BaseLSECost


def reference_lse_forward(
    cost: BaseLSECost, batched_x: torch.Tensor, batched_y: torch.Tensor
) -> torch.Tensor:
    """Same math as :class:`BaseLSECost` / ``func``, evaluated row-wise without ``vmap``."""
    eps = cost.epsilon
    out = torch.empty(
        batched_x.shape[0], device=batched_x.device, dtype=batched_x.dtype
    )
    for i in range(batched_x.shape[0]):
        log_v_m = cost.compute_log_v_m(batched_x[i])
        b_m = cost.compute_b_m(batched_x[i])
        b_ty = (b_m * batched_y[i]).sum(dim=1)
        out[i] = -eps * torch.logsumexp(log_v_m + b_ty / eps, dim=0)
    return out


def test_lse_cost_forward_matches_reference(
    cost: BaseLSECost,
    batched_x: torch.Tensor,
    batched_y: torch.Tensor,
):
    got = cost.forward(batched_x, batched_y)
    ref = reference_lse_forward(cost, batched_x, batched_y)
    assert torch.allclose(got, ref, rtol=1e-5, atol=1e-6)

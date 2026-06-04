import math

import torch

from src.ebieot.costs.lse import MLPLSECost
from src.ebieot.ebieot_gmm import EbieotGmm


def _tiny_cost(y_dim: int = 2, m: int = 3):
    return MLPLSECost(
        log_v_m_hidden_channels=[m],
        b_m_hidden_channels=[m * y_dim],
        x_dim=y_dim,
        y_dim=y_dim,
        m_potentials=m,
        epsilon=1.0,
    )


def reference_f(
    batched_y: torch.Tensor,
    epsilon: torch.Tensor,
    log_w_n: torch.Tensor,
    a_n: torch.Tensor,
    A_n: torch.Tensor,
) -> torch.Tensor:
    """Naive mixture log-density times ``epsilon`` (matches :meth:`EbieotGmm.f`)."""
    scale_sq = epsilon * A_n  # [N x y_dim]
    diff = batched_y[:, None, :] - a_n[None, :, :]  # [bs x N x y_dim]
    log_2pi = math.log(2 * math.pi)
    log_comp = -0.5 * (
        log_2pi + torch.log(scale_sq)[None, :, :] + (diff**2) / scale_sq[None, :, :]
    ).sum(dim=2)
    log_mix = torch.logsumexp(log_w_n[None, :] + log_comp, dim=1)
    log_Z_pi = torch.logsumexp(log_w_n, dim=0)
    return epsilon * (log_mix - log_Z_pi)


def reference_log_Z_nm(
    epsilon: torch.Tensor,
    log_w_n: torch.Tensor,
    a_n: torch.Tensor,
    A_n: torch.Tensor,
    log_v_m: torch.Tensor,
    b_m: torch.Tensor,
) -> torch.Tensor:
    """Loop version of :meth:`EbieotGmm.log_Z_nm` (small ``N``, ``M`` only)."""
    bs, M, y_dim = b_m.shape
    N = log_w_n.shape[0]
    out = torch.zeros(bs, N, M, device=b_m.device, dtype=b_m.dtype)
    for i in range(bs):
        for n in range(N):
            for m in range(M):
                b = b_m[i, m]
                corr = ((b * A_n[n] + 2 * a_n[n]) * b).sum()
                out[i, n, m] = log_v_m[i, m] + log_w_n[n] + 0.5 * corr / epsilon + 1e-12
    return out


def test_f_matches_naive_mixture():
    torch.manual_seed(0)
    cost = _tiny_cost()
    model = EbieotGmm(y_dim=2, n_potentials=4, cost=cost, epsilon=1.0)
    batched_y = torch.randn(8, 2)
    log_w_n = model.log_w_n()
    a_n = model.a_n()
    A_n = model.A_n()
    got = model.f(batched_y, log_w_n, a_n, A_n)
    ref = reference_f(batched_y, model.epsilon, log_w_n, a_n, A_n)
    assert torch.allclose(got, ref, rtol=1e-4, atol=1e-5)


def test_log_Z_nm_matches_reference():
    torch.manual_seed(0)
    cost = _tiny_cost()
    model = EbieotGmm(y_dim=2, n_potentials=4, cost=cost, epsilon=1.0)
    batched_x = torch.randn(8, 2)
    log_w_n = model.log_w_n()
    a_n = model.a_n()
    A_n = model.A_n()
    log_v_m = model.cost.log_v_m(batched_x)
    b_m = model.cost.b_m(batched_x)
    got = model.log_Z_nm(log_w_n, a_n, A_n, log_v_m, b_m)
    ref = reference_log_Z_nm(model.epsilon, log_w_n, a_n, A_n, log_v_m, b_m)
    assert torch.allclose(got, ref, rtol=1e-5, atol=1e-6)


def test_log_w_n_is_log_probability_simplex():
    torch.manual_seed(0)
    cost = _tiny_cost()
    model = EbieotGmm(y_dim=2, n_potentials=7, cost=cost, epsilon=2.0)
    model._log_w_n.data = torch.randn(model.n_potentials)

    w = torch.exp(model.log_w_n())
    assert torch.isfinite(w).all()
    assert torch.allclose(w.sum(), torch.tensor(1.0), atol=1e-5, rtol=0)


def test_A_n_softplus_positive_and_init_near_A_diagonal_init():
    cost = _tiny_cost()
    model = EbieotGmm(y_dim=2, n_potentials=4, cost=cost, A_diagonal_init=0.1)
    A = model.A_n()
    assert (A > 0).all()
    assert torch.isfinite(A).all()
    assert torch.allclose(A, torch.full_like(A, 0.1), atol=0.02, rtol=0.1)


def test_paired_unpaired_loss_scaled_by_epsilon():
    cost = _tiny_cost()
    model = EbieotGmm(y_dim=2, n_potentials=4, cost=cost, epsilon=0.5)
    x = torch.randn(8, 2)
    y = torch.randn(8, 2)
    paired = model.compute_paired_loss(x, y)["loss"]
    unpaired = model.compute_unpaired_loss(x, y)["loss"]
    assert paired == model.cost(x, y).mean() / model.epsilon
    assert torch.isfinite(unpaired)

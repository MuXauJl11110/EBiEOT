import torch

from src.costs.lse import MLPLSECost
from src.models.gmm_based import GMMEOT


def _tiny_cost(y_dim: int = 2, m: int = 3):
    return MLPLSECost(
        log_v_m_hidden_channels=[m],
        b_m_hidden_channels=[m * y_dim],
        x_dim=y_dim,
        y_dim=y_dim,
        m_potentials=m,
        epsilon=1.0,
    )


def test_log_w_n_is_log_probability_simplex():
    torch.manual_seed(0)
    cost = _tiny_cost()
    model = GMMEOT(y_dim=2, n_potentials=7, cost=cost, epsilon=2.0)
    model._log_w_n.data = torch.randn(model.n_potentials)

    w = torch.exp(model.log_w_n())
    assert torch.isfinite(w).all()
    assert torch.allclose(w.sum(), torch.tensor(1.0), atol=1e-5, rtol=0)


def test_A_n_softplus_positive_and_init_near_A_diagonal_init():
    cost = _tiny_cost()
    model = GMMEOT(y_dim=2, n_potentials=4, cost=cost, A_diagonal_init=0.1)
    A = model.A_n()
    assert (A > 0).all()
    assert torch.isfinite(A).all()
    assert torch.allclose(A, torch.full_like(A, 0.1), atol=0.02, rtol=0.1)


def test_paired_unpaired_loss_scaled_by_epsilon():
    cost = _tiny_cost()
    model = GMMEOT(y_dim=2, n_potentials=4, cost=cost, epsilon=0.5)
    x = torch.randn(8, 2)
    y = torch.randn(8, 2)
    paired = model.compute_paired_loss(x, y)["loss"]
    unpaired = model.compute_unpaired_loss(x, y)["loss"]
    assert paired == model.cost(x, y).mean() / model.epsilon
    assert torch.isfinite(unpaired)

import torch
from torch.distributions.categorical import Categorical
from torch.distributions.independent import Independent
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal

from src.light_gcot import LightGCOT


def test_quadratic_cost(D: LightGCOT, batched_x: torch.Tensor, batched_y: torch.Tensor, x_dim: int, y_dim: int):
    if D.m_potentials == 1 and x_dim == y_dim:
        b_m = D.compute_b_m(batched_x)
        B_m = D.compute_B_m(batched_x)
        log_v_m = D.compute_log_v_m(batched_x)
        c = D.compute_cost(b_m, B_m, log_v_m, batched_y=batched_y)
        _c = 0.5 * torch.sum((batched_x - batched_y) ** 2, dim=1)

        assert torch.allclose(c, _c + 0.5 * D.y_dim * torch.log(2 * torch.pi * D.epsilon))


def test_custom_cost(
    D: LightGCOT, batched_x: torch.Tensor, batched_y: torch.Tensor, batch_size: int, x_dim: int, y_dim: int
):
    if D.m_potentials == 2 and x_dim == y_dim:
        D.cost_function = "Alexander's"  # mock
        b_m = D.compute_b_m(batched_x)
        B_m = D.compute_B_m(batched_x)
        log_v_m = D.compute_log_v_m(batched_x)
        c = D.compute_cost(b_m, B_m, log_v_m, batched_y=batched_y)

        S = torch.ones(D.m_potentials, y_dim).repeat(batch_size, 1, 1)
        r = torch.stack((batched_x, -batched_x), dim=1)
        mix = Categorical(probs=torch.tensor([0.5, 0.5]))
        comp = Independent(Normal(loc=r, scale=torch.sqrt(D.epsilon * S)), 1)
        gmm = MixtureSameFamily(mix, comp)
        _c = -D.epsilon * gmm.log_prob(batched_y)

        assert torch.allclose(c, _c)

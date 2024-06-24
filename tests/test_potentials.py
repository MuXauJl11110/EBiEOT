import torch

from src.light_gcot import LightGCOT


def test_compute_primal_potential(D: LightGCOT, batch_size: int, batched_y: torch.Tensor):
    se = torch.zeros((batch_size, D.n_potentials))  # sum of exponents
    for n in range(D.n_potentials):
        diff = batched_y - D.a[n]  # [bs x y_dim] - [y_dim] = [bs x y_dim]
        se[:, n] += torch.exp(D.log_w[n]) * torch.exp(
            (-0.5 / D.epsilon) * torch.sum(diff * D.A_diagonal_matrix[n] * diff, dim=1)
        )

    _f = D.epsilon * torch.log(torch.sum(se, dim=1))
    f = D.compute_primal_potential(batched_y)
    assert torch.allclose(f, _f)

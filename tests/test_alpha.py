import torch

from src.light_gcot import LightGCOT


def test_compute_alpha(D: LightGCOT):
    c = D.compute_c()
    G_xx, G_xy, G_yx, G_yy = D.compute_G()
    _log_alpha = torch.zeros((D.n_potentials, D.m_potentials))

    for n in range(D.n_potentials):
        for m in range(D.m_potentials):
            _log_alpha[n][m] = (
                D.log_w[n]
                + D.log_v[m]
                - 0.5
                * (
                    c[n][m] / D.epsilon
                    - torch.log(2 * torch.pi * D.epsilon) * D.y_dim
                    + torch.sum(torch.log(G_yy[n][m]))
                )
                # - torch.log(2 * torch.pi * D.epsilon) * (D.x_dim - D.y_dim))
                # - torch.sum(torch.log(G_xx[n][m]))
            )

    log_alpha = D.compute_log_alpha(G_xx, G_xy, G_yx, G_yy)
    assert torch.allclose(log_alpha, _log_alpha)

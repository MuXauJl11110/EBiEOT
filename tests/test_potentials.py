import torch

from src.light_gcot import LightGCOT


def test_compute_log_quadratic(D: LightGCOT, batch_size: int):
    x = torch.randn((batch_size, D.x_dim))
    G_xx, G_xy, G_yx, G_yy = D.compute_G()
    g_x, _ = D.compute_g()

    _log_quadratic = torch.zeros((D.n_potentials, D.m_potentials))
    for n in range(D.n_potentials):
        for m in range(D.m_potentials):
            diff = x - g_x[n][m]
            _log_quadratic[n][m] = -0.5 / D.epsilon * torch.sum(diff * G_xx[n][m] * diff)

    log_quadratic = D.compute_log_quadratic(x, g_x, G_xx, G_xy, G_yx, G_yy)
    assert torch.allclose(log_quadratic, _log_quadratic)


def test_compute_primal_potential(D: LightGCOT, batch_size: int):
    y = torch.randn((batch_size, D.y_dim))

    se = 0  # sum of exponents
    for n in range(D.n_potentials):
        diff = y - D.a_y[n]  # [bs x y_dim] - [y_dim] = [bs x y_dim]
        se += torch.exp(D.log_w[n]) * torch.exp(
            (-0.5 / D.epsilon) * torch.sum(diff * D.A_yy_diagonal_matrix[n] * diff)
        )

    _f = D.epsilon * torch.log(se)
    f = D.compute_primal_potential(y)
    assert torch.allclose(f, _f)


def test_compute_dual_potential(D: LightGCOT, batch_size: int):
    x = torch.randn((batch_size, D.x_dim))
    G_xx, _, _, G_yy = D.compute_G()
    g_x, _ = D.compute_g()
    c = D.compute_c()

    se = torch.zeros((D.n_potentials, D.m_potentials))
    for n in range(D.n_potentials):
        for m in range(D.m_potentials):
            diff = x - g_x[n][m]  # [bs x y_dim] - [y_dim] = [bs x y_dim]
            se[n][m] += torch.exp(
                D.log_w[n]
                + D.log_v[m]
                + 0.5 * (torch.log(2 * torch.pi * D.epsilon) * D.y_dim - torch.sum(torch.log(G_yy[n][m])))
                - 0.5 / D.epsilon * c[n][m]
                - 0.5 / D.epsilon * torch.sum(diff * G_xx[n][m] * diff)
            )

    print(se[0][0])
    _f_c = -D.epsilon * torch.log(torch.sum(se))
    f_c = D.compute_dual_potential(x)
    assert torch.allclose(f_c, _f_c)

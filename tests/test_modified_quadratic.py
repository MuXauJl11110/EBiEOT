import torch

from src.light_gcot import LightGCOT


def test_compute_G(D: LightGCOT):
    _G_xx = torch.zeros((D.n_potentials, D.m_potentials, D.x_dim))
    _G_yy = torch.zeros((D.n_potentials, D.m_potentials, D.y_dim))
    for n in range(D.n_potentials):
        for m in range(D.m_potentials):
            _G_xx[n][m] = D.A_xx_diagonal_matrix[n] + D.B_xx_diagonal_matrix[m]
            _G_yy[n][m] = D.A_yy_diagonal_matrix[n] + D.B_yy_diagonal_matrix[m]

    G_xx, _, _, G_yy = D.compute_G()
    assert torch.allclose(G_xx, _G_xx)
    assert torch.allclose(G_yy, _G_yy)


def test_compute_g(D: LightGCOT):
    _g_x = torch.zeros((D.n_potentials, D.m_potentials, D.x_dim))
    _g_y = torch.zeros((D.n_potentials, D.m_potentials, D.y_dim))

    for n in range(D.n_potentials):
        for m in range(D.m_potentials):
            _g_x[n][m] = (D.A_xx_diagonal_matrix[n] * D.a_x[n] + D.B_xx_diagonal_matrix[m] * D.b_x[m]) / (
                D.A_xx_diagonal_matrix[n] + D.B_xx_diagonal_matrix[m]
            )
            _g_y[n][m] = (D.A_yy_diagonal_matrix[n] * D.a_y[n] + D.B_yy_diagonal_matrix[m] * D.b_y[m]) / (
                D.A_yy_diagonal_matrix[n] + D.B_yy_diagonal_matrix[m]
            )

    g_x, g_y = D.compute_g()
    assert torch.allclose(g_x, _g_x)
    assert torch.allclose(g_y, _g_y)


def test_compute_c(D: LightGCOT):
    _c = torch.zeros((D.n_potentials, D.m_potentials))

    for n in range(D.n_potentials):
        for m in range(D.m_potentials):
            _c[n][m] = torch.sum(
                D.A_yy_diagonal_matrix[n]
                * D.B_yy_diagonal_matrix[m]
                * ((D.a_y[n] - D.b_y[m]) ** 2)
                / (D.A_yy_diagonal_matrix[n] + D.B_yy_diagonal_matrix[m])
            )

    c = D.compute_c()
    assert torch.allclose(c, _c)

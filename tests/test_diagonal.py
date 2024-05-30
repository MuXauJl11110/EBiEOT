import pytest
import torch

from src.light_gcot import LightGCOT

# @pytest.mark.parametrize("n_potentials", [100, 200])
# @pytest.mark.parametrize("m_potentials", [300, 400])
# @pytest.mark.parametrize("A_diagonal_init", [0.1])
# @pytest.mark.parametrize("B_diagonal_init", [0.3])
# def test_compute_G(n_potentials: int, m_potentials: int, A_diagonal_init: float, B_diagonal_init: float):
#     D = LightGCOT(
#         n_potentials=n_potentials,
#         m_potentials=m_potentials,
#         A_diagonal_init=A_diagonal_init,
#         B_diagonal_init=B_diagonal_init,
#     )
#     _G_xx = torch.zeros((n_potentials, m_potentials, D.x_dim))
#     _G_yy = torch.zeros((n_potentials, m_potentials, D.y_dim))
#     for n in range(n_potentials):
#         for m in range(m_potentials):
#             _G_xx[n][m] = D.A_xx_diagonal_matrix[n] + D.B_xx_diagonal_matrix[m]
#             _G_yy[n][m] = D.A_yy_diagonal_matrix[n] + D.B_yy_diagonal_matrix[m]

#     G_xx, _, _, G_yy = D.compute_G()
#     assert torch.allclose(G_xx, _G_xx)
#     assert torch.allclose(G_yy, _G_yy)


# @pytest.mark.parametrize("n_potentials", [100, 200])
# @pytest.mark.parametrize("m_potentials", [300, 400])
# @pytest.mark.parametrize("A_diagonal_init", [0.1])
# @pytest.mark.parametrize("B_diagonal_init", [0.3])
# def test_compute_g(n_potentials: int, m_potentials: int, A_diagonal_init: float, B_diagonal_init: float):
#     D = LightGCOT(
#         n_potentials=n_potentials,
#         m_potentials=m_potentials,
#         A_diagonal_init=A_diagonal_init,
#         B_diagonal_init=B_diagonal_init,
#     )
#     _g_x = torch.zeros((n_potentials, m_potentials, D.x_dim))
#     _g_y = torch.zeros((n_potentials, m_potentials, D.y_dim))

#     for n in range(n_potentials):
#         for m in range(m_potentials):
#             _g_x[n][m] = (D.A_xx_diagonal_matrix[n] * D.a_x[n] + D.B_xx_diagonal_matrix[m] * D.b_x[m]) / (
#                 D.A_xx_diagonal_matrix[n] + D.B_xx_diagonal_matrix[m]
#             )
#             _g_y[n][m] = (D.A_yy_diagonal_matrix[n] * D.a_y[n] + D.B_yy_diagonal_matrix[m] * D.b_y[m]) / (
#                 D.A_yy_diagonal_matrix[n] + D.B_yy_diagonal_matrix[m]
#             )

#     g_x, g_y = D.compute_g()
#     assert torch.allclose(g_x, _g_x)
#     assert torch.allclose(g_y, _g_y)


# @pytest.mark.parametrize("n_potentials", [100, 200])
# @pytest.mark.parametrize("m_potentials", [300, 400])
# @pytest.mark.parametrize("A_diagonal_init", [0.1])
# @pytest.mark.parametrize("B_diagonal_init", [0.3])
# def test_compute_c(n_potentials: int, m_potentials: int, A_diagonal_init: float, B_diagonal_init: float):
#     D = LightGCOT(
#         n_potentials=n_potentials,
#         m_potentials=m_potentials,
#         A_diagonal_init=A_diagonal_init,
#         B_diagonal_init=B_diagonal_init,
#     )
#     _c = torch.zeros((D.n_potentials, D.m_potentials))

#     for n in range(n_potentials):
#         for m in range(m_potentials):
#             _c[n][m] = torch.sum(
#                 D.A_yy_diagonal_matrix[n]
#                 * D.B_yy_diagonal_matrix[m]
#                 * ((D.a_y[n] - D.b_y[m]) ** 2)
#                 / (D.A_yy_diagonal_matrix[n] + D.B_yy_diagonal_matrix[m])
#             )

#     c = D.compute_c()
#     assert torch.allclose(c, _c)


@pytest.mark.parametrize("n_potentials", [100, 200])
@pytest.mark.parametrize("m_potentials", [300, 400])
@pytest.mark.parametrize("A_diagonal_init", [0.1])
@pytest.mark.parametrize("B_diagonal_init", [0.3])
def test_compute_alpha(n_potentials: int, m_potentials: int, A_diagonal_init: float, B_diagonal_init: float):
    D = LightGCOT(
        n_potentials=n_potentials,
        m_potentials=m_potentials,
        A_diagonal_init=A_diagonal_init,
        B_diagonal_init=B_diagonal_init,
    )
    c = D.compute_c()
    G_xx, G_xy, G_yx, G_yy = D.compute_G()
    _log_alpha = torch.zeros((n_potentials, m_potentials))

    for n in range(n_potentials):
        for m in range(m_potentials):
            _log_alpha[n][m] = (
                D.log_w[n]
                + D.log_v[m]
                - 0.5 * (c[n][m] / D.epsilon - torch.log(2 * torch.pi * D.epsilon) * (D.x_dim - D.y_dim))
                - torch.sum(torch.log(G_xx[n][m]))
                + torch.sum(torch.log(G_yy[n][m]))
            )

    log_alpha = D.compute_log_alpha(G_xx, G_xy, G_yx, G_yy)
    assert torch.allclose(log_alpha, _log_alpha)

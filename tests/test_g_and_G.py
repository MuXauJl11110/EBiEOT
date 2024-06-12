import torch

from src.light_gcot import LightGCOT


def test_compute_G_nm(D: LightGCOT, B_m: torch.Tensor, batch_size: int):
    G_nm, G_inv_nm, GGT_nm = D.compute_G_nm(B_m)

    _G_nm = torch.zeros((batch_size, D.n_potentials, D.m_potentials, D.y_dim))
    _G_inv_nm = torch.zeros((batch_size, D.n_potentials, D.m_potentials, D.y_dim))
    _GGT_nm = torch.zeros((batch_size, D.n_potentials, D.m_potentials, D.y_dim))

    for n in range(D.n_potentials):
        for m in range(D.m_potentials):
            _G_nm[:, n, m, :] = D.A_diagonal_matrix[n] + B_m[:, m]
            _G_inv_nm[:, n, m, :] = 1 / (D.A_diagonal_matrix[n] + B_m[:, m])
            _GGT_nm[:, n, m, :] = 1 / (D.A_diagonal_matrix[n] + B_m[:, m] + D.A_diagonal_matrix[n] + B_m[:, m])

    assert torch.allclose(G_nm, _G_nm)
    assert torch.allclose(G_inv_nm, _G_inv_nm)
    assert torch.allclose(GGT_nm, _GGT_nm)


def test_compute_g_nm(D: LightGCOT, b_m: torch.Tensor, B_m: torch.Tensor, batch_size: int):
    b_nm = D.compute_b_nm(b_m, B_m)
    _, _, GGT_inv_nm = D.compute_G_nm(B_m)
    g_nm = D.compute_g_nm(b_nm, GGT_inv_nm)
    _g_nm = torch.zeros((batch_size, D.n_potentials, D.m_potentials, D.y_dim))

    for n in range(D.n_potentials):
        for m in range(D.m_potentials):
            G_nm = D.A_diagonal_matrix[n] + B_m[:, m, :]
            _g_nm[:, n, m, :] = (
                (D.A_diagonal_matrix[n] + D.A_diagonal_matrix[n]) * D.a[n]
                + (B_m[:, m, :] + B_m[:, m, :]) * b_m[:, m, :]
            ) / (G_nm + G_nm)

    assert torch.allclose(g_nm, _g_nm)

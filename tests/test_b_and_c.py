import torch

from src.light_gcot import LightGCOT


def test_compute_b_nm(D: LightGCOT, b_m: torch.Tensor, B_m: torch.Tensor, A_n: torch.Tensor, batch_size: int):
    b_nm = D.compute_b_nm(b_m, B_m, A_n)
    _b_nm = torch.zeros((batch_size, D.n_potentials, D.m_potentials, D.y_dim))

    for n in range(D.n_potentials):
        for m in range(D.m_potentials):
            _b_nm[:, n, m, :] = (A_n[:, n, :] + A_n[:, n, :]) * D.a_n[n] + (B_m[:, m, :] + B_m[:, m, :]) * b_m[:, m, :]

    assert torch.allclose(b_nm, _b_nm)


def test_compute_c_nm(D: LightGCOT, b_m: torch.Tensor, B_m: torch.Tensor, A_n: torch.Tensor, batch_size: int):
    c_nm = D.compute_c_nm(b_m, B_m, A_n)
    _c_nm = torch.zeros((batch_size, D.n_potentials, D.m_potentials))

    for n in range(D.n_potentials):
        for m in range(D.m_potentials):
            _c_nm[:, n, m] = torch.sum(D.a_n[n] * A_n[:, n, :] * D.a_n[n], dim=1) + torch.sum(
                b_m[:, m, :] * B_m[:, m, :] * b_m[:, m, :], dim=1
            )

    assert torch.allclose(c_nm, _c_nm)

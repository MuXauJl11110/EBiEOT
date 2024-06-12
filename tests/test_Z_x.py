import torch

from src.light_gcot import LightGCOT


def test_compute_log_alpha_nm(
    batch_size: int, D: LightGCOT, b_m: torch.Tensor, B_m: torch.Tensor, log_v_m: torch.Tensor
):
    G_nm, _, _ = D.compute_G_nm(B_m)  # [bs x N x M x y_dim]
    c_nm = D.compute_c_nm(b_m, B_m)  # [bs x N x M]
    log_alpha_nm = D.compute_log_alpha_nm(log_v_m, B_m, G_nm, c_nm)  # [bs x N x M]

    _log_alpha_nm = torch.zeros((batch_size, D.n_potentials, D.m_potentials))
    for n in range(D.n_potentials):
        for m in range(D.m_potentials):
            _log_alpha_nm[:, n, m] = (
                D.log_w[n]
                + log_v_m[:, m]
                + 0.5
                * (
                    torch.log(B_m[:, m, :]).sum(dim=1)
                    - torch.log(G_nm[:, n, m, :]).sum(dim=1)
                    - c_nm[:, n, m] / D.epsilon
                )
            )

    assert torch.allclose(log_alpha_nm, _log_alpha_nm)


def test_compute_Z(batch_size: int, D: LightGCOT, b_m: torch.Tensor, B_m: torch.Tensor, log_v_m: torch.Tensor):
    b_nm = D.compute_b_nm(b_m, B_m)  # [bs x N x M x y_dim]
    G_nm, G_inv_nm, _ = D.compute_G_nm(B_m)  # [bs x N x M x y_dim]
    c_nm = D.compute_c_nm(b_m, B_m)  # [bs x N x M]
    log_alpha_nm = D.compute_log_alpha_nm(log_v_m, B_m, G_nm, c_nm)  # [bs x N x M]
    Z = D.compute_Z(log_alpha_nm, G_inv_nm, b_nm)  # [bs]

    _log_Z_nm = torch.zeros((batch_size, D.n_potentials, D.m_potentials))
    for n in range(D.n_potentials):
        for m in range(D.m_potentials):
            _log_Z_nm[:, n, m] = (
                D.log_w[n]
                + log_v_m[:, m]
                + 0.5
                * (
                    torch.log(B_m[:, m, :]).sum(dim=1)
                    - torch.log(G_nm[:, n, m, :]).sum(dim=1)
                    + (
                        0.25 * torch.sum(b_nm[:, n, m, :] * G_inv_nm[:, n, m, :] * b_nm[:, n, m, :], dim=1)
                        - c_nm[:, n, m]
                    )
                    / D.epsilon
                )
            )

    _Z = torch.exp(_log_Z_nm).sum(dim=(1, 2))
    assert torch.allclose(Z, _Z)


def test_compute_beta(batch_size: int, D: LightGCOT, b_m: torch.Tensor, B_m: torch.Tensor, log_v_m: torch.Tensor):
    b_nm = D.compute_b_nm(b_m, B_m)  # [bs x N x M x y_dim]
    G_nm, G_inv_nm, GGT_inv_nm = D.compute_G_nm(B_m)  # [bs x N x M x y_dim]
    g_nm = D.compute_g_nm(b_nm, GGT_inv_nm)  # [bs x N x M x y_dim]
    c_nm = D.compute_c_nm(b_m, B_m)  # [bs x N x M]
    log_alpha_nm = D.compute_log_alpha_nm(log_v_m, B_m, G_nm, c_nm)  # [bs x N x M]
    Z = D.compute_Z(log_alpha_nm, G_inv_nm, b_nm)  # [bs]
    log_beta_nm = D.compute_log_beta_nm(g_nm, G_nm, log_alpha_nm, Z)  # [bs x N x M]

    _log_beta_nm = torch.zeros((batch_size, D.n_potentials, D.m_potentials))
    for n in range(D.n_potentials):
        for m in range(D.m_potentials):
            _log_beta_nm[:, n, m] = (
                D.log_w[n]
                + log_v_m[:, m]
                + 0.5
                * (
                    torch.log(B_m[:, m, :]).sum(dim=1)
                    - torch.log(G_nm[:, n, m, :]).sum(dim=1)
                    + (torch.sum(g_nm[:, n, m, :] * G_nm[:, n, m, :] * g_nm[:, n, m, :], dim=1) - c_nm[:, n, m])
                    / D.epsilon
                )
                - torch.log(Z)
            )

    assert torch.allclose(log_beta_nm, _log_beta_nm)

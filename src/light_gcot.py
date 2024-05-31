import torch
from torch import nn


class LightGCOT(nn.Module):
    def __init__(
        self,
        x_dim: int = 2,
        y_dim: int = 3,
        n_potentials: int = 5,
        m_potentials: int = 10,
        epsilon: float = 1.0,
        sampling_batch_size: int = 1,
        A_diagonal_init: float | None = None,
        B_diagonal_init: float | None = None,
    ):
        r"""
        :param int x_dim: Dimension of X space, defaults to 2
        :param int y_dim: Dimension of Y space, defaults to 3
        :param int n_potentials: Number of potentials for approximating dual variable :math:`f(y)=\varepsilon\log\sum_{n=1}^N w_n Q(y\vert a_n, A_n/\varepsilon)`, defaults to 5
        :param int m_potentials: Number of potentials for approximating plan :math:`\pi(x, y)=\sum_{m=1}^M v_m Q((x, y)^\Top\vert \tilde{b}_m, B_m/\varepsilon)`, defaults to 10
        :param float epsilon: Regularization parameter, defaults to 1.0
        :param int sampling_batch_size: Sampling batch size, defaults to 1
        :param float A_diagonal_init: Init of diagonal matrices for dual variable potential, defaults to 0.1
        :param float B_diagonal_init: Init of diagonal matrices for plan potential, defaults to 0.1
        """
        super().__init__()
        self.A_diagonal_init = A_diagonal_init
        self.B_diagonal_init = B_diagonal_init
        assert A_diagonal_init is not None  # TODO: add non-diagonal
        assert B_diagonal_init is not None  # TODO: add non-diagonal
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.n_potentials = n_potentials
        self.m_potentials = m_potentials
        self.register_buffer("epsilon", torch.tensor(epsilon))
        self.sampling_batch_size = sampling_batch_size

        self.log_w = nn.Parameter(torch.log(torch.ones(n_potentials) / n_potentials))
        self.a_x = torch.zeros(n_potentials, x_dim)
        self.a_y = nn.Parameter(torch.randn(n_potentials, y_dim))
        if A_diagonal_init is not None:
            self.A_xx_diagonal_matrix = torch.zeros((n_potentials, x_dim))
            self.A_yy_diagonal_matrix = nn.Parameter(A_diagonal_init * torch.ones(n_potentials, y_dim))

        self.log_v = nn.Parameter(torch.log(torch.ones(m_potentials) / m_potentials))
        self.b_x = nn.Parameter(torch.randn(m_potentials, x_dim))
        self.b_y = nn.Parameter(torch.randn(m_potentials, y_dim))
        if B_diagonal_init is not None:
            self.B_xx_diagonal_matrix = nn.Parameter(B_diagonal_init * torch.ones(m_potentials, x_dim))
            self.B_yy_diagonal_matrix = nn.Parameter(B_diagonal_init * torch.ones(m_potentials, y_dim))

    def compute_G(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.A_diagonal_init is not None and self.B_diagonal_init is not None:
            G_xx = self.B_xx_diagonal_matrix.repeat(self.n_potentials, 1, 1)  # [N x M x x_dim]
            G_yy = (
                self.A_yy_diagonal_matrix[:, None, :] + self.B_yy_diagonal_matrix[None, :, :]
            )  # [N x 1 x y_dim] + [1 x M x y_dim] = [N x M x y_dim]
            return G_xx, None, None, G_yy
        else:
            raise NotImplementedError("Other options are not implemented yet!")

    def compute_g(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.A_diagonal_init is not None and self.B_diagonal_init is not None:
            g_x = self.b_x.repeat(self.n_potentials, 1, 1)  # [N x M x x_dim]
            g_y = (
                self.A_yy_diagonal_matrix[:, None, :] * self.a_y[:, None, :]
                + self.B_yy_diagonal_matrix[None, :, :] * self.b_y[None, :, :]
            ) / (self.A_yy_diagonal_matrix[:, None, :] + self.B_yy_diagonal_matrix[None, :, :])
            # ([N x 1 x y_dim] * [N x 1 x y_dim] + [1 x M x y_dim] * [1 x M x y_dim]) / ([N x 1 x y_dim] + [1 x M x y_dim]) = [N x M x y_dim]
            return g_x, g_y
        else:
            raise NotImplementedError("Other options are not implemented yet!")

    def compute_c(self) -> torch.Tensor:
        if self.A_diagonal_init is not None and self.B_diagonal_init is not None:
            return torch.sum(
                self.A_yy_diagonal_matrix[:, None, :]
                * self.B_yy_diagonal_matrix[None, :, :]
                * (self.a_y[:, None, :] - self.b_y[None, :, :])
                * (self.a_y[:, None, :] - self.b_y[None, :, :])
                / (self.A_yy_diagonal_matrix[:, None, :] + self.B_yy_diagonal_matrix[None, :, :]),
                dim=2,
            )
            # sum([N x 1 x y_dim] * [1 x M x y_dim] * ([N x 1 x y_dim] - [1 x M x y_dim)^2 / ([N x 1 x y_dim] - [1 x M x y_dim), dim=2) = [N x M]
        else:
            raise NotImplementedError("Other options are not implemented yet!")

    def compute_log_alpha(
        self, G_xx: torch.Tensor, G_xy: torch.Tensor, G_yx: torch.Tensor, G_yy: torch.Tensor
    ) -> torch.Tensor:
        if self.A_diagonal_init is not None and self.B_diagonal_init is not None:
            return (
                self.log_w[:, None]
                + self.log_v[None, :]
                - 0.5
                * (
                    self.compute_c() / self.epsilon
                    # + torch.log(2 * torch.pi * self.epsilon) * (self.y_dim - self.x_dim)
                    - torch.log(2 * torch.pi * self.epsilon) * self.y_dim
                    + torch.log(G_yy).sum(dim=2)
                )
                # - torch.log(G_xx).sum(dim=2)
            )
            # [N x 1] + [1 x M] - [N x M] - sum([N x M x x_dim], dim=2) + sum([N x M x y_dim], dim=2) = [N x M]
        else:
            raise NotImplementedError("Other options are not implemented yet!")

    def compute_log_quadratic(
        self,
        batched_x: torch.Tensor,
        g_x: torch.Tensor,
        G_xx: torch.Tensor,
        G_xy: torch.Tensor,
        G_yx: torch.Tensor,
        G_yy: torch.Tensor,
    ) -> torch.Tensor:
        if self.A_diagonal_init is not None and self.B_diagonal_init is not None:
            diff = (
                batched_x[None, None, :, :] - g_x[:, :, None, :]
            )  # [1 x 1 x bs x x_dim] + [N x M x 1 x x_dim] = [N x M x bs x x_dim]
            return -0.5 / self.epsilon * torch.sum(diff * G_xx[:, :, None, :] * diff, dim=(2, 3))  # [N x M]

    def compute_primal_potential(self, batched_y: torch.Tensor) -> float:
        diff = batched_y[None, :, :] - self.a_y[:, None, :]  # [1 x bs x y_dim] - [N x 1 x y_dim] = [N x bs x y_dim]
        log_quadratic = (
            -0.5 / self.epsilon * torch.sum(diff * self.A_yy_diagonal_matrix[:, None, :] * diff, dim=(1, 2))
        )
        return self.epsilon * torch.log(torch.sum(torch.exp(self.log_w + log_quadratic)))

    def compute_dual_potential(self, batched_x: torch.Tensor) -> float:
        G_xx, G_xy, G_yx, G_yy = self.compute_G()
        g_x, _ = self.compute_g()
        log_alpha = self.compute_log_alpha(G_xx, G_xy, G_yx, G_yy)
        log_quadratic = self.compute_log_quadratic(batched_x, g_x, G_xx, G_xy, G_yx, G_yy)

        return -self.epsilon * torch.log(torch.sum(torch.exp(log_alpha + log_quadratic)))

    def set_epsilon(self, new_epsilon):
        self.epsilon = torch.tensor(new_epsilon, device=self.epsilon.device)

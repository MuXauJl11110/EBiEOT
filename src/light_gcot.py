import torch
from torch import nn
from torch.distributions.categorical import Categorical
from torch.distributions.independent import Independent
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.normal import Normal


class LightGCOT(nn.Module):
    def __init__(
        self,
        x_dim: int = 2,
        y_dim: int = 2,
        n_potentials: int = 5,
        m_potentials: int = 10,
        epsilon: float = 1.0,
        sampling_batch_size: int = 1,
        A_diagonal_init: float | None = None,
        is_B_diagonal: bool = True,
    ):
        r"""
        :param int x_dim: Dimension of X space, defaults to 2
        :param int y_dim: Dimension of Y space, defaults to 3
        :param int n_potentials: Number of potentials for approximating dual variable :math:`f(y)=\varepsilon\log\sum_{n=1}^N w_n Q(y\vert a_n, A_n/\varepsilon)`, defaults to 5
        :param int m_potentials: Number of potentials for approximating plan :math:`c(x, y)=-\varepsilon\log\sum_{m=1}^M v_m(x) \mathcal{N}((x, y)\vert \tilde{b}_m, \varepsilon B_m^{-1})`, defaults to 10
        :param float epsilon: Regularization parameter, defaults to 1.0
        :param int sampling_batch_size: Sampling batch size, defaults to 1
        :param float A_diagonal_init: Init of diagonal matrices for dual variable potential, defaults to 0.1
        :param float B_diagonal_init: Init of diagonal matrices for plan potential, defaults to 0.1
        """
        super().__init__()
        self.A_diagonal_init = A_diagonal_init
        self.is_B_diagonal = is_B_diagonal
        assert A_diagonal_init is not None  # TODO: add non-diagonal
        assert is_B_diagonal is True  # TODO: add non-diagonal
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.n_potentials = n_potentials
        self.m_potentials = m_potentials
        self.register_buffer("epsilon", torch.tensor(epsilon))
        self.sampling_batch_size = sampling_batch_size

        self.log_w = nn.Parameter(torch.log(torch.ones(n_potentials) / n_potentials))
        self.a = nn.Parameter(torch.randn(n_potentials, y_dim))
        if A_diagonal_init is not None:
            self.A_diagonal_matrix = nn.Parameter(A_diagonal_init * torch.ones(n_potentials, y_dim))

    def compute_log_v_m(self, batched_x: torch.Tensor) -> torch.Tensor:
        batch_size = batched_x.shape[0]
        return torch.log(torch.ones(self.m_potentials) / self.m_potentials).repeat(batch_size, 1)  # [bs x M]

    def compute_b_m(self, batched_x: torch.Tensor) -> torch.Tensor:
        # TODO: make general case
        # return torch.stack((batched_x, -batched_x), dim=1)  # [bs x M x y_dim]
        return batched_x  # [bs x M x y_dim]

    def compute_B_m(self, batched_x: torch.Tensor) -> torch.Tensor:
        batch_size = batched_x.shape[0]
        # epsilonI = self.epsilon * torch.ones(self.m_potentials // 2, self.y_dim)
        # return torch.cat((epsilonI, epsilonI)).repeat(batch_size, 1, 1)  # [bs x M x y_dim]
        return (torch.ones(self.y_dim)).repeat(batch_size, 1, 1)  # [bs x M x y_dim]

    def compute_b_nm(self, b_m: torch.Tensor, B_m: torch.Tensor) -> tuple[torch.Tensor]:
        if self.A_diagonal_init is not None and self.is_B_diagonal:
            BT_b = B_m * b_m  # [bs x M x y_dim] * [bs x M x y_dim] = [bs x M x y_dim]
            return 2 * (
                (self.A_diagonal_matrix * self.a)[None, :, None, :] + BT_b[:, None, :, :]
            )  # [1 x N x 1 x y_dim] + [bs x 1 x M x y_dim] = [bs x N x M x y_dim]
        else:
            raise NotImplementedError("Other options are not implemented yet!")

    def compute_c_nm(self, b_m: torch.Tensor, B_m: torch.Tensor) -> tuple[torch.Tensor]:
        if self.A_diagonal_init is not None and self.is_B_diagonal:
            bT_B_b = b_m * B_m * b_m  # [bs x M x y_dim] * [bs x M x y_dim] * [bs x M x y_dim]
            return torch.sum(
                (self.a * self.A_diagonal_matrix * self.a)[None, :, None, :] + bT_B_b[:, None, :, :], dim=3
            )
            # sum([1 x N x 1 x y_dim] + [bs x 1 x M x y_dim], dim=3) = sum([bs x N x M x y_dim], dim=3) = [bs x N x M]
        else:
            raise NotImplementedError("Other options are not implemented yet!")

    def compute_G_nm(self, B_m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes G_{nm}, G_{nm}^{-1}, (G_{nm} + G_{nm}^\top)^{-1}. TODO: think about memory
        """
        if self.A_diagonal_init is not None and self.is_B_diagonal:
            G = self.A_diagonal_matrix[None, :, None, :] + B_m[:, None, :, :]
            # [1 x N x 1 x y_dim] + [bs x 1 x M x y_dim] = [bs x N x M x y_dim]
            G_inv = 1 / G
            return G, G_inv, 0.5 * G_inv
        else:
            raise NotImplementedError("Other options are not implemented yet!")

    def compute_g_nm(self, b_nm: torch.Tensor, GGT_inv_nm: torch.Tensor) -> tuple[torch.Tensor]:
        if self.A_diagonal_init is not None and self.is_B_diagonal:
            return GGT_inv_nm * b_nm  # [bs x N x M x y_dim] * [bs x N x M x y_dim] = [bs x N x M x y_dim]
        else:
            raise NotImplementedError("Other options are not implemented yet!")

    def compute_log_alpha_nm(
        self, log_v_m: torch.Tensor, B_m: torch.Tensor, G_nm: torch.Tensor, c_nm: torch.Tensor
    ) -> tuple[torch.Tensor]:
        if self.A_diagonal_init is not None and self.is_B_diagonal:
            return (
                self.log_w[None, :, None]
                + log_v_m[:, None, :]
                + 0.5 * (torch.log(B_m).sum(dim=2)[:, None, :] - torch.log(G_nm).sum(dim=3) - c_nm / self.epsilon)
            )  # [1 x N x 1] + [bs x 1 x M] + [bs x 1 x M] + [bs x N x M] + [bs x N x M] = [bs x N x M]
        else:
            raise NotImplementedError("Other options are not implemented yet!")

    def compute_Z(self, log_alpha_nm: torch.Tensor, G_inv_nm: torch.Tensor, b_nm: torch.Tensor) -> float:
        if self.A_diagonal_init is not None and self.is_B_diagonal:
            return torch.sum(
                torch.exp(log_alpha_nm + 0.125 * torch.sum(b_nm * G_inv_nm * b_nm, dim=3)), dim=(1, 2)
            )  # sum([bs x N x M] + [bs x N x M], dim=(1, 2)) = [bs]
        else:
            raise NotImplementedError("Other options are not implemented yet!")

    def compute_log_beta_nm(
        self, g_nm: torch.Tensor, G_nm: torch.Tensor, log_alpha_nm: torch.Tensor, Z: torch.Tensor
    ) -> torch.Tensor:
        if self.A_diagonal_init is not None and self.is_B_diagonal:
            return (
                log_alpha_nm + 0.5 / self.epsilon * torch.sum(g_nm * G_nm * g_nm, dim=3) - torch.log(Z)[:, None, None]
            )
            # [bs x N x M] + [bs x N x M] x [bs x 1 x 1] = [bs x N x M]
        else:
            raise NotImplementedError("Other options are not implemented yet!")

    def compute_primal_potential(self, batched_y: torch.Tensor) -> float:
        if self.A_diagonal_init is not None and self.is_B_diagonal:
            diff = batched_y[:, None, :] - self.a[None, :, :]  # [bs x 1 x y_dim] - [1 x N x y_dim] = [bs x N x y_dim]
            log_quadratic = (
                -0.5 / self.epsilon * torch.sum(diff * self.A_diagonal_matrix[None, :, :] * diff, dim=2)
            )  # [bs x N]
            return self.epsilon * torch.log(torch.sum(torch.exp(self.log_w[None, :] + log_quadratic), dim=1))  # [bs]

    def compute_dual_potential(self, batched_x: torch.Tensor) -> float:
        B_m = self.compute_B_m(batched_x)  # [bs x M x y_dim]
        b_m = self.compute_b_m(batched_x)  # [bs x M x y_dim]
        b_nm = self.compute_b_nm(b_m, B_m)  # [bs x N x M x y_dim]
        c_nm = self.compute_c_nm(b_m, B_m)  # [bs x N x M]
        G_nm, G_inv_nm, _ = self.compute_G_nm(B_m)  # [bs x N x M x y_dim], [bs x N x M x y_dim], _
        assert torch.min(G_nm) > 0
        log_v_m = self.compute_log_v_m(batched_x)  # [bs x M]
        log_alpha_nm = self.compute_log_alpha_nm(log_v_m, B_m, G_nm, c_nm)  # [bs x N x M]
        Z = self.compute_Z(log_alpha_nm, G_inv_nm, b_nm)
        return -self.epsilon * torch.log(Z)  # [bs]

    def set_epsilon(self, new_epsilon):
        self.epsilon = torch.tensor(new_epsilon, device=self.epsilon.device)

    @torch.no_grad()
    def forward(self, batched_x: torch.Tensor):
        samples = []
        batch_size = batched_x.shape[0]
        sampling_batch_size = self.sampling_batch_size

        num_sampling_iterations = (
            batch_size // sampling_batch_size
            if batch_size % sampling_batch_size == 0
            else (batch_size // sampling_batch_size) + 1
        )

        for i in range(num_sampling_iterations):
            sub_batch_x = batched_x[sampling_batch_size * i : sampling_batch_size * (i + 1)]

            B_m = self.compute_B_m(sub_batch_x)  # [bs x M x y_dim]
            b_m = self.compute_b_m(sub_batch_x)  # [bs x M x y_dim]
            log_v_m = self.compute_log_v_m(sub_batch_x)  # [bs x M]

            b_nm = self.compute_b_nm(b_m, B_m)  # [bs x N x M x y_dim]
            c_nm = self.compute_c_nm(b_m, B_m)  # [bs x N x M]

            G_nm, G_inv_nm, GGT_inv_nm = self.compute_G_nm(B_m)
            # [bs x N x M x y_dim], [bs x N x M x y_dim], [bs x N x M x y_dim]
            g_nm = self.compute_g_nm(b_nm, GGT_inv_nm)  # [bs x N x M x y_dim]

            log_alpha_nm = self.compute_log_alpha_nm(log_v_m, B_m, G_nm, c_nm)  # [bs x N x M]
            Z = self.compute_Z(log_alpha_nm, G_inv_nm, b_nm)  # [bs]
            log_beta_nm = self.compute_log_beta_nm(g_nm, G_nm, log_alpha_nm, Z)

            logits = torch.exp(log_beta_nm)

            if self.A_diagonal_init is not None and self.is_B_diagonal:
                mix = Categorical(logits=logits)
                comp = Independent(Normal(loc=g_nm, scale=torch.sqrt(self.epsilon * G_inv_nm)), 1)
                gmm = MixtureSameFamily(mix, comp)

            else:
                raise NotImplementedError("Other options are not implemented yet!")

            samples.append(gmm.sample())

        samples = torch.cat(samples, dim=0)

        return samples

import torch
import torchvision
from torch import nn
from torch.distributions.categorical import Categorical
from torch.distributions.independent import Independent
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.normal import Normal

from src.distributions import StandardNormalOnCircleSampler


class LightGCOT(nn.Module):
    def __init__(
        self,
        x_dim: int = 2,
        y_dim: int = 2,
        n_potentials: int = 5,
        m_potentials: int = 10,
        epsilon: float = 1.0,
        sampling_batch_size: int = 128,
        A_diagonal_init: float | None = None,
        B_diagonal_init: float | None = None,
        cost_function: str = "l2",
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
        :param str cost_function: Cost function parametrization.
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

        self.log_w_n = nn.Parameter(torch.log(torch.ones(n_potentials) / n_potentials))
        self.a_n = nn.Parameter(torch.randn(n_potentials, y_dim))
        if A_diagonal_init is not None:
            self.A_n = nn.Parameter(A_diagonal_init * torch.ones(n_potentials, y_dim))  # [N x y_dim]

        self.known_costs = {
            "l2",
            "Alexander's",
            "uniform_on_circle",
            "uniform_on_circle_plus_x",
            "double_uniform_on_circle_plus_x",
            "B_m_parametrization",
            "parameters",
            "MLP",
        }
        self.cost_function = cost_function
        if self.cost_function not in self.known_costs:
            raise NotImplementedError(f"Cost function: {self.cost_function} not implemented yet!")

        self.log_v_m = torch.log(torch.ones(self.m_potentials) / self.m_potentials)
        self.B_m = torch.ones(self.m_potentials, self.y_dim)
        if self.cost_function in {"uniform_on_circle", "uniform_on_circle_plus_x"}:
            t = torch.arange(0, self.m_potentials) / m_potentials
            R, D = 2, torch.tensor([[0.1, 0], [0, 10]])
            self.b_m, self.B_m = StandardNormalOnCircleSampler(R, D).compute(t, diag=True)
        elif self.cost_function == "double_uniform_on_circle_plus_x":
            num_t = self.m_potentials // 2
            t = torch.arange(0, num_t) / num_t
            D = torch.tensor([[0.1, 0], [0, 10]])
            b_m_1, B_m_1 = StandardNormalOnCircleSampler(1, D).compute(t, diag=True)
            b_m_2, B_m_2 = StandardNormalOnCircleSampler(2, D).compute(t, diag=True)
            self.b_m = torch.cat((b_m_1, b_m_2), dim=0)
            self.B_m = torch.cat((B_m_1, B_m_2), dim=0)
        elif self.cost_function == "B_m_parametrization":
            self.log_v_m = torch.log(torch.ones(m_potentials) / m_potentials)
            self.b_m = torch.randn(m_potentials, y_dim)
            self.B_m = nn.Parameter(torch.exp(B_diagonal_init * torch.ones(m_potentials, y_dim)))
            self.A_n = nn.Parameter(torch.exp(A_diagonal_init * torch.ones(n_potentials, y_dim)))
        elif self.cost_function == "parameters":
            self.log_v_m = nn.Parameter(torch.log(torch.ones(m_potentials) / m_potentials))
            self.b_m = nn.Parameter(torch.randn(m_potentials, y_dim))
            self.B_m = nn.Parameter(B_diagonal_init * torch.ones(m_potentials, y_dim))
        elif self.cost_function == "MLP":
            self.log_v_m = nn.Sequential(
                torchvision.ops.MLP(in_channels=x_dim, hidden_channels=[m_potentials], activation_layer=torch.nn.ReLU),
                nn.LogSoftmax(dim=-1),
            )
            self.b_m = torchvision.ops.MLP(
                in_channels=x_dim, hidden_channels=[m_potentials * y_dim], activation_layer=torch.nn.ReLU
            )
            # init = Exponential(torch.ones(m_potentials, y_dim))  # torch.rand(m_potentials, y_dim))
            # self.B_m = nn.Parameter(torch.exp(B_diagonal_init * init.sample()))
            self.B_m = nn.Parameter(B_diagonal_init * torch.ones(m_potentials, y_dim))
            self.A_n = nn.Parameter(A_diagonal_init * torch.rand(n_potentials, y_dim) + A_diagonal_init)

    def init_a_by_samples(self, samples):
        assert samples.shape[0] == self.a_n.shape[0]

        self.a_n.data = torch.clone(samples.to(self.a_n.device))

    def compute_cost(
        self,
        b_m: torch.Tensor,
        B_m: torch.Tensor,
        log_v_m: torch.Tensor,
        sample: bool = False,
        batched_y: torch.Tensor | None = None,
    ) -> torch.Tensor:  # -> [bs]
        if batched_y is None and not sample:
            raise ValueError("You must specify 'batched_y' argument!")
        if self.A_diagonal_init is not None and self.B_diagonal_init is not None:
            mix = Categorical(logits=log_v_m)
            comp = Independent(Normal(loc=b_m, scale=torch.sqrt(self.epsilon * B_m)), 1)
            gmm = MixtureSameFamily(mix, comp)
        else:
            raise NotImplementedError("Other options are not implemented yet!")
        if sample:
            return gmm.sample()
        else:
            return -self.epsilon * gmm.log_prob(batched_y)

    def compute_A_n(self, B_m: torch.Tensor) -> torch.Tensor:  # -> [1 x N x y_dim]
        batch_size = B_m.shape[0]
        self.A_n_matrix = self.A_n.repeat(batch_size, 1, 1)
        return self.A_n_matrix

    def compute_log_v_m(self, batched_x: torch.Tensor) -> torch.Tensor:  # -> [bs x M]
        batch_size = batched_x.shape[0]
        if self.cost_function in {
            "l2",
            "Alexander's",
            "uniform_on_circle",
            "uniform_on_circle_plus_x",
            "double_uniform_on_circle_plus_x",
            "B_m_parametrization",
            "parameters",
        }:
            return self.log_v_m.repeat(batch_size, 1)
        elif self.cost_function == "MLP":
            return self.log_v_m(batched_x)
        else:
            raise NotImplementedError("Other options are not implemented yet!")

    def compute_b_m(self, batched_x: torch.Tensor) -> torch.Tensor:  # -> [bs x M x y_dim]
        batch_size = batched_x.shape[0]
        if self.cost_function == "l2":
            assert self.m_potentials == 1
            return batched_x.view(batch_size, 1, self.y_dim)
        elif self.cost_function == "Alexander's":
            assert self.m_potentials == 2
            return torch.stack((batched_x, -batched_x), dim=1)
        elif self.cost_function in {"uniform_on_circle", "parameters"}:
            return self.b_m.repeat(batch_size, 1, 1)
        elif self.cost_function in {"uniform_on_circle_plus_x", "double_uniform_on_circle_plus_x"}:
            return self.b_m.repeat(batch_size, 1, 1) + batched_x[:, None, :].repeat(1, self.m_potentials, 1)
        elif self.cost_function == "B_m_parametrization":
            return batched_x[:, None, :].repeat(1, self.m_potentials, 1)
        elif self.cost_function == "MLP":
            return self.b_m(batched_x).reshape(batch_size, self.m_potentials, self.y_dim)
        else:
            raise NotImplementedError("Other options are not implemented yet!")

    def compute_B_m(self, batched_x: torch.Tensor) -> torch.Tensor:  # -> [bs x M x y_dim]
        batch_size = batched_x.shape[0]
        if self.cost_function in {
            "l2",
            "Alexander's",
            "uniform_on_circle",
            "uniform_on_circle_plus_x",
            "double_uniform_on_circle_plus_x",
            "B_m_parametrization",
            "parameters",
            "MLP",
        }:
            self.B_m_matrix = self.B_m.repeat(batch_size, 1, 1)
        else:
            raise NotImplementedError("Other options are not implemented yet!")
        return self.B_m_matrix

    def compute_b_nm(
        self, b_m: torch.Tensor, B_m: torch.Tensor, A_n: torch.Tensor
    ) -> torch.Tensor:  # -> [bs x N x M x y_dim]
        if self.A_diagonal_init is not None and self.B_diagonal_init is not None:
            AT_a = A_n * self.a_n[None, :, :]  # [bs x N x y_dim] * [1 x N x y_dim] = [bs x N x y_dim]
            BT_b = B_m * b_m  # [bs x M x y_dim] * [bs x M x y_dim] = [bs x M x y_dim]
            return 2 * (
                AT_a[:, :, None, :] + BT_b[:, None, :, :]
            )  # [bs x N x 1 x y_dim] + [bs x 1 x M x y_dim] = [bs x N x M x y_dim]
        else:
            raise NotImplementedError("Other options are not implemented yet!")

    def compute_c_nm(self, b_m: torch.Tensor, B_m: torch.Tensor, A_n: torch.Tensor) -> torch.Tensor:  # -> [bs x N x M]
        if self.A_diagonal_init is not None and self.B_diagonal_init is not None:
            aT_A_a = torch.sum(
                self.a_n[None, :, :] * A_n * self.a_n[None, :, :], dim=2
            )  # sum([1 x N x y_dim] * [bs x N x y_dim] * [1 x N x y_dim], dim=2) = [bs x N]
            bT_B_b = torch.sum(
                b_m * B_m * b_m, dim=2
            )  # sum([bs x M x y_dim] * [bs x M x y_dim] * [bs x M x y_dim], dim=2) = [bs x M]
            return aT_A_a[:, :, None] + bT_B_b[:, None, :]
            # [bs x N x 1] + [bs x 1 x M] = [bs x N x M]
        else:
            raise NotImplementedError("Other options are not implemented yet!")

    def compute_G_nm(
        self, B_m: torch.Tensor, A_n: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:  # -> ([bs x N x M x y_dim], [bs x N x M x y_dim])
        """
        Computes G_{nm}, G_{nm}^{-1}. TODO: think about memory
        """
        if self.A_diagonal_init is not None and self.B_diagonal_init is not None:
            G = A_n[:, :, None, :] + B_m[:, None, :, :]
            # [bs x N x 1 x y_dim] + [bs x 1 x M x y_dim] = [bs x N x M x y_dim]
            G_inv = 1 / G
            return G, G_inv
        else:
            raise NotImplementedError("Other options are not implemented yet!")

    def compute_g_nm(self, b_nm: torch.Tensor, G_inv_nm: torch.Tensor) -> torch.Tensor:  # -> [bs x N x M x y_dim]
        if self.A_diagonal_init is not None and self.B_diagonal_init is not None:
            return 0.5 * G_inv_nm * b_nm  # [bs x N x M x y_dim] * [bs x N x M x y_dim] = [bs x N x M x y_dim]
        else:
            raise NotImplementedError("Other options are not implemented yet!")

    def compute_log_alpha_nm(
        self, log_v_m: torch.Tensor, B_m: torch.Tensor, G_nm: torch.Tensor, c_nm: torch.Tensor
    ) -> torch.Tensor:  # -> [bs x N x M]
        if self.A_diagonal_init is not None and self.B_diagonal_init is not None:
            return (
                self.log_w_n[None, :, None]
                + log_v_m[:, None, :]
                + 0.5 * (torch.log(B_m).sum(dim=2)[:, None, :] - torch.log(G_nm).sum(dim=3) - c_nm / self.epsilon)
            )  # [1 x N x 1] + [bs x 1 x M] + [bs x 1 x M] + [bs x N x M] + [bs x N x M] = [bs x N x M]
        else:
            raise NotImplementedError("Other options are not implemented yet!")

    # def compute_log_Z_nm(
    #     self, log_alpha_nm: torch.Tensor, G_inv_nm: torch.Tensor, b_nm: torch.Tensor
    # ) -> torch.Tensor:  # -> [bs x N x M]
    #     if self.A_diagonal_init is not None and self.B_diagonal_init is not None:
    #         return (
    #             log_alpha_nm + 0.125 * torch.sum(b_nm * G_inv_nm * b_nm, dim=3) / self.epsilon
    #         )  # [bs x N x M] + [bs x N x M] = [bs x N x M]
    #     else:
    #         raise NotImplementedError("Other options are not implemented yet!")

    def compute_log_Z_nm(
        self, log_alpha_nm: torch.Tensor, G_nm: torch.Tensor, g_nm: torch.Tensor
    ) -> torch.Tensor:  # -> [bs x N x M]
        if self.A_diagonal_init is not None and self.B_diagonal_init is not None:
            return (
                log_alpha_nm + 0.5 * torch.sum(g_nm * G_nm * g_nm, dim=3) / self.epsilon
            )  # [bs x N x M] + [bs x N x M] = [bs x N x M]
        else:
            raise NotImplementedError("Other options are not implemented yet!")

    def compute_primal_potential(self, batched_y: torch.Tensor, A_n: torch.Tensor) -> torch.Tensor:  # -> [bs]
        if self.A_diagonal_init is not None and self.B_diagonal_init is not None:
            diff = (
                batched_y[:, None, :] - self.a_n[None, :, :]
            )  # [bs x 1 x y_dim] - [1 x N x y_dim] = [bs x N x y_dim]
            log_quadratic = (
                -0.5 / self.epsilon * torch.sum(diff * A_n * diff, dim=2)
            )  # sum([bs x N x y_dim] * [bs x N x y_dim] * [bs x N x y_dim], dim=2) = [bs x N]
            return self.epsilon * torch.logsumexp(self.log_w_n[None, :] + log_quadratic, dim=1)  # [bs]

    def compute_dual_potential(
        self, b_m: torch.Tensor, B_m: torch.Tensor, log_v_m: torch.Tensor, A_n: torch.Tensor
    ) -> torch.Tensor:  # -> [bs]
        G_nm, G_inv_nm = self.compute_G_nm(B_m, A_n)
        c_nm = self.compute_c_nm(b_m, B_m, A_n)
        b_nm = self.compute_b_nm(b_m, B_m, A_n)
        log_alpha_nm = self.compute_log_alpha_nm(log_v_m, B_m, G_nm, c_nm)
        # log_Z_nm = self.compute_log_Z_nm(log_alpha_nm, G_inv_nm, b_nm)
        g_nm = self.compute_g_nm(b_nm, G_inv_nm)
        log_Z_nm = self.compute_log_Z_nm(log_alpha_nm, G_nm, g_nm)
        return -self.epsilon * torch.logsumexp(log_Z_nm, dim=(1, 2))  # [bs]

    def set_epsilon(self, new_epsilon):
        self.epsilon = torch.tensor(new_epsilon, device=self.epsilon.device)

    @torch.no_grad()
    def forward(self, batched_x: torch.Tensor) -> torch.Tensor:  # -> [bs]
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

            A_n = self.compute_A_n(B_m)  # [bs x N x y_dim]

            c_nm = self.compute_c_nm(b_m, B_m, A_n)  # [bs x N x M]
            b_nm = self.compute_b_nm(b_m, B_m, A_n)  # [bs x N x M x y_dim]

            G_nm, G_inv_nm = self.compute_G_nm(B_m, A_n)  # [bs x N x M x y_dim], [bs x N x M x y_dim]
            g_nm = self.compute_g_nm(b_nm, G_inv_nm)  # [bs x N x M x y_dim]

            log_alpha_nm = self.compute_log_alpha_nm(log_v_m, B_m, G_nm, c_nm)  # [bs x N x M]
            # log_Z_nm = self.compute_log_Z_nm(log_alpha_nm, G_inv_nm, b_nm)  # [bs x N x M]
            log_Z_nm = self.compute_log_Z_nm(log_alpha_nm, G_nm, g_nm)  # [bs x N x M]

            loc = g_nm.view(min(sampling_batch_size, batch_size), self.n_potentials * self.m_potentials, self.y_dim)
            logits = log_Z_nm.view(min(sampling_batch_size, batch_size), self.n_potentials * self.m_potentials)
            scale = torch.sqrt(self.epsilon * G_inv_nm).view(
                min(sampling_batch_size, batch_size), self.n_potentials * self.m_potentials, self.y_dim
            )
            if self.A_diagonal_init is not None and self.B_diagonal_init is not None:
                mix = Categorical(logits=logits)
                comp = Independent(Normal(loc=loc, scale=scale), 1)
                gmm = MixtureSameFamily(mix, comp)

            else:
                raise NotImplementedError("Other options are not implemented yet!")

            samples.append(gmm.sample())

        samples = torch.cat(samples, dim=0)

        return samples

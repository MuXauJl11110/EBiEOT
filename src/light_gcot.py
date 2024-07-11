import torch
import torchvision
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
        sampling_batch_size: int = 128,
        A_diagonal_init: float | None = None,
        cost_function: str = "parameters",
    ):
        r"""
        :param int x_dim: Dimension of X space, defaults to 2
        :param int y_dim: Dimension of Y space, defaults to 3
        :param int n_potentials: Number of potentials for approximating dual variable :math:`f(y)=\varepsilon\log\sum_{n=1}^N w_n \mathcal{N}(y\vert a_n, A_n/\varepsilon)`, defaults to 5
        :param int m_potentials: Number of potentials for approximating plan :math:`c(x, y)=-\varepsilon\log\sum_{m=1}^M v_m(x) \exp(\langle b_m(x), y \rangle) /\varepsilon`, defaults to 10
        :param float epsilon: Regularization parameter, defaults to 1.0
        :param int sampling_batch_size: Sampling batch size, defaults to 1
        :param float A_diagonal_init: Init of diagonal matrices for dual variable potential, defaults to 0.1
        :param str cost_function: Cost function parametrization.
        """
        super().__init__()
        self.A_diagonal_init = A_diagonal_init
        assert A_diagonal_init is not None  # TODO: add non-diagonal
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.n_potentials = n_potentials
        self.m_potentials = m_potentials
        self.register_buffer("epsilon", torch.tensor(epsilon))
        self.sampling_batch_size = sampling_batch_size

        self.log_w_n = nn.Parameter(torch.log(torch.ones(n_potentials) / n_potentials))
        self.a_n = nn.Parameter(torch.randn(n_potentials, y_dim))
        if A_diagonal_init is not None:
            self.log_A_n = nn.Parameter(torch.log(A_diagonal_init * torch.ones(n_potentials, y_dim)))  # [N x y_dim]

        self.known_costs = {
            "parameters",
            "MLP",
        }
        self.cost_function = cost_function
        if self.cost_function not in self.known_costs:
            raise NotImplementedError(f"Cost function: {self.cost_function} not implemented yet!")

        self.log_v_m = torch.log(torch.ones(self.m_potentials) / self.m_potentials)
        if self.cost_function == "parameters":
            self.log_v_m = nn.Parameter(torch.log(torch.ones(m_potentials) / m_potentials))
            self.b_m = nn.Parameter(torch.randn(m_potentials, y_dim))
        elif self.cost_function == "MLP":
            self.log_v_m = nn.Sequential(
                torchvision.ops.MLP(in_channels=x_dim, hidden_channels=[m_potentials], activation_layer=torch.nn.ReLU),
                nn.LogSoftmax(dim=-1),
            )
            self.b_m = torchvision.ops.MLP(
                in_channels=x_dim, hidden_channels=[m_potentials * y_dim], activation_layer=torch.nn.ReLU
            )

    def init_a_by_samples(self, samples):
        assert samples.shape[0] == self.a_n.shape[0]

        self.a_n.data = torch.clone(samples.to(self.a_n.device))

    def compute_cost(
        self,
        batched_y: torch.Tensor,
        log_v_m: torch.Tensor,
        b_m: torch.Tensor,
    ) -> torch.Tensor:  # -> [bs]
        if self.A_diagonal_init is not None:
            bT_y = torch.sum(
                b_m * batched_y[:, None, :], dim=2
            )  # sum([bs x M x y_dim] * [bs x 1 x y_dim], dim=2) = [bs x M]
            return -self.epsilon * torch.logsumexp(
                log_v_m + bT_y / self.epsilon, dim=1
            )  # sum([bs x M] +[bs x M], dim=1) = [bs]
        else:
            raise NotImplementedError("Other options are not implemented yet!")

    def compute_log_w_n(self):  # -> [N]
        return self.log_w_n

    def compute_a_n(self):  # -> [N x y_dim]
        return self.a_n

    def compute_A_n(self):  # -> [N x y_dim]
        if self.A_diagonal_init is not None:
            A_n = torch.exp(self.log_A_n)
        else:
            raise NotImplementedError("Other options are not implemented yet!")
        return A_n

    def compute_log_v_m(self, batched_x: torch.Tensor) -> torch.Tensor:  # -> [bs x M]
        batch_size = batched_x.shape[0]
        if self.cost_function == "parameters":
            return self.log_v_m.repeat(batch_size, 1)
        elif self.cost_function == "MLP":
            return self.log_v_m(batched_x)
        else:
            raise NotImplementedError("Other options are not implemented yet!")

    def compute_b_m(self, batched_x: torch.Tensor) -> torch.Tensor:  # -> [bs x M x y_dim]
        batch_size = batched_x.shape[0]
        if self.cost_function == "parameters":
            return self.b_m.repeat(batch_size, 1, 1)
        elif self.cost_function == "MLP":
            return self.b_m(batched_x).reshape(batch_size, self.m_potentials, self.y_dim)
        else:
            raise NotImplementedError("Other options are not implemented yet!")

    def compute_log_Z_nm(
        self, log_w_n: torch.Tensor, a_n: torch.Tensor, A_n: torch.Tensor, log_v_m: torch.Tensor, b_m: torch.Tensor
    ) -> torch.Tensor:  # -> [bs x N x M]
        if self.A_diagonal_init is not None:
            bT_A = (
                b_m[:, None, :, :] * A_n[None, :, None, :]
            )  # [bs x 1 x M x y_dim] * [1 x N x 1 x y_dim] = [bs x N x M x y_dim]
            correction = torch.sum(
                (bT_A + 2 * a_n[None, :, None, :]) * b_m[:, None, :, :], dim=3
            )  # sum(([bs x N x M x y_dim] + [1 x N x 1 x y_dim]) * [bs x 1 x M x y_dim], dim=3) = [bs x N x M]
            return (
                log_v_m[:, None, :] + log_w_n[None, :, None] + 0.5 * correction / self.epsilon
            )  # [bs x 1 x M] + [1 x N x 1] + [bs x N x M]
        else:
            raise NotImplementedError("Other options are not implemented yet!")

    def compute_primal_potential(
        self, batched_y: torch.Tensor, log_w_n: torch.Tensor, a_n: torch.Tensor, A_n: torch.Tensor
    ) -> torch.Tensor:  # -> [bs]
        if self.A_diagonal_init is not None:
            mix = Categorical(logits=log_w_n)
            comp = Independent(Normal(loc=a_n, scale=torch.sqrt(self.epsilon * A_n)), 1)  # [N x y_dim]
            gmm = MixtureSameFamily(mix, comp)
            return self.epsilon * gmm.log_prob(batched_y)  # [bs]

    def compute_dual_potential(
        self, log_w_n: torch.Tensor, a_n: torch.Tensor, A_n: torch.Tensor, log_v_m: torch.Tensor, b_m: torch.Tensor
    ) -> torch.Tensor:  # -> [bs]
        log_Z_nm = self.compute_log_Z_nm(log_w_n, a_n, A_n, log_v_m, b_m)
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

        log_w_n = self.compute_log_w_n()
        a_n = self.compute_a_n()
        A_n = self.compute_A_n()
        for i in range(num_sampling_iterations):
            sub_batch_x = batched_x[sampling_batch_size * i : sampling_batch_size * (i + 1)]

            b_m = self.compute_b_m(sub_batch_x)  # [bs x M x y_dim]
            log_v_m = self.compute_log_v_m(sub_batch_x)  # [bs x M]

            log_Z_nm = self.compute_log_Z_nm(log_w_n, a_n, A_n, log_v_m, b_m)  # [bs x N x M]

            logits = log_Z_nm.view(min(sampling_batch_size, batch_size), self.n_potentials * self.m_potentials)
            if self.A_diagonal_init is not None:
                scale = (
                    torch.sqrt(self.epsilon * A_n)[None, :, None, :]
                    .repeat(min(sampling_batch_size, batch_size), 1, self.m_potentials, 1)
                    .view(min(sampling_batch_size, batch_size), self.n_potentials * self.m_potentials, self.y_dim)
                )
                loc = (a_n[None, :, None, :] + A_n[None, :, None, :] * b_m[:, None, :, :]).view(
                    min(sampling_batch_size, batch_size), self.n_potentials * self.m_potentials, self.y_dim
                )  # view([1 x N x 1 x y_dim] + [1 x N x 1 x y_dim] * [bs x 1 x M x y_dim] = [bs x N x M x y_dim]) = [bs x N * M]
                mix = Categorical(logits=logits)
                comp = Independent(Normal(loc=loc, scale=scale), 1)
                gmm = MixtureSameFamily(mix, comp)

            else:
                raise NotImplementedError("Other options are not implemented yet!")

            samples.append(gmm.sample())

        samples = torch.cat(samples, dim=0)

        return samples

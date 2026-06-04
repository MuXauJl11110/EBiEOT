import torch
import torch.nn.functional as F
from src.ebieot.base import BaseEBiEOT
from src.ebieot.costs.lse import BaseLSECost
from torch import nn
from torch.distributions.categorical import Categorical
from torch.distributions.independent import Independent
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.normal import Normal


class EbieotGmm(BaseEBiEOT):
    """EBiEOT-GMM: GMM dual potentials with a learnable LSE cost."""

    def __init__(
        self,
        y_dim: int,
        n_potentials: int,
        cost: BaseLSECost,
        epsilon: float = 1,
        sampling_batch_size: int = 128,
        A_diagonal_init: float | None = 0.1,
    ):
        r"""
        :param int y_dim: Dimension of Y space
        :param int n_potentials: Number of potentials for approximating dual variable :math:`f(y)=\varepsilon\log\sum_{n=1}^N w_n \mathcal{N}(y\vert a_n, A_n/\varepsilon)`, defaults to 5
        :param float epsilon: Regularization parameter, defaults to 1.0
        :param BaseLSECost cost: LSE cost function class.
        :param int sampling_batch_size: Sampling batch size, defaults to 1
        :param float A_diagonal_init: Init of diagonal matrices for dual variable potential, defaults to 0.1
        """
        super().__init__()
        self.y_dim = y_dim
        self.n_potentials = n_potentials
        self.register_buffer("epsilon", torch.tensor(epsilon))

        self.cost = cost

        self.A_diagonal_init = A_diagonal_init
        assert A_diagonal_init is not None  # TODO: add non-diagonal initialization
        self.sampling_batch_size = sampling_batch_size

        self._log_w_n = nn.Parameter(torch.zeros(n_potentials))
        self._a_n = nn.Parameter(torch.randn(n_potentials, y_dim))
        if A_diagonal_init is not None:
            # softplus(raw) ≈ A_diagonal_init at t=0
            self._raw_A_n = nn.Parameter(
                torch.log(
                    torch.expm1(
                        torch.full((n_potentials, y_dim), float(A_diagonal_init))
                    )
                )
            )

    def init_a_by_samples(self, samples: torch.Tensor):
        assert samples.shape[0] == self._a_n.shape[0]

        self._a_n.data = torch.clone(samples.to(self._a_n.device))

    def log_w_n(self):  # -> [N]
        return F.log_softmax(self._log_w_n, dim=0)

    def a_n(self):  # -> [N x y_dim]
        return self._a_n

    def A_n(self):  # -> [N x y_dim]
        if self.A_diagonal_init is not None:
            A_n = F.softplus(self._raw_A_n) + 1e-12
        else:
            raise NotImplementedError("Other options are not implemented yet!")
        return A_n

    def log_Z_nm(
        self,
        log_w_n: torch.Tensor,
        a_n: torch.Tensor,
        A_n: torch.Tensor,
        log_v_m: torch.Tensor,
        b_m: torch.Tensor,
    ) -> torch.Tensor:  # -> [bs x N x M]
        if self.A_diagonal_init is not None:
            bT_A = (
                b_m[:, None, :, :] * A_n[None, :, None, :]
            )  # [bs x 1 x M x y_dim] * [1 x N x 1 x y_dim] = [bs x N x M x y_dim]
            correction = torch.sum(
                (bT_A + 2 * a_n[None, :, None, :]) * b_m[:, None, :, :], dim=3
            )  # sum(([bs x N x M x y_dim] + [1 x N x 1 x y_dim]) * [bs x 1 x M x y_dim], dim=3) = [bs x N x M]
            return (
                log_v_m[:, None, :]
                + log_w_n[None, :, None]
                + 0.5 * correction / self.epsilon
            ) + 1e-12  # [bs x 1 x M] + [1 x N x 1] + [bs x N x M]
        else:
            raise NotImplementedError("Other options are not implemented yet!")

    def f(
        self,
        batched_y: torch.Tensor,
        log_w_n: torch.Tensor,
        a_n: torch.Tensor,
        A_n: torch.Tensor,
    ) -> torch.Tensor:  # -> [bs]
        if self.A_diagonal_init is not None:
            mix = Categorical(logits=log_w_n)
            comp = Independent(
                Normal(loc=a_n, scale=torch.sqrt(self.epsilon * A_n)), 1
            )  # [N x y_dim]
            gmm = MixtureSameFamily(mix, comp)
            return self.epsilon * gmm.log_prob(batched_y)  # [bs]
        else:
            raise NotImplementedError("Other options are not implemented yet!")

    def f_c(
        self,
        log_w_n: torch.Tensor,
        a_n: torch.Tensor,
        A_n: torch.Tensor,
        log_v_m: torch.Tensor,
        b_m: torch.Tensor,
    ) -> torch.Tensor:  # -> [bs]
        log_Z_nm = self.log_Z_nm(log_w_n, a_n, A_n, log_v_m, b_m)
        return -self.epsilon * torch.logsumexp(log_Z_nm, dim=(1, 2))  # [bs]

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

        log_w_n = self.log_w_n()
        a_n = self.a_n()
        A_n = self.A_n()
        for i in range(num_sampling_iterations):
            sub_batch_x = batched_x[
                sampling_batch_size * i : sampling_batch_size * (i + 1)
            ]
            gmm = self.get_conditional_distribution(sub_batch_x, log_w_n, a_n, A_n)

            samples.append(gmm.sample())

        samples = torch.cat(samples, dim=0)

        return samples

    def get_conditional_distribution(
        self,
        batched_x: torch.Tensor,
        log_w_n: torch.Tensor,
        a_n: torch.Tensor,
        A_n: torch.Tensor,
    ) -> MixtureSameFamily:
        b_m = self.cost.b_m(batched_x)  # [bs x M x y_dim]
        log_v_m = self.cost.log_v_m(batched_x)  # [bs x M]

        log_Z_nm = self.log_Z_nm(log_w_n, a_n, A_n, log_v_m, b_m)  # [bs x N x M]

        logits = log_Z_nm.reshape(
            batched_x.shape[0], self.n_potentials * self.cost.m_potentials
        )
        if self.A_diagonal_init is not None:
            scale = (
                torch.sqrt(self.epsilon * A_n)[None, :, None, :]
                .repeat(batched_x.shape[0], 1, self.cost.m_potentials, 1)
                .reshape(
                    batched_x.shape[0],
                    self.n_potentials * self.cost.m_potentials,
                    self.y_dim,
                )
            )
            loc = (
                a_n[None, :, None, :] + A_n[None, :, None, :] * b_m[:, None, :, :]
            ).reshape(
                batched_x.shape[0],
                self.n_potentials * self.cost.m_potentials,
                self.y_dim,
            )  # view([1 x N x 1 x y_dim] + [1 x N x 1 x y_dim] * [bs x 1 x M x y_dim] = [bs x N x M x y_dim]) = [bs x N * M]
            mix = Categorical(logits=logits)
            comp = Independent(Normal(loc=loc, scale=scale), 1)
            gmm = MixtureSameFamily(mix, comp)
        else:
            raise NotImplementedError("Other options are not implemented yet!")

        return gmm

    def compute_paired_loss(
        self, X_paired: torch.Tensor, Y_paired: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        c = self.cost(X_paired, Y_paired)

        return {"loss": c.mean() / self.epsilon}

    def compute_unpaired_loss(
        self, X_unpaired: torch.Tensor, Y_unpaired: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        log_v_m = self.cost.log_v_m(X_unpaired)  # [bs x M]
        b_m = self.cost.b_m(X_unpaired)  # [bs x M x y_dim]

        log_w_n = self.log_w_n()  # [N]
        a_n = self.a_n()  # [N x y_dim]
        A_n = self.A_n()  # [N x y_dim]

        f_c = self.f_c(log_w_n, a_n, A_n, log_v_m, b_m)
        f = self.f(Y_unpaired, log_w_n, a_n, A_n)

        return {
            "log_w_n": log_w_n,
            "a_n": a_n,
            "A_n": A_n,
            "f_c": f_c,
            "f": f,
            "loss": -(f_c + f).mean() / self.epsilon,
        }

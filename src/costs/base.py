from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.func import grad, vmap


# TODO: remove x_dim, y_dim
class BaseCost(ABC, nn.Module):
    def __init__(
        self,
        x_dim: int = 2,
        y_dim: int = 2,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self._grad_y = vmap(grad(self.func, argnums=1), randomness="different")  # TODO: change?
        self._func = vmap(self.func, randomness="different")

    @abstractmethod
    def func(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # [1]
        pass

    def forward(self, batched_x: torch.Tensor, batched_y: torch.Tensor) -> torch.Tensor:  # [bs]
        return self._func(batched_x, batched_y)

    # WARNING: returns torch.Tensor with requires_grad=True if context manager torch.no_grad() was not used.
    def grad_y(self, batched_x: torch.Tensor, batched_y: torch.Tensor) -> torch.Tensor:  # [bs]
        return self._grad_y(batched_x, batched_y)


class BaseLSECost(BaseCost):
    def __init__(
        self,
        x_dim: int = 2,
        y_dim: int = 2,
        m_potentials: int = 25,
        epsilon: float = 1.0,
    ):
        r"""
        :param int x_dim: Dimension of X space, defaults to 2
        :param int y_dim: Dimension of Y space, defaults to 2
        :param int n_potentials: Number of potentials for approximating dual variable :math:`f(y)=\varepsilon\log\sum_{n=1}^N w_n \mathcal{N}(y\vert a_n, A_n/\varepsilon)`, defaults to 5
        :param int m_potentials: Number of potentials for approximating plan :math:`c(x, y)=-\varepsilon\log\sum_{m=1}^M v_m(x) \exp(\langle b_m(x), y \rangle) /\varepsilon`, defaults to 10
        :param float epsilon: Regularization parameter, defaults to 1.0
        """
        super().__init__(x_dim, y_dim)
        self.m_potentials = m_potentials
        self.register_buffer("epsilon", torch.tensor(epsilon))
        self.b_m = vmap(self.compute_b_m)  # batched version of self.compute_b_m
        self.log_v_m = vmap(self.compute_log_v_m)  # batched version of self.compute_log_v_m

    def func(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # -> [1]
        log_v_m = self.compute_log_v_m(x)
        b_m = self.compute_b_m(x)

        # sum([M x y_dim] * [1 x y_dim], dim=1) = [M]
        bT_y = torch.sum(b_m * y[None, :], dim=1)

        # sum([M] + [M], dim=0) = [1]
        return -self.epsilon * torch.logsumexp(log_v_m + bT_y / self.epsilon, dim=0)

    @abstractmethod
    def compute_b_m(self, x: torch.Tensor) -> torch.Tensor:  # -> [M x y_dim]
        pass

    @abstractmethod
    def compute_log_v_m(self, x: torch.Tensor) -> torch.Tensor:  # -> [M]
        pass

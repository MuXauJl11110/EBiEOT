from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.func import grad, vmap


class BaseCost(ABC, nn.Module):
    """Per-sample cost ``c(x, y)`` for EBiEOT.

    Subclasses implement ``func``. ``__init__`` wraps it with ``vmap`` so ``forward`` and ``grad_y``
    map over batch rows. ``x_dim`` / ``y_dim`` are feature sizes (not batch).
    """

    def __init__(
        self,
        x_dim: int = 2,
        y_dim: int = 2,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self._grad_y = vmap(grad(self.func, argnums=1))
        self._func = vmap(self.func)

    @abstractmethod
    def func(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # [1]
        """Scalar cost for one pair ``(x, y)`` (no batch dimension).

        Batched use goes through ``forward`` / ``grad_y``. Return shape is a scalar tensor (often
        ``()`` or ``(1,)`` depending on the subclass).
        """
        raise NotImplementedError()

    def forward(
        self, batched_x: torch.Tensor, batched_y: torch.Tensor
    ) -> torch.Tensor:  # [bs]
        """``c(x_i, y_i)`` for each row; inputs ``(bs, x_dim)``, ``(bs, y_dim)``, output ``(bs,)``."""
        return self._func(batched_x, batched_y)

    # WARNING: returns torch.Tensor with requires_grad=True if context manager torch.no_grad() was not used.
    def grad_y(
        self, batched_x: torch.Tensor, batched_y: torch.Tensor
    ) -> torch.Tensor:  # [bs, y_dim]
        """Gradient of ``c`` w.r.t. ``y`` per batch row; shape ``(bs, y_dim)``."""
        return self._grad_y(batched_x, batched_y)


class BaseLSECost(BaseCost):
    def __init__(
        self,
        x_dim: int = 2,
        y_dim: int = 2,
        m_potentials: int = 25,
        epsilon: float = 1.0,
    ):
        r"""Log-sum-exp cost with :math:`M` learnable potentials per ``x``.

        Args:
            x_dim: Dimension of :math:`x`. Default ``2``.
            y_dim: Dimension of :math:`y`. Default ``2``.
            m_potentials: Mixture size :math:`M` in
                :math:`c(x,y)=-\varepsilon\log\sum_{m=1}^M v_m(x)\,e^{\langle b_m(x),y\rangle/\varepsilon}`.
                Default ``25``.
            epsilon: Temperature :math:`\varepsilon` in the same formula. Default ``1.0``.
        """
        super().__init__(x_dim, y_dim)
        self.m_potentials = m_potentials
        self.register_buffer("epsilon", torch.tensor(epsilon))
        self.b_m = vmap(self.compute_b_m)  # batched version of self.compute_b_m
        self.log_v_m = vmap(
            self.compute_log_v_m
        )  # batched version of self.compute_log_v_m

    def func(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # -> [1]
        log_v_m = self.compute_log_v_m(x)
        b_m = self.compute_b_m(x)

        # sum([M x y_dim] * [1 x y_dim], dim=1) = [M]
        bT_y = torch.sum(b_m * y[None, :], dim=1)

        # sum([M] + [M], dim=0) = [1]
        return -self.epsilon * torch.logsumexp(log_v_m + bT_y / self.epsilon, dim=0)

    @abstractmethod
    def compute_b_m(self, x: torch.Tensor) -> torch.Tensor:  # -> [M x y_dim]
        """Coefficients :math:`b_m(x)`; shape ``(M, y_dim)``.

        ``x`` is a single sample (not batched), shape ``(x_dim,)``.
        """
        pass

    @abstractmethod
    def compute_log_v_m(self, x: torch.Tensor) -> torch.Tensor:  # -> [M]
        """``log v_m(x)`` for :math:`m=1..M`; shape ``(M,)``.

        ``x`` is a single sample (not batched), shape ``(x_dim,)``.
        """
        pass

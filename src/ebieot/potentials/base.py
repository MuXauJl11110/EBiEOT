from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.func import grad, vmap


class BasePotential(ABC, nn.Module):
    """Per-sample potential ``φ(y)`` for EBiEOT.

    Subclasses implement ``func``. ``__init__`` wraps it with ``vmap`` so ``forward`` and ``grad_y``
    map over batch rows. ``y_dim`` is the feature size (not batch).
    """

    def __init__(self, y_dim: int = 2):
        super().__init__()
        self.y_dim = y_dim
        self._grad_y = vmap(grad(self.func))
        self._func = vmap(self.func)

    @abstractmethod
    def func(self, y: torch.Tensor) -> torch.Tensor:  # [1]
        """Scalar potential for one ``y`` (no batch dimension).

        Batched use goes through ``forward`` / ``grad_y``. Return shape is a scalar tensor (often
        ``()`` or ``(1,)`` depending on the subclass).
        """
        raise NotImplementedError()

    def forward(self, batched_y: torch.Tensor) -> torch.Tensor:  # [bs]
        """``f(y_i)`` for each row; input ``(bs, y_dim)``, output ``(bs,)``."""
        return self._func(batched_y)

    # WARNING: returns torch.Tensor with requires_grad=True if context manager torch.no_grad() was not used.
    def grad_y(self, batched_y: torch.Tensor) -> torch.Tensor:  # [bs, y_dim]
        """Gradient of ``f`` w.r.t. ``y`` per batch row; shape ``(bs, y_dim)``."""
        return self._grad_y(batched_y)

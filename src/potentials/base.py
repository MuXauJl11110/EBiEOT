from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.func import grad, vmap


class BasePotential(ABC, nn.Module):
    def __init__(self, y_dim: int = 2):
        super().__init__()
        self.y_dim = y_dim
        self._grad_y = vmap(grad(self.func))
        self._func = vmap(self.func)

    @abstractmethod
    def func(self, y: torch.Tensor) -> torch.Tensor:  # [1]
        pass

    def forward(self, batched_y: torch.Tensor) -> torch.Tensor:  # [bs]
        return self._func(batched_y)

    def grad_y(self, batched_y: torch.Tensor) -> torch.Tensor:  # [bs]
        return self._grad_y(batched_y)

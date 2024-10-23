from typing import Callable

import torch
import torch.nn as nn

from src.auxiliary_models.mlp_based import FullyConnectedMLP
from src.costs.base import BaseCost


# TODO: add config to log_v_m and b_m
class MLPCost(BaseCost):
    def __init__(
        self,
        hidden_layers: list[int],
        activation_function: Callable[[], nn.Module],
        x_dim: int = 2,
        y_dim: int = 2,
    ):
        r"""
        :param int x_dim: Dimension of X space, defaults to 2
        :param int y_dim: Dimension of Y space, defaults to 3
        """
        super().__init__(x_dim, y_dim)

        self.net = FullyConnectedMLP(
            input_dim=x_dim + y_dim, hidden_layers=hidden_layers, output_dim=1, activation_function=activation_function
        )

    def func(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # [1]
        return self.net.func(torch.cat([x, y]))

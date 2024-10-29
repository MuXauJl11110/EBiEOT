from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.auxiliary_models.mlp_based import FullyConnectedMLP
from src.costs.base import BaseCost


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


class MLPL2Cost(BaseCost):
    def __init__(
        self,
        x_hidden_layers: list[int],
        x_activation_function: Callable[[], nn.Module],
        y_hidden_layers: list[int],
        y_activation_function: Callable[[], nn.Module],
        x_dim: int = 2,
        y_dim: int = 2,
    ):
        r"""
        :param int x_dim: Dimension of X space, defaults to 2
        :param int y_dim: Dimension of Y space, defaults to 3
        """
        super().__init__(x_dim, y_dim)

        self.x_net = FullyConnectedMLP(
            input_dim=x_dim, hidden_layers=x_hidden_layers, output_dim=x_dim, activation_function=x_activation_function
        )
        self.y_net = FullyConnectedMLP(
            input_dim=y_dim, hidden_layers=y_hidden_layers, output_dim=y_dim, activation_function=y_activation_function
        )

    def func(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # [1]
        # return torch.sum((self.x_net.func(x) - self.y_net.func(y)) ** 2)
        # return (
        #     1 - F.cosine_similarity(F.normalize(self.x_net(x[None, :])), F.normalize(self.y_net(y[None, :])))
        # ).squeeze()
        # return (1 - F.cosine_similarity(F.normalize(self.x_net(x[None, :])), F.normalize(y[None, :]))).squeeze()
        return torch.sum((self.x_net(x[None, :]) - y[None, :]) ** 2)

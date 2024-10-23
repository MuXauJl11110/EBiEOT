import torch
import torch.nn as nn
from torchvision.ops import MLP

from src.costs.base import BaseCost


# TODO: add config to log_v_m and b_m
class MLPCost(BaseCost):
    def __init__(
        self,
        hidden_channels: list[int],
        activation_layer: nn.Module,
        x_dim: int = 2,
        y_dim: int = 2,
    ):
        r"""
        :param int x_dim: Dimension of X space, defaults to 2
        :param int y_dim: Dimension of Y space, defaults to 3
        """
        super().__init__(x_dim, y_dim)

        self.cost = MLP(in_channels=x_dim, hidden_channels=hidden_channels, activation_layer=activation_layer)

    def func(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # [1]
        return self.cost(torch.stack([x, y])).squeeze()

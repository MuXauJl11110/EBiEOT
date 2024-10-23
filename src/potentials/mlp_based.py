import torch
import torch.nn as nn
from torchvision.ops import MLP

from src.potentials.base import BasePotential


class MLPPotential(BasePotential):
    def __init__(self, y_dim: int, hidden_channels: list[int], activation_layer: nn.Module = nn.ReLU):
        super().__init__(y_dim)

        self.net = MLP(in_channels=y_dim, hidden_channels=hidden_channels, activation_layer=activation_layer)

    def func(self, y: torch.Tensor) -> torch.Tensor:  # -> [1]
        return self.net(y).squeeze()

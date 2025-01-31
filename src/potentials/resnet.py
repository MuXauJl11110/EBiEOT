from typing import Callable

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from src.auxiliary_models.mlp_based import FullyConnectedMLP
from src.auxiliary_models.resnet import ResNet_D
from src.potentials.base import BasePotential


def deactivate_batchnorm(m):
    if isinstance(m, torch.nn.BatchNorm2d):
        m.reset_parameters()
        m.eval()
        with torch.no_grad():
            m.track_running_stats = False


class ResNet18Potential(BasePotential):
    def __init__(
        self,
        hidden_layers: list[int],
        activation_function: Callable[[], nn.Module],
        train_resnet: bool = False,
    ):
        super().__init__()
        if train_resnet:
            self.net = resnet18(weights=ResNet18_Weights.DEFAULT)
            self.net.apply(deactivate_batchnorm)
        else:
            self.net = resnet18(weights=ResNet18_Weights.DEFAULT).eval()
        self.linear = FullyConnectedMLP(
            input_dim=1000, hidden_layers=hidden_layers, output_dim=1, activation_function=activation_function
        )

    def func(self, y: torch.Tensor) -> torch.Tensor:  # -> [1]
        return self.linear(self.net(y.repeat(1, 3, 1, 1))).squeeze()


class ResNetPotential(BasePotential):
    def __init__(self, size: int = 64, nc: int = 3, nfilter: int = 64, nfilter_max: int = 512, res_ratio: float = 0.1):
        super().__init__()
        self.net = ResNet_D(size, nc, nfilter, nfilter_max, res_ratio)

    def func(self, y: torch.Tensor) -> torch.Tensor:  # -> [1]
        return self.net(y).squeeze()

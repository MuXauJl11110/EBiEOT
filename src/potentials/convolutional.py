from typing import Callable

import torch
import torch.nn as nn

from src.auxiliary_models.convolutional import NonlocalNet, VanillaNet
from src.auxiliary_models.mlp_based import FullyConnectedMLP
from src.auxiliary_models.resnet import ResNet_D
from src.auxiliary_models.unet import CondUNetV2
from src.auxiliary_models.unet_v2 import UNetForScalarOutput
from src.potentials.base import BasePotential


class VanillaPotential(BasePotential):
    def __init__(
        self,
        hidden_layers: list[int],
        activation_function: Callable[[], nn.Module],
        n_c: int = 3,
        n_f: int = 32,
        leak: float = 0.05,
    ):
        super().__init__()
        self.net = VanillaNet(n_c, n_f, leak)
        self.linear = FullyConnectedMLP(
            input_dim=28 * 28, hidden_layers=hidden_layers, output_dim=1, activation_function=activation_function
        )

    def func(self, y: torch.Tensor) -> torch.Tensor:  # -> [1]
        return self.linear.func(self.net(y).flatten()).squeeze()


class NonlocalPotential(BasePotential):
    def __init__(self, n_c: int = 3, n_f: int = 32, leak: float = 0.05):
        super().__init__()
        self.net = NonlocalNet(n_c, n_f, leak)

    def func(self, y: torch.Tensor) -> torch.Tensor:  # -> [1]
        return self.net(y[None, :, :, :]).squeeze()


class ResNetPotential(BasePotential):
    def __init__(self, size: int = 64, nc: int = 3, nfilter: int = 64, nfilter_max: int = 512, res_ratio: float = 0.1):
        super().__init__()
        self.net = ResNet_D(size, nc, nfilter, nfilter_max, res_ratio)

    def func(self, y: torch.Tensor) -> torch.Tensor:  # -> [1]
        return self.net(y).squeeze()


class UNetPotential(BasePotential):
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.net = UNetForScalarOutput(in_channels, 1)

    def func(self, y: torch.Tensor) -> torch.Tensor:  # -> [1]
        return self.net(y[None, :, :, :]).squeeze()

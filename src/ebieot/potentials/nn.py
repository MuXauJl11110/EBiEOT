from typing import Callable

import torch
import torch.nn as nn

from src.ebieot.potentials.base import BasePotential
from src.networks.mlp import FullyConnectedMLP


class MLPPotential(BasePotential):
    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int],
        activation_function: Callable[[], nn.Module],
    ):
        super().__init__(input_dim)

        self.net = FullyConnectedMLP(
            input_dim=input_dim,
            hidden_layers=hidden_layers,
            output_dim=1,
            activation_function=activation_function,
        )

    def func(self, y: torch.Tensor) -> torch.Tensor:  # -> [1]
        return self.net.func(y)


class ClassificationClassPotential(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_classes))

    def forward(self) -> torch.Tensor:
        return self.logits

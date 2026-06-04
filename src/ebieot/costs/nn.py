from typing import Callable

import torch
import torch.nn as nn

from src.ebieot.costs.base import BaseCost
from src.networks.mlp import FullyConnectedMLP


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
            input_dim=x_dim + y_dim,
            hidden_layers=hidden_layers,
            output_dim=1,
            activation_function=activation_function,
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
            input_dim=x_dim,
            hidden_layers=x_hidden_layers,
            output_dim=x_dim,
            activation_function=x_activation_function,
        )
        self.y_net = FullyConnectedMLP(
            input_dim=y_dim,
            hidden_layers=y_hidden_layers,
            output_dim=y_dim,
            activation_function=y_activation_function,
        )

    def func(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # [1]
        return torch.sum((self.x_net(x[None, :]) - y[None, :]) ** 2)


class ClassificationMlpEnergy(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        image_size: int,
        y_embed_dim: int = 32,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        x_dim = in_channels * image_size * image_size
        self.y_embed = nn.Embedding(num_classes, y_embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(x_dim + y_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x_batch: torch.Tensor) -> torch.Tensor:
        batch_size = x_batch.size(0)
        x_flat = x_batch.view(batch_size, -1)
        y_all = self.y_embed.weight.unsqueeze(0).expand(
            batch_size, self.num_classes, -1
        )
        x_all = x_flat.unsqueeze(1).expand(batch_size, self.num_classes, -1)
        input_tensor = torch.cat([x_all, y_all], dim=2)
        return self.mlp(input_tensor.view(batch_size * self.num_classes, -1)).view(
            batch_size, self.num_classes
        )


class ClassificationCnnEnergy(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        image_size: int,
        y_embed_dim: int = 64,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        feature_size = image_size // 4
        feature_dim = 64 * feature_size * feature_size
        self.feature_projector = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
        )
        self.y_embed = nn.Embedding(num_classes, y_embed_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + y_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x_batch: torch.Tensor) -> torch.Tensor:
        batch_size = x_batch.size(0)
        feature = self.encoder(x_batch).view(batch_size, -1)
        feature = self.feature_projector(feature)
        feature_all = feature.unsqueeze(1).expand(batch_size, self.num_classes, -1)
        y_all = self.y_embed.weight.unsqueeze(0).expand(
            batch_size, self.num_classes, -1
        )
        input_tensor = torch.cat([feature_all, y_all], dim=2)
        return self.head(input_tensor.view(batch_size * self.num_classes, -1)).view(
            batch_size, self.num_classes
        )

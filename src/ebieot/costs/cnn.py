"""Image-space costs for colored MNIST."""


import torch
import torch.nn as nn

from src.ebieot.costs.base import BaseCost
from src.networks.cnn.convolutional import NonlocalNet, VanillaNet
from src.networks.cnn.unet import UNet2


class SquareCost(BaseCost):
    """Squared L2 cost on RGB images."""

    def __init__(self, x_dim: int = 3072, y_dim: int = 3072):
        super().__init__(x_dim=x_dim, y_dim=y_dim)

    def func(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (x - y).square().mean()


class VanillaCost(BaseCost):
    def __init__(
        self,
        n_c: int = 3,
        n_f: int = 32,
        leak: float = 0.05,
        x_dim: int = 3072,
        y_dim: int = 3072,
    ):
        super().__init__(x_dim=x_dim, y_dim=y_dim)
        self.net = VanillaNet(n_c * 2, n_f, leak)

    def func(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, y], dim=0))


class NonlocalCost(BaseCost):
    def __init__(
        self,
        n_c: int = 3,
        n_f: int = 32,
        leak: float = 0.05,
        x_dim: int = 3072,
        y_dim: int = 3072,
    ):
        super().__init__(x_dim=x_dim, y_dim=y_dim)
        self.net = NonlocalNet(n_c * 2, n_f, leak)

    def func(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pair = torch.cat([x.unsqueeze(0), y.unsqueeze(0)], dim=1)
        return self.net(pair).squeeze()


class UNetCost(BaseCost):
    def __init__(
        self,
        n_c: int = 3,
        num_layers: int = 3,
        base_filters: int = 64,
        x_dim: int = 3072,
        y_dim: int = 3072,
    ):
        super().__init__(x_dim=x_dim, y_dim=y_dim)
        self.net = nn.Sequential(
            UNet2(
                in_channels=n_c,
                out_channels=n_c,
                num_layers=num_layers,
                base_filters=base_filters,
            ),
            nn.Tanh(),
        )

    def func(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pred = self.net(x.unsqueeze(0)).squeeze(0)
        return (pred - y).square().mean()

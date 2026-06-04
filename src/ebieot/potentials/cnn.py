"""Image-space potentials for colored MNIST."""


import torch

from src.ebieot.potentials.base import BasePotential
from src.networks.cnn.convolutional import NonlocalNet, VanillaNet


class VanillaPotential(BasePotential):
    def __init__(
        self,
        n_c: int = 3,
        n_f: int = 32,
        leak: float = 0.05,
        y_dim: int = 3072,
    ):
        super().__init__(y_dim=y_dim)
        self.net = VanillaNet(n_c, n_f, leak)

    def func(self, y: torch.Tensor) -> torch.Tensor:
        return self.net(y.unsqueeze(0)).squeeze()


class NonlocalPotential(BasePotential):
    def __init__(
        self,
        n_c: int = 3,
        n_f: int = 32,
        leak: float = 0.05,
        y_dim: int = 3072,
    ):
        super().__init__(y_dim=y_dim)
        self.net = NonlocalNet(n_c, n_f, leak)

    def func(self, y: torch.Tensor) -> torch.Tensor:
        return self.net(y.unsqueeze(0)).squeeze()

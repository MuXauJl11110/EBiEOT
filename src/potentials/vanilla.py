import torch

from src.auxiliary_models.convolutional import NonlocalNet, VanillaNet
from src.potentials.base import BasePotential


class VanillaPotential(BasePotential):
    def __init__(self, n_c: int = 3, n_f: int = 32, leak: float = 0.05):
        super().__init__()
        self.net = VanillaNet(n_c, n_f, leak)

    def func(self, y: torch.Tensor) -> torch.Tensor:  # -> [1]
        return self.net(y).squeeze()


class NonlocalPotential(BasePotential):
    def __init__(self, n_c: int = 3, n_f: int = 32, leak: float = 0.05):
        super().__init__()
        self.net = NonlocalNet(n_c, n_f, leak)

    def func(self, y: torch.Tensor) -> torch.Tensor:  # -> [1]
        return self.net(y[None, :, :, :]).squeeze()

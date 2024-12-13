import torch

from src.auxiliary_models.convolutional import NonlocalNet, VanillaNet
from src.auxiliary_models.resnet import ResNet_D
from src.auxiliary_models.unet import CondUNetV2
from src.auxiliary_models.unet_v2 import UNetForScalarOutput
from src.costs.base import BaseCost


class VanillaCost(BaseCost):
    def __init__(self, n_c: int = 3, n_f: int = 32, leak: float = 0.05):
        super().__init__()
        self.net = VanillaNet(n_c * 2, n_f, leak)

    def func(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # [1]
        return self.net(torch.cat([x, y]))


class NonlocalCost(BaseCost):
    def __init__(self, n_c: int = 3, n_f: int = 32, leak: float = 0.05):
        super().__init__()
        self.net = NonlocalNet(n_c * 2, n_f, leak)

    def func(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # [1]
        return self.net(torch.cat([x[None, :, :, :], y[None, :, :, :]], dim=1)).squeeze()


class ResNetCost(BaseCost):
    def __init__(self, size: int = 64, nc: int = 3, nfilter: int = 64, nfilter_max: int = 512, res_ratio: float = 0.1):
        super().__init__()
        self.net = ResNet_D(size, nc * 2, nfilter, nfilter_max, res_ratio)

    def func(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # [1]
        return self.net(torch.cat([x, y], dim=1)).squeeze()


class UnetCost(BaseCost):
    def __init__(self, n_channels: int, n_classes: int, z_channels: int, base_factor: int = 32):
        super().__init__()
        self.net = CondUNetV2(n_channels, n_classes, z_channels, base_factor)

    def func(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # [1]
        return self.net(x[None, :, :, :], y[None, :, :, :]).squeeze()
        # return torch.jit.trace(self.net, example_inputs=[x, y])


class UnetV2Cost(BaseCost):
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.net = UNetForScalarOutput(2 * in_channels, 1)

    def func(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # [1]
        input = torch.cat([x[None, :, :, :], y[None, :, :, :]], dim=1)
        return self.net(input).squeeze()
        # return torch.jit.trace(self.net, example_inputs=[x, y])

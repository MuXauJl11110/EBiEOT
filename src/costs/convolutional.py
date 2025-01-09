import torch
import torch.nn as nn
from torchvision.transforms import functional

from src.auxiliary_models.convolutional import NonlocalNet, VanillaNet
from src.auxiliary_models.resnet import ResNet_D
from src.auxiliary_models.unet import UNet
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


class UNetCost(BaseCost):
    def __init__(self, n_c: int = 3, num_layers: int = 3, base_filters: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            UNet(in_channels=n_c, out_channels=n_c, num_layers=num_layers, base_filters=base_filters), nn.Tanh()
        )

    def func(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # [1]
        return (self.net(x.unsqueeze(0)) - y.unsqueeze(0)).square().mean()


class UNetV2Cost(BaseCost):
    def __init__(self, n_c: int = 3, num_layers: int = 3, base_filters: int = 64):
        super().__init__()
        self.net = UNet(in_channels=n_c, out_channels=n_c, num_layers=num_layers, base_filters=base_filters)

    def func(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # [1]
        return (self.net(x.unsqueeze(0)) - y.roll(shifts=-1, dims=0).unsqueeze(0)).square().mean()


class UNetV3Cost(BaseCost):
    def __init__(self, n_c: int = 3, num_layers: int = 3, base_filters: int = 64):
        super().__init__()
        self.net = UNet(in_channels=n_c, out_channels=n_c, num_layers=num_layers, base_filters=base_filters)

    def func(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # [1]
        return (self.net(y.unsqueeze(0)) - x.unsqueeze(0)).square().mean()


class UNetV4Cost(BaseCost):
    def __init__(self, n_c: int = 3, num_layers: int = 3, base_filters: int = 64):
        super().__init__()
        self.net = UNet(in_channels=n_c, out_channels=n_c, num_layers=num_layers, base_filters=base_filters)

    def func(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # [1]
        with torch.no_grad():
            self.eval()
            return (functional.adjust_hue(x, 1 / 3) - y).square().mean()

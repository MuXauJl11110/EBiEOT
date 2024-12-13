import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG16_Weights

from src.costs.base import BaseCost
from transformation_cr.model import Net
from transformation_cr.pytorch_tcr import TCR


class TCRCost(BaseCost):
    def __init__(
        self,
        upscale_factor: int,
        device: str = "cuda",
    ):
        super().__init__()
        self.tcr = TCR()
        self.device = device
        self.model = Net(upscale_factor=upscale_factor).to(device)
        self.mse_loss = nn.MSELoss()

        vgg = models.vgg16(weights=VGG16_Weights.DEFAULT)
        self.loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()

    def func(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # [1]
        return self.perception_loss(self.model(x).repeat(3, 1, 1), y.repeat(3, 1, 1))

    # def func(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # [1]
    #     random = torch.rand((1, 1))
    #     transformed_input = self.tcr(x, random, self.device)

    #     output_transformed = self.model(transformed_input)
    #     input_model_transformed = self.tcr(self.model(x), random, self.device)

    #     return self.perception_loss(
    #         torch.cat((output_transformed, output_transformed, output_transformed), 1),
    #         torch.cat((input_model_transformed, input_model_transformed, input_model_transformed), 1),
    #     )

    def perception_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        perception_loss = self.mse_loss(self.loss_network(x), self.loss_network(y))
        return perception_loss

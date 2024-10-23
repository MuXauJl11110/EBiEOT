from typing import Callable

import torch.nn as nn
from pydantic import BaseModel


class PotentialConfig(BaseModel):
    y_dim: int = 2
    hidden_channels: list[int] = [256, 256, 256, 1]
    activation_layer: Callable[[], nn.Module] = lambda: nn.LeakyReLU(0.2)

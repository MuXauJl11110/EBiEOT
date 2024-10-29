from typing import Callable

import torch.nn as nn
from pydantic import BaseModel


class PotentialConfig(BaseModel):
    input_dim: int = 2
    hidden_layers: list[int] = [256, 256]
    activation_function: Callable[[], nn.Module] = lambda: nn.LeakyReLU(0.2)
    output_dim: int = 1

from typing import Callable

import torch.nn as nn
from pydantic import model_validator

from configs.base.cost import BaseCostConfig


class MLPLSECostConfig(BaseCostConfig):
    m_potentials: int = 25
    epsilon: float = 1.0

    log_v_m_hidden_channels: list[int] = [128, 128]
    b_m_hidden_channels: list[int] = [128, 128]

    @model_validator(mode="after")
    def append_hidden_channels(self):
        self.log_v_m_hidden_channels.append(self.m_potentials)
        self.b_m_hidden_channels.append(self.m_potentials * self.y_dim)

        return self


class MLPCostConfig(BaseCostConfig):
    hidden_channels: list[int] = [256, 256, 1]
    activation_layer: Callable[[], nn.Module] = lambda: nn.LeakyReLU(0.2)

from typing import Callable

import torch.nn as nn
from configs.base.cost import BaseCostConfig
from pydantic import model_validator


class MLPLSECostConfig(BaseCostConfig):
    m_potentials: int = 4
    epsilon: float = 1.0

    log_v_m_hidden_channels: list[int] = [128, 128]
    b_m_hidden_channels: list[int] = [256, 256]


class MLPCostConfig(BaseCostConfig):
    hidden_layers: list[int] = [256]
    activation_function: Callable[[], nn.Module] = lambda: nn.LeakyReLU(0.2)


class MLPL2CostConfig(BaseCostConfig):
    x_hidden_layers: list[int] = [128, 128, 1]
    x_activation_function: Callable[[], nn.Module] = lambda: nn.LeakyReLU(0.2)
    y_hidden_layers: list[int] = [128, 128, 1]
    y_activation_function: Callable[[], nn.Module] = lambda: nn.LeakyReLU(0.2)

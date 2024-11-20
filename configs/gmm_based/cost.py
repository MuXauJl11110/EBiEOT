from typing import Callable

import torch.nn as nn
from configs.base.cost import BaseCostConfig


class MLPLSECostConfig(BaseCostConfig):
    m_potentials: int = 10
    epsilon: float = 1.0

    log_v_m_hidden_channels: list[int] = [128, 128]
    b_m_hidden_channels: list[int] = [256, 256]

    log_v_m_activation_layer: Callable[[], nn.Module] = lambda: nn.ReLU()
    b_m_activation_layer: Callable[[], nn.Module] = lambda: nn.ReLU()

import torch
import torch.nn as nn
import torchvision
from src.costs.base import BaseLSECost


class MLPLSECost(BaseLSECost):
    def __init__(
        self,
        log_v_m_hidden_channels: list[int],
        b_m_hidden_channels: list[int],
        x_dim: int = 2,
        y_dim: int = 2,
        m_potentials: int = 25,
        epsilon: float = 1.0,
        log_v_m_activation_layer: nn.Module = nn.ReLU,
        b_m_activation_layer: nn.Module = nn.ReLU,
    ):
        r"""
        :param int x_dim: Dimension of X space, defaults to 2
        :param int y_dim: Dimension of Y space, defaults to 3
        :param int m_potentials: Number of potentials for approximating plan :math:`c(x, y)=-\varepsilon\log\sum_{m=1}^M v_m(x) \exp(\langle b_m(x), y \rangle) /\varepsilon`, defaults to 10
        :param float epsilon: Regularization parameter, defaults to 1.0
        """
        super().__init__(x_dim, y_dim)
        self.m_potentials = m_potentials
        self.register_buffer("epsilon", torch.tensor(epsilon))

        self._log_v_m = nn.Sequential(
            torchvision.ops.MLP(
                in_channels=x_dim, hidden_channels=log_v_m_hidden_channels, activation_layer=log_v_m_activation_layer
            ),
            nn.LogSoftmax(dim=-1),
        )

        self._b_m = torchvision.ops.MLP(
            in_channels=x_dim, hidden_channels=b_m_hidden_channels, activation_layer=b_m_activation_layer
        )

    def compute_log_v_m(self, x: torch.Tensor) -> torch.Tensor:  # [M]
        return self._log_v_m(x[None, :]).reshape(self.m_potentials)

    def compute_b_m(self, x: torch.Tensor) -> torch.Tensor:  # [M x y_dim]
        return self._b_m(x[None, :]).reshape(self.m_potentials, self.y_dim)


class SharedMLPLSECost(BaseLSECost):
    def __init__(
        self,
        shared_hidden_channels: list[int],
        x_dim: int = 2,
        y_dim: int = 2,
        m_potentials: int = 25,
        epsilon: float = 1.0,
        activation_layer: nn.Module = nn.SiLU,  # Upgraded to SiLU
        use_layer_norm: bool = True,
    ):
        super().__init__(x_dim, y_dim)
        self.m_potentials = m_potentials
        self.register_buffer("epsilon", torch.tensor(epsilon))

        # 1. Shared Feature Extractor
        layers = []
        in_dim = x_dim
        for hidden_dim in shared_hidden_channels:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(activation_layer())
            in_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)

        # 2. Heads (Mapping latent to targets)
        latent_dim = shared_hidden_channels[-1] if shared_hidden_channels else x_dim

        self.log_v_m_head = nn.Sequential(nn.Linear(latent_dim, m_potentials), nn.LogSoftmax(dim=-1))

        self.b_m_head = nn.Linear(latent_dim, m_potentials * y_dim)

        # 3. Safe Initialization to prevent early LogSumExp saturation
        nn.init.normal_(self.b_m_head.weight, std=0.01)
        nn.init.zeros_(self.b_m_head.bias)

    def compute_log_v_m(self, x: torch.Tensor) -> torch.Tensor:  # [M]
        features = self.backbone(x[None, :])
        return self.log_v_m_head(features).reshape(self.m_potentials)

    def compute_b_m(self, x: torch.Tensor) -> torch.Tensor:  # [M, y_dim]
        features = self.backbone(x[None, :])
        return self.b_m_head(features).reshape(self.m_potentials, self.y_dim)

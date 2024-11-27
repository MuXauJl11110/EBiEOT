import torch
import torch.nn as nn
import torchvision

from src.auxiliary_models.mlp_based import FullyConnectedMLP
from src.costs.base import BaseLSECost


# TODO: add config to log_v_m and b_m
class MLPLSECost(BaseLSECost):
    def __init__(
        self,
        log_v_m_hidden_channels: list[int],
        b_m_hidden_channels: list[int],
        x_dim: int = 2,
        y_dim: int = 2,
        m_potentials: int = 25,
        epsilon: float = 1.0,
        log_v_m_activation_layer: nn.Module = lambda: nn.LeakyReLU(0.2),
        b_m_activation_layer: nn.Module = lambda: nn.LeakyReLU(0.2),
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
                in_channels=x_dim, hidden_channels=log_v_m_hidden_channels, activation_layer=torch.nn.ReLU
            ),
            nn.LogSoftmax(dim=-1),
        )

        self._b_m = torchvision.ops.MLP(
            in_channels=x_dim, hidden_channels=b_m_hidden_channels, activation_layer=torch.nn.ReLU
        )

        # Parametrization below work wierdly. TODO: Fix it
        # self._log_v_m = nn.Sequential(
        #     FullyConnectedMLP(
        #         input_dim=x_dim,
        #         hidden_layers=log_v_m_hidden_channels,
        #         output_dim=m_potentials,
        #         activation_function=log_v_m_activation_layer,
        #     ),
        #     nn.LogSoftmax(dim=-1),
        # )
        # self._b_m = FullyConnectedMLP(
        #     input_dim=x_dim,
        #     hidden_layers=b_m_hidden_channels,
        #     output_dim=m_potentials * y_dim,
        #     activation_function=b_m_activation_layer,
        # )

    def compute_log_v_m(self, x: torch.Tensor) -> torch.Tensor:  # [M]
        # return self._log_v_m(x[None, :]).squeeze()
        return self._log_v_m(x[None, :]).reshape(self.m_potentials)

    def compute_b_m(self, x: torch.Tensor) -> torch.Tensor:  # [M x y_dim]
        # return self._b_m(x[None, :]).reshape(self.m_potentials, self.y_dim).squeeze()
        return self._b_m(x[None, :]).reshape(self.m_potentials, self.y_dim)

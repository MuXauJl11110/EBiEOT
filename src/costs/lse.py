import torch
import torch.nn as nn
import torchvision

from src.auxiliary_models.mlp_based import FullyConnectedMLP
from src.auxiliary_models.unet import UNet
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
        super().__init__(m_potentials, epsilon)

        self.x_dim = x_dim
        self.y_dim = y_dim
        log_v_m_hidden_channels.append(m_potentials)
        self._log_v_m = nn.Sequential(
            torchvision.ops.MLP(
                in_channels=x_dim, hidden_channels=log_v_m_hidden_channels, activation_layer=log_v_m_activation_layer
            ),
            nn.LogSoftmax(dim=-1),
        )

        b_m_hidden_channels.append(m_potentials * y_dim)
        # self._b_m = torchvision.ops.MLP(
        #     in_channels=x_dim, hidden_channels=b_m_hidden_channels, activation_layer=b_m_activation_layer
        # )
        self._b_m = UNet(1, 1, True)

    def compute_log_v_m(self, x: torch.Tensor) -> torch.Tensor:  # [M]
        # if self.m_potentials == 1:
        #     return self._log_v_m(x[None, :])
        # else:
        #     return self._log_v_m(x[None, :]).squeeze()
        # return self._log_v_m(x[None, :]).squeeze()
        # return self._log_v_m(x[None, :])
        return self._log_v_m(x[None, :]).squeeze()

    def compute_b_m(self, x: torch.Tensor) -> torch.Tensor:  # [M x y_dim]
        # if self.m_potentials == 1:
        #     return self._b_m(x[None, :]).reshape(self.m_potentials, self.y_dim)
        # else:
        #     return self._b_m(x[None, :]).reshape(self.m_potentials, self.y_dim).squeeze()
        # return self._b_m(x[None, None, None, :]).squeeze()
        # return self._b_m(x[None, :]).reshape(self.m_potentials, self.y_dim)
        return (
            self._b_m(x[None, None, None, :])
            .repeat(1, 1, self.m_potentials, 1)
            .reshape(self.m_potentials, self.y_dim)
            .squeeze()
        )


class BatchedLSECost(nn.Module):
    def __init__(
        self,
        b_m_net: nn.Module,
        log_v_m_net: nn.Module,
        m_potentials: int = 16,
        epsilon: float = 1.0,
        y_dim: int = 512,
    ):
        r"""
        :param int n_potentials: Number of potentials for approximating dual variable :math:`f(y)=\varepsilon\log\sum_{n=1}^N w_n \mathcal{N}(y\vert a_n, A_n/\varepsilon)`, defaults to 5
        :param int m_potentials: Number of potentials for approximating plan :math:`c(x, y)=-\varepsilon\log\sum_{m=1}^M v_m(x) \exp(\langle b_m(x), y \rangle) /\varepsilon`, defaults to 10
        :param float epsilon: Regularization parameter, defaults to 1.0
        """
        super().__init__()
        self.m_potentials = m_potentials
        self.register_buffer("epsilon", torch.tensor(epsilon))
        self.b_m_net = b_m_net
        self.log_v_m_net = log_v_m_net
        self.y_dim = y_dim

    def forward(self, batched_x: torch.Tensor, batched_y: torch.Tensor) -> torch.Tensor:  # -> [1]
        log_v_m = self.log_v_m(batched_x)
        b_m = self.b_m(batched_x)

        # sum([bs x M x y_dim] * [bs x 1 x y_dim], dim=1) = [bs x M]
        bT_y = torch.sum(b_m * batched_y[:, None, :], dim=2)

        # sum([bs x M] + [bsx M], dim=0) = [1]
        return -self.epsilon * torch.logsumexp(
            log_v_m[
                :,
                None,
            ]
            + bT_y / self.epsilon,
            dim=1,
        )

    def b_m(self, batched_x: torch.Tensor) -> torch.Tensor:  # -> [bs x M x y_dim]
        return self.b_m_net(batched_x[:, None, None, :].repeat(1, 1, self.m_potentials, 1)).squeeze()

    def log_v_m(self, batched_x: torch.Tensor) -> torch.Tensor:  # -> [bs x M]
        return self.log_v_m_net(batched_x)

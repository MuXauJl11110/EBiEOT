from functools import partial
from typing import Callable

import torch
import torch.nn as nn


class MLPDiscriminator(nn.Module):
    def __init__(
        self,
        x_dim: int = 2,
        layers: list[int] = [128, 128, 128],
        active: Callable[[], nn.Module] = partial(nn.LeakyReLU, 0.2),
    ):
        super().__init__()

        self.x_dim = x_dim

        self.model = []
        ch_prev = x_dim

        for ch_next in layers:
            self.model.append(nn.Linear(ch_prev, ch_next))
            self.model.append(active())
            ch_prev = ch_next

        self.model.append(nn.Linear(ch_prev, 1))
        self.model = nn.Sequential(*self.model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze()


class CGANDiscriminator(nn.Module):
    def __init__(
        self,
        x_dim: int = 2,
        y_dim: int = 2,
        n_y: int = 4,
        layers: list[int] = [128, 128, 128],
        active: Callable[[], nn.Module] = partial(nn.LeakyReLU, 0.2),
    ):
        """
        :param int x_dim: Dimension of X space, defaults to 2
        :param int y_dim: Dimension of Y space, defaults to 2
        :param int n_y: Number of Y categories, defaults to 4
        :param list[int] layers: List of hidden layers sizes, defaults to [128, 128, 128]
        :param Callable[[], nn.Module] active: Activation function, defaults to partial(nn.LeakyReLU, 0.2)
        """
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim

        self.model = []
        ch_prev = 2 * x_dim + y_dim

        self.t_transform = nn.Embedding(n_y, y_dim)

        for ch_next in layers:
            self.model.append(nn.Linear(ch_prev, ch_next))
            self.model.append(active())
            ch_prev = ch_next

        self.model.append(nn.Linear(ch_prev, 1))
        self.model = nn.Sequential(*self.model)

    def forward(
        self, x_y: torch.Tensor, y: torch.Tensor, x_y_next: torch.Tensor
    ) -> torch.Tensor:
        return self.model(
            torch.cat([x_y, self.t_transform(y), x_y_next], dim=1)
        ).squeeze()

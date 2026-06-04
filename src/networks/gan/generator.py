from functools import partial
from typing import Callable

import torch
import torch.nn as nn


class MLPGenerator(nn.Module):
    def __init__(
        self,
        x_dim: int = 2,
        z_dim: int = 1,
        out_dim: int = 2,
        layers: list[int] = [128, 128, 128],
        active: Callable[[], nn.Module] = partial(nn.LeakyReLU, 0.2),
    ):
        super().__init__()

        self.x_dim = x_dim
        self.z_dim = z_dim

        self.model = []
        ch_prev = x_dim + z_dim

        for ch_next in layers:
            self.model.append(nn.Linear(ch_prev, ch_next))
            self.model.append(active())
            ch_prev = ch_next

        self.model.append(nn.Linear(ch_prev, out_dim))
        self.model = nn.Sequential(*self.model)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        if z.shape != (batch_size, self.z_dim):
            z = z.reshape((batch_size, self.z_dim))

        return self.model(
            torch.cat(
                [
                    x,
                    z,
                ],
                dim=1,
            )
        )


class CGANGenerator(nn.Module):
    def __init__(
        self,
        x_dim: int = 2,
        y_dim: int = 2,
        n_y: int = 4,
        z_dim: int = 1,
        out_dim: int = 2,
        layers: list[int] = [128, 128, 128],
        active: Callable[[], nn.Module] = partial(nn.LeakyReLU, 0.2),
    ):
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim

        self.model = []
        ch_prev = x_dim + y_dim + z_dim

        self.y_transform = nn.Embedding(
            n_y,
            y_dim,
        )

        for ch_next in layers:
            self.model.append(nn.Linear(ch_prev, ch_next))
            self.model.append(active())
            ch_prev = ch_next

        self.model.append(nn.Linear(ch_prev, out_dim))
        self.model = nn.Sequential(*self.model)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        batch_size = x.shape[0]

        if z.shape != (batch_size, self.z_dim):
            z = z.reshape((batch_size, self.z_dim))

        return self.model(
            torch.cat(
                [
                    x,
                    self.y_transform(y),
                    z,
                ],
                dim=1,
            )
        )

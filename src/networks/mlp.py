import itertools
from typing import Callable

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class FullyConnectedMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int],
        output_dim: int,
        activation_function: Callable[[], nn.Module] = lambda: nn.SiLU(),
        sn_iters=0,
    ):
        def apply_spectral_norm(module: nn.Module):
            if sn_iters == 0:
                return module
            return spectral_norm(module, n_power_iterations=sn_iters)

        assert isinstance(hidden_layers, list)
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hiddens = hidden_layers

        model = []
        prev_h = input_dim
        for h in hidden_layers:
            model.append(apply_spectral_norm(nn.Linear(prev_h, h)))
            model.append(activation_function())
            prev_h = h
        model.append(apply_spectral_norm(nn.Linear(hidden_layers[-1], output_dim)))
        self.net = nn.Sequential(*model)

    def forward(self, batched_x: torch.Tensor) -> torch.Tensor:  # -> [bs x output_dim]
        batch_size = batched_x.shape[0]
        batched_x = batched_x.view(batch_size, -1)
        return self.net(batched_x).view(batch_size, self.output_dim)

    def func(self, x: torch.Tensor) -> torch.Tensor:  # -> [1]
        return self.net(x).squeeze()


class MLPnet(nn.Sequential):
    def __init__(self, input_size: int, hidden_size: int, num_hidden_layers: int):
        super().__init__(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(True),
            *itertools.chain.from_iterable(
                (nn.Linear(hidden_size, hidden_size), nn.ReLU(True))
                for _ in range(num_hidden_layers)
            ),
            nn.Linear(hidden_size, input_size)
        )

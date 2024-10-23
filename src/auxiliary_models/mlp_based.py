from typing import Callable

import torch
import torch.nn as nn

from src.utils.energy_based import spectral_norm


class FullyConnectedMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int],
        output_dim: int,
        activation_function: Callable[[], nn.Module] = lambda: nn.ReLU(),
        sn_iters=0,
    ):
        def _SN(module: nn.Module):
            if sn_iters == 0:
                return module
            return spectral_norm(module, init=False, zero_bias=False, n_iters=sn_iters)

        assert isinstance(hidden_layers, list)
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hiddens = hidden_layers

        model = []
        prev_h = input_dim
        for h in hidden_layers:
            model.append(_SN(nn.Linear(prev_h, h)))
            model.append(activation_function())
            prev_h = h
        model.append(_SN(nn.Linear(hidden_layers[-1], output_dim)))
        self.net = nn.Sequential(*model)

    def forward(self, batched_x: torch.Tensor) -> torch.Tensor:  # -> [bs x output_dim]
        batch_size = batched_x.shape[0]
        batched_x = batched_x.view(batch_size, -1)
        return self.net(batched_x).view(batch_size, self.output_dim)

    def func(self, x: torch.Tensor) -> torch.Tensor:  # -> [1]
        return self.net(x).squeeze()

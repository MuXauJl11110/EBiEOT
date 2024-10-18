import torch.nn as nn

from src.utils.energy_based import spectral_norm


class FullyConnectedMLP(nn.Module):
    def __init__(
        self, input_dim: int, hiddens: list[int], output_dim: int, activation_gen=lambda: nn.ReLU(), sn_iters=0
    ):
        def _SN(module: nn.Module):
            if sn_iters == 0:
                return module
            return spectral_norm(module, init=False, zero_bias=False, n_iters=sn_iters)

        assert isinstance(hiddens, list)
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hiddens = hiddens

        model = []
        prev_h = input_dim
        for h in hiddens:
            model.append(_SN(nn.Linear(prev_h, h)))
            model.append(activation_gen())
            prev_h = h
        model.append(_SN(nn.Linear(hiddens[-1], output_dim)))
        self.net = nn.Sequential(*model)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        return self.net(x).view(batch_size, self.output_dim)

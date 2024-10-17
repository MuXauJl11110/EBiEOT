from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.func import grad, vmap

from src.utils.energy_based import spectral_norm


class CostBase(ABC, nn.Module):
    def __init__(
        self,
        x_dim: int = 2,
        y_dim: int = 2,
    ):
        super(CostBase, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self._grad_y = vmap(grad(self.func, argnums=1))
        self._func = vmap(self.func)

    @abstractmethod
    def func(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # [1]
        pass

    def forward(self, batched_x: torch.Tensor, batched_y: torch.Tensor) -> torch.Tensor:  # [bs]
        return self._func(batched_x, batched_y)

    def grad_y(self, batched_x: torch.Tensor, batched_y: torch.Tensor) -> torch.Tensor:  # [bs]
        return self._grad_y(batched_x, batched_y)


class MLPCost(CostBase):
    def __init__(
        self,
        x_dim: int = 2,
        y_dim: int = 2,
        n_potentials: int = 5,
        m_potentials: int = 10,
        epsilon: float = 1.0,
    ):
        r"""
        :param int x_dim: Dimension of X space, defaults to 2
        :param int y_dim: Dimension of Y space, defaults to 3
        :param int n_potentials: Number of potentials for approximating dual variable :math:`f(y)=\varepsilon\log\sum_{n=1}^N w_n \mathcal{N}(y\vert a_n, A_n/\varepsilon)`, defaults to 5
        :param int m_potentials: Number of potentials for approximating plan :math:`c(x, y)=-\varepsilon\log\sum_{m=1}^M v_m(x) \exp(\langle b_m(x), y \rangle) /\varepsilon`, defaults to 10
        :param float epsilon: Regularization parameter, defaults to 1.0
        """
        super(MLPCost, self).__init__(x_dim, y_dim)
        self.n_potentials = n_potentials
        self.m_potentials = m_potentials
        self.register_buffer("epsilon", torch.tensor(epsilon))

        self.log_v_m = nn.Sequential(
            torchvision.ops.MLP(in_channels=x_dim, hidden_channels=[m_potentials], activation_layer=torch.nn.ReLU),
            nn.LogSoftmax(dim=-1),
        )
        self.b_m = torchvision.ops.MLP(
            in_channels=x_dim, hidden_channels=[m_potentials * y_dim], activation_layer=torch.nn.ReLU
        )

    def func(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # -> [1]
        log_v_m = self.log_v_m(x)
        b_m = self.b_m(x).reshape(self.m_potentials, self.y_dim)

        # sum([M x y_dim] * [1 x y_dim], dim=1) = [M]
        bT_y = torch.sum(b_m * y[None, :], dim=1)

        # sum([[M] + [M], dim=0) = [1]
        return -self.epsilon * torch.logsumexp(log_v_m + bT_y / self.epsilon, dim=0)


class TriangularMLP(nn.Module):
    def __init__(
        self, in_dim, out_dim, n_min_neurons=100, n_steps=2, activation_gen=lambda: nn.ReLU(True), use_dropout=False
    ):

        super().__init__()
        net_list = []
        curr_dim = in_dim

        for i in range(1, n_steps + 1):
            next_dim = max(n_min_neurons, (2**i) * in_dim)
            net_list.append(nn.Linear(curr_dim, next_dim))
            net_list.append(activation_gen())
            if use_dropout:
                net_list.append(nn.Dropout(0.005))
            curr_dim = next_dim

        for i in range(n_steps - 1, 0, -1):
            next_dim = max(n_min_neurons, (2**i) * in_dim)
            net_list.append(nn.Linear(curr_dim, next_dim))
            net_list.append(activation_gen())
            if use_dropout:
                net_list.append(nn.Dropout(0.005))
            curr_dim = next_dim

        net_list.append(nn.Linear(curr_dim, out_dim))
        self.net = nn.Sequential(*net_list)

    def forward(self, x):
        return self.net(x)


class WeakMover(TriangularMLP):

    def __init__(
        self,
        in_dim=4,
        out_dim=2,
        n_min_neurons=100,
        n_steps=2,
        activation_gen=lambda: nn.ReLU(True),
        use_dropout=False,
    ):

        super().__init__(in_dim, out_dim, n_min_neurons, n_steps, activation_gen, use_dropout)


class Mover(TriangularMLP):

    def __init__(self, dim=2, n_min_neurons=100, n_steps=2, activation_gen=lambda: nn.ReLU(True), use_dropout=False):
        super().__init__(dim, dim, n_min_neurons, n_steps, activation_gen, use_dropout)


class Discriminator(TriangularMLP):

    def __init__(self, in_dim=2, n_min_neurons=100, n_steps=4, activation_gen=lambda: nn.ReLU(True)):
        super().__init__(in_dim, 1, n_min_neurons, n_steps, activation_gen, False)


class FullyConnectedMLP(nn.Module):
    def __init__(self, input_dim, hiddens, output_dim, activation_gen=lambda: nn.ReLU(), sn_iters=0):

        def _SN(module):
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


class ResNet_G(nn.Module):
    "Generator ResNet architecture from https://github.com/harryliew/WGAN-QC"

    def __init__(self, z_dim, size, nc=3, nfilter=64, nfilter_max=512, bn=True, res_ratio=0.1, **kwargs):
        super().__init__()
        s0 = self.s0 = 4
        nf = self.nf = nfilter
        nf_max = self.nf_max = nfilter_max
        self.bn = bn
        self.z_dim = z_dim
        self.nc = nc

        # Submodules
        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2 ** (nlayers + 1))

        self.fc = nn.Linear(z_dim, self.nf0 * s0 * s0)
        if self.bn:
            self.bn1d = nn.BatchNorm1d(self.nf0 * s0 * s0)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        blocks = []
        for i in range(nlayers, 0, -1):
            nf0 = min(nf * 2 ** (i + 1), nf_max)
            nf1 = min(nf * 2**i, nf_max)
            blocks += [
                ResNetBlock(nf0, nf1, bn=self.bn, res_ratio=res_ratio),
                ResNetBlock(nf1, nf1, bn=self.bn, res_ratio=res_ratio),
                nn.Upsample(scale_factor=2),
            ]

        nf0 = min(nf * 2, nf_max)
        nf1 = min(nf, nf_max)
        blocks += [
            ResNetBlock(nf0, nf1, bn=self.bn, res_ratio=res_ratio),
            ResNetBlock(nf1, nf1, bn=self.bn, res_ratio=res_ratio),
        ]

        self.resnet = nn.Sequential(*blocks)
        self.conv_img = nn.Conv2d(nf, nc, 3, padding=1)

    def forward(self, z):
        batch_size = z.size(0)
        z = z.view(batch_size, -1)
        out = self.fc(z)
        if self.bn:
            out = self.bn1d(out)
        out = self.relu(out)
        out = out.view(batch_size, self.nf0, self.s0, self.s0)

        out = self.resnet(out)

        out = self.conv_img(out)
        out = torch.tanh(out)

        return out


class ResNet_D(nn.Module):
    "Discriminator ResNet architecture from https://github.com/harryliew/WGAN-QC"

    def __init__(self, size=64, nc=3, nfilter=64, nfilter_max=512, res_ratio=0.1):
        super().__init__()
        s0 = self.s0 = 4
        nf = self.nf = nfilter
        nf_max = self.nf_max = nfilter_max
        self.nc = nc

        # Submodules
        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2**nlayers)

        nf0 = min(nf, nf_max)
        nf1 = min(nf * 2, nf_max)
        blocks = [
            ResNetBlock(nf0, nf0, bn=False, res_ratio=res_ratio),
            ResNetBlock(nf0, nf1, bn=False, res_ratio=res_ratio),
        ]

        for i in range(1, nlayers + 1):
            nf0 = min(nf * 2**i, nf_max)
            nf1 = min(nf * 2 ** (i + 1), nf_max)
            blocks += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResNetBlock(nf0, nf0, bn=False, res_ratio=res_ratio),
                ResNetBlock(nf0, nf1, bn=False, res_ratio=res_ratio),
            ]

        self.conv_img = nn.Conv2d(nc, 1 * nf, 3, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.resnet = nn.Sequential(*blocks)
        self.fc = nn.Linear(self.nf0 * s0 * s0, 1)

    def forward(self, x):
        batch_size = x.size(0)

        out = self.relu((self.conv_img(x)))
        out = self.resnet(out)
        out = out.view(batch_size, self.nf0 * self.s0 * self.s0)
        out = self.fc(out)

        return out


class ResNetBlock(nn.Module):
    def __init__(self, fin, fout, fhidden=None, bn=True, res_ratio=0.1):
        super().__init__()
        # Attributes
        self.bn = bn
        self.is_bias = not bn
        self.learned_shortcut = fin != fout
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden
        self.res_ratio = res_ratio

        # Submodules
        self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1, bias=self.is_bias)
        if self.bn:
            self.bn2d_0 = nn.BatchNorm2d(self.fhidden)
        self.conv_1 = nn.Conv2d(self.fhidden, self.fout, 3, stride=1, padding=1, bias=self.is_bias)
        if self.bn:
            self.bn2d_1 = nn.BatchNorm2d(self.fout)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(self.fin, self.fout, 1, stride=1, padding=0, bias=False)
            if self.bn:
                self.bn2d_s = nn.BatchNorm2d(self.fout)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(x)
        if self.bn:
            dx = self.bn2d_0(dx)
        dx = self.relu(dx)
        dx = self.conv_1(dx)
        if self.bn:
            dx = self.bn2d_1(dx)
        out = self.relu(x_s + self.res_ratio * dx)
        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
            if self.bn:
                x_s = self.bn2d_s(x_s)
        else:
            x_s = x
        return x_s

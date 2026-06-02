import gc
import itertools
import os
import random
import sys
from functools import partial
from typing import Dict, List, Tuple

import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from nflows.distributions.normal import ConditionalDiagonalNormal, StandardNormal
from nflows.flows.base import Flow
from nflows.nn import nets as nets
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.base import CompositeTransform
from nflows.transforms.coupling import (
    AdditiveCouplingTransform,
    AffineCouplingTransform,
)
from nflows.transforms.normalization import BatchNorm
from nflows.transforms.permutations import RandomPermutation, ReversePermutation
from torch import optim
from tqdm import tqdm


class ConditionalRealNVP(Flow):
    def __init__(
        self,
        features,
        hidden_features,
        context_features=None,
        hidden_context_features=None,
        num_layers=5,
        num_blocks_per_layer=2,
        use_volume_preserving=False,
        activation=F.relu,
        dropout_probability=0.0,
        batch_norm_within_layers=False,
        batch_norm_between_layers=False,
    ):

        if use_volume_preserving:
            coupling_constructor = AdditiveCouplingTransform
        else:
            coupling_constructor = AffineCouplingTransform

        mask = torch.ones(features)
        mask[::2] = -1

        def create_resnet(in_features, out_features):
            return nets.ResidualNet(
                in_features,
                out_features,
                hidden_features=hidden_features,
                context_features=context_features,
                num_blocks=num_blocks_per_layer,
                activation=activation,
                dropout_probability=dropout_probability,
                use_batch_norm=batch_norm_within_layers,
            )

        layers = []
        for _ in range(num_layers):
            transform = coupling_constructor(mask=mask, transform_net_create_fn=create_resnet)
            layers.append(transform)
            mask *= -1
            if batch_norm_between_layers:
                layers.append(BatchNorm(features=features))

        context_encoder = nn.Sequential(
            nn.Linear(context_features, hidden_context_features),
            nn.ReLU(True),
            *itertools.chain.from_iterable(
                (
                    nn.Linear(hidden_context_features, hidden_context_features),
                    nn.LeakyReLU(),
                )
                for _ in range(num_layers)
            ),
            nn.Linear(hidden_context_features, 2 * features)
        )
        distribution = ConditionalDiagonalNormal(shape=(features,), context_encoder=context_encoder)
        super().__init__(
            transform=CompositeTransform(layers),
            distribution=distribution,
        )


class ConditionalMaskedAutoregressiveFlow(Flow):
    def __init__(
        self,
        features,
        hidden_features,
        context_features=None,
        hidden_context_features=None,
        num_layers=5,
        num_blocks_per_layer=2,
        use_residual_blocks=True,
        use_random_masks=False,
        use_random_permutations=False,
        activation=F.relu,
        dropout_probability=0.0,
        batch_norm_within_layers=False,
        batch_norm_between_layers=False,
    ):

        if use_random_permutations:
            permutation_constructor = RandomPermutation
        else:
            permutation_constructor = ReversePermutation

        layers = []
        for _ in range(num_layers):
            layers.append(permutation_constructor(features))
            layers.append(
                MaskedAffineAutoregressiveTransform(
                    features=features,
                    hidden_features=hidden_features,
                    context_features=context_features,
                    num_blocks=num_blocks_per_layer,
                    use_residual_blocks=use_residual_blocks,
                    random_mask=use_random_masks,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=batch_norm_within_layers,
                )
            )
            if batch_norm_between_layers:
                layers.append(BatchNorm(features))

        context_encoder = nn.Sequential(
            nn.Linear(context_features, hidden_context_features),
            nn.ReLU(True),
            *itertools.chain.from_iterable(
                (
                    nn.Linear(hidden_context_features, hidden_context_features),
                    nn.LeakyReLU(),
                )
                for _ in range(num_layers)
            ),
            nn.Linear(hidden_context_features, 2 * features)
        )
        distribution = ConditionalDiagonalNormal(shape=(features,), context_encoder=context_encoder)
        super().__init__(
            transform=CompositeTransform(layers),
            distribution=distribution,
        )


class compat_patch:
    def __init__(self, flow: ConditionalRealNVP):
        self.flow = flow

    def __call__(self, x, X_DIM=2):
        # x.shape: batch, DIM+ZD
        x = x[:, :X_DIM]
        samples = self.flow.sample(num_samples=1, context=x)
        samples = samples.squeeze(1)
        return samples

    def parameters(self):
        return self.flow.parameters()

    def eval(self):
        return self.flow.eval()


class MLPnet(nn.Sequential):
    def __init__(self, input_size, hidden_size, num_hidden_layers):
        super().__init__(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(True),
            *itertools.chain.from_iterable(
                (nn.Linear(hidden_size, hidden_size), nn.ReLU(True)) for _ in range(num_hidden_layers)
            ),
            nn.Linear(hidden_size, input_size)
        )


class MyCGenerator(nn.Module):
    def __init__(
        self,
        x_dim=2,
        t_dim=2,
        n_t=4,
        z_dim=1,
        out_dim=2,
        layers=[128, 128, 128],
        active=partial(nn.LeakyReLU, 0.2),
    ):
        super().__init__()

        self.x_dim = x_dim
        self.t_dim = t_dim
        self.z_dim = z_dim

        self.model = []
        ch_prev = x_dim + t_dim + z_dim

        self.t_transform = nn.Embedding(
            n_t,
            t_dim,
        )

        for ch_next in layers:
            self.model.append(nn.Linear(ch_prev, ch_next))
            self.model.append(active())
            ch_prev = ch_next

        self.model.append(nn.Linear(ch_prev, out_dim))
        self.model = nn.Sequential(*self.model)

    def forward(self, x, t, z):
        batch_size = x.shape[0]

        if z.shape != (batch_size, self.z_dim):
            z = z.reshape((batch_size, self.z_dim))

        return self.model(
            torch.cat(
                [
                    x,
                    self.t_transform(t),
                    z,
                ],
                dim=1,
            )
        )


class MyCDiscriminator(nn.Module):
    def __init__(
        self,
        x_dim=2,
        t_dim=2,
        n_t=4,
        layers=[128, 128, 128],
        active=partial(nn.LeakyReLU, 0.2),
    ):
        super().__init__()

        self.x_dim = x_dim
        self.t_dim = t_dim

        self.model = []
        ch_prev = 2 * x_dim + t_dim

        self.t_transform = nn.Embedding(
            n_t,
            t_dim,
        )

        for ch_next in layers:
            self.model.append(nn.Linear(ch_prev, ch_next))
            self.model.append(active())
            ch_prev = ch_next

        self.model.append(nn.Linear(ch_prev, 1))
        self.model = nn.Sequential(*self.model)

    def forward(
        self,
        x_t,
        t,
        x_tp1,
    ):
        return self.model(
            torch.cat(
                [
                    x_t,
                    self.t_transform(t),
                    x_tp1,
                ],
                dim=1,
            )
        ).squeeze()


class MyGenerator(nn.Module):
    def __init__(
        self,
        x_dim=2,
        z_dim=1,
        out_dim=2,
        layers=[128, 128, 128],
        active=partial(nn.LeakyReLU, 0.2),
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

    def forward(self, x, z):
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


class MyDiscriminator(nn.Module):
    def __init__(
        self,
        x_dim=2,
        layers=[128, 128, 128],
        active=partial(nn.LeakyReLU, 0.2),
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

    def forward(self, x):
        return self.model(x).squeeze()

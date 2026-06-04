import itertools
from typing import Callable, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from nflows.distributions.normal import ConditionalDiagonalNormal
from nflows.flows.base import Flow
from nflows.nn import nets as nets
from nflows.transforms.base import CompositeTransform
from nflows.transforms.coupling import (
    AdditiveCouplingTransform,
    AffineCouplingTransform,
)
from nflows.transforms.normalization import BatchNorm


class ConditionalRealNVP(Flow):
    def __init__(
        self,
        features: int,
        hidden_features: list[int],
        context_features: int | None = None,
        hidden_context_features: list[int] | None = None,
        num_layers: int = 5,
        num_blocks_per_layer: int = 2,
        use_volume_preserving: bool = False,
        activation: Callable[[], nn.Module] = F.relu,
        dropout_probability: float = 0.0,
        batch_norm_within_layers: bool = False,
        batch_norm_between_layers: bool = False,
    ):

        if use_volume_preserving:
            coupling_constructor = AdditiveCouplingTransform
        else:
            coupling_constructor = AffineCouplingTransform

        mask = torch.ones(features)
        mask[::2] = -1

        def create_resnet(in_features: int, out_features: int) -> nn.Module:
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
            transform = coupling_constructor(
                mask=mask, transform_net_create_fn=create_resnet
            )
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
        distribution = ConditionalDiagonalNormal(
            shape=(features,), context_encoder=context_encoder
        )
        super().__init__(
            transform=CompositeTransform(layers),
            distribution=distribution,
        )


class compat_patch:
    def __init__(self, flow: Flow):
        self.flow = flow

    def __call__(self, x: torch.Tensor, X_DIM: int = 2) -> torch.Tensor:
        # x.shape: batch, DIM+ZD
        x = x[:, :X_DIM]
        samples = self.flow.sample(num_samples=1, context=x)
        samples = samples.squeeze(1)
        return samples

    def parameters(self) -> Iterator[torch.Tensor]:
        return self.flow.parameters()

    def eval(self):
        return self.flow.eval()

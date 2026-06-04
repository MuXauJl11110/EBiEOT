import itertools
from typing import Callable

import torch.nn as nn
import torch.nn.functional as F
from nflows.distributions.normal import ConditionalDiagonalNormal
from nflows.flows.base import Flow
from nflows.nn import nets as nets
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.base import CompositeTransform
from nflows.transforms.normalization import BatchNorm
from nflows.transforms.permutations import RandomPermutation, ReversePermutation


class ConditionalMaskedAutoregressiveFlow(Flow):
    def __init__(
        self,
        features: int,
        hidden_features: list[int],
        context_features: int | None = None,
        hidden_context_features: list[int] | None = None,
        num_layers: int = 5,
        num_blocks_per_layer: int = 2,
        use_residual_blocks: bool = True,
        use_random_masks: bool = False,
        use_random_permutations: bool = False,
        activation: Callable[[], nn.Module] = F.relu,
        dropout_probability: float = 0.0,
        batch_norm_within_layers: bool = False,
        batch_norm_between_layers: bool = False,
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
        distribution = ConditionalDiagonalNormal(
            shape=(features,), context_encoder=context_encoder
        )
        super().__init__(
            transform=CompositeTransform(layers),
            distribution=distribution,
        )

import os
from collections.abc import Callable
from functools import partial
from typing import Literal

import torch
from src.ebieot.base import BaseEBiEOT
from src.ebieot.costs.base import BaseCost
from src.ebieot.potentials.base import BasePotential
from src.ebieot.sampling.langevin import sample_langevin_batch
from src.ebieot.sampling.sample_buffer import SampleBuffer
from src.utils.samplers.base import Sampler


# The code of this class is based on https://github.com/PetrMokrov/Energy-guided-Entropic-OT/tree/main
class EbieotNn(BaseEBiEOT):
    """EBiEOT-NN: neural cost and potential with Langevin / pseudo sampling."""

    def __init__(
        self,
        potential: BasePotential,
        cost: BaseCost,
        sample_buffer: SampleBuffer,
        *,
        epsilon: float = 1.0,
        alpha: float = 0.0,
        reference_data_noise_sigma: float = 0.0,
        sampler_type: Literal["langevin", "pseudo"] = "pseudo",
        step_size: float = 0.05,
        noise: float = 0.005,
        num_iterations: int = 100,
        decay: float = 1.0,
        thresh: float | None = None,
        grad_proj_type: Literal["value", "norm", "none"] = "none",
        norm_thresh: float = 1.0,
        value_thresh: float = 0.01,
        data_projector: Callable[[torch.Tensor], torch.Tensor] | None = None,
        projection_min: float = 0.0,
        projection_max: float = 1.0,
        is_projected: bool = False,
        compute_stats: bool = False,
    ) -> None:
        super().__init__()
        self.potential = potential
        self.cost = cost
        self.sample_buffer = sample_buffer
        self.register_buffer("epsilon", torch.tensor(epsilon))
        self.alpha = alpha
        self.reference_data_noise_sigma = reference_data_noise_sigma
        self.sampler_type = sampler_type
        self.step_size = step_size
        self.noise = noise
        self.num_iterations = num_iterations
        self.decay = decay
        self.thresh = thresh
        self.grad_proj_type = grad_proj_type
        self.norm_thresh = norm_thresh
        self.value_thresh = value_thresh
        self.projection_min = projection_min
        self.projection_max = projection_max
        self.is_projected = is_projected
        self.compute_stats = compute_stats

        if data_projector is None:
            if is_projected:
                lo, hi = projection_min, projection_max
                data_projector = lambda x: x.clamp_(lo, hi)
            else:
                data_projector = lambda x: x
        self.data_projector = data_projector

    def negative_energy_function(
        self, batched_x: torch.Tensor, batched_y: torch.Tensor
    ) -> torch.Tensor:  # -> [bs]
        return (
            -(self.cost(batched_x, batched_y) - self.potential(batched_y))
            / self.epsilon
        )

    def negative_energy_function_grad_y(
        self, batched_x: torch.Tensor, batched_y: torch.Tensor, stats: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # -> [bs]
        cost_part = self.cost.grad_y(batched_x, batched_y) / self.epsilon
        potential_part = self.potential.grad_y(batched_y) / self.epsilon
        if stats:
            return -(cost_part - potential_part), cost_part, potential_part
        return -(cost_part - potential_part)

    def get_samples_energy(
        self,
        batched_x: torch.Tensor,
        batched_init_y: torch.Tensor,
        compute_stats: bool = False,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        return sample_langevin_batch(
            score_function=partial(self.negative_energy_function_grad_y, batched_x),
            y=batched_init_y,
            sampler_type=self.sampler_type,
            step_size=self.step_size,
            noise=self.noise,
            num_iterations=self.num_iterations,
            decay=self.decay,
            data_projector=self.data_projector,
            compute_stats=compute_stats,
            grad_proj_type=self.grad_proj_type,
            norm_thresh=self.norm_thresh,
            value_thresh=self.value_thresh,
            thresh=self.thresh,
        )

    def sample(
        self,
        x_samples: torch.Tensor,
        init_y_samples: torch.Tensor | None = None,
        init_sampler: Sampler | None = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            if init_y_samples is not None:  # sample from initial distribution
                y_samples = init_y_samples
            else:
                if init_sampler is None:  # sample from Normal
                    y_samples = self.sample_buffer.noise_gen.sample(
                        (x_samples.size(0),)
                    ).to(x_samples)
                    # y_samples = torch.randn_like(x_samples) * init_sigma
                else:  # sample from Sampler
                    y_samples = init_sampler.sample(x_samples.size(0)).to(x_samples)
            output_samples, _ = self.get_samples_energy(x_samples, y_samples)

            return output_samples

    def forward(self, x_samples: torch.Tensor) -> torch.Tensor:  # -> [bs]
        with torch.no_grad():
            y_samples = self.sample_buffer.noise_gen.sample((x_samples.size(0),)).to(
                x_samples
            )
            output_samples, _ = self.get_samples_energy(x_samples, y_samples)

            return output_samples

    def compute_unpaired_loss(
        self, X_unpaired: torch.Tensor, Y_unpaired: torch.Tensor
    ) -> dict[str, torch.Tensor | float]:
        """
        You can find details about training at https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial8/Deep_Energy_Models.html.
        """
        # slightly noise the data
        if self.reference_data_noise_sigma > 0.0:
            Y_unpaired += self.reference_data_noise_sigma * torch.randn_like(Y_unpaired)

        x_samples, y_samples_0, indices = self.sample_buffer(X_unpaired)

        with torch.no_grad():
            Y_sampled, stats = self.get_samples_energy(
                x_samples, y_samples_0, compute_stats=self.compute_stats
            )
            output = dict(stats) if self.compute_stats else {}

        self.sample_buffer.push(x_samples, Y_sampled, indices)
        pos_out = self.potential.forward(Y_unpaired)
        pos_out_mean = pos_out.mean()
        neg_out = self.negative_energy_function(x_samples, Y_sampled)
        neg_out_mean = neg_out.mean()
        loss = -(pos_out_mean - neg_out_mean)  # we maximize this loss
        loss += self.alpha * (pos_out.pow(2) + neg_out.pow(2)).mean()

        return output | {
            "loss": loss,
            "int_potential": pos_out_mean,
            "int_log_Z": neg_out_mean,
        }

    def compute_paired_loss(
        self, X_paired: torch.Tensor, Y_paired: torch.Tensor
    ) -> dict[str, torch.Tensor | float]:
        output = {}
        cost = self.cost.forward(X_paired, Y_paired)
        loss = cost.mean()
        loss += self.alpha * (cost.pow(2)).mean()

        return output | {"loss": loss}

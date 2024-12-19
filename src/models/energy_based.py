import os

import torch
from configs.energy_based.model import EBMConfig
from configs.energy_based.sampling import LangevinConfig, PseudoLangevinConfig

from src.costs.base import BaseCost
from src.models.base import BaseGenerativeModel
from src.potentials.base import BasePotential
from src.samplers.base import Sampler
from src.samplers.energy_based.langevin import (
    sample_langevin_batch,
    sample_pseudo_langevin_batch,
)
from src.samplers.energy_based.sample_buffer import SampleBuffer


# The code of this class is based on https://github.com/PetrMokrov/Energy-guided-Entropic-OT/tree/main
class EGEOT(BaseGenerativeModel):
    """
    Energy-guided entropic optimal transport (EOT) with general cost function class
    """

    def __init__(self, potential: BasePotential, cost: BaseCost, sample_buffer: SampleBuffer, config: EBMConfig):
        super().__init__()
        self.potential = potential
        self.cost = cost
        self.sample_buffer = sample_buffer
        self.config = config

    def negative_energy_function(self, batched_x: torch.Tensor, batched_y: torch.Tensor) -> torch.Tensor:  # -> [bs]
        return -(self.cost(batched_x, batched_y) - self.potential(batched_y)) / self.config.epsilon

    def negative_energy_function_grad_y(
        self, batched_x: torch.Tensor, batched_y: torch.Tensor, stats: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # -> [bs]
        cost_part = self.cost.grad_y(batched_x, batched_y) / self.config.epsilon
        potential_part = self.potential.grad_y(batched_y) / self.config.epsilon
        if stats:
            return -(cost_part - potential_part), cost_part, potential_part
        return -(cost_part - potential_part)

    def get_samples_energy(
        self,
        batched_x: torch.Tensor,
        batched_init_y: torch.Tensor,
        compute_stats: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor:
        if isinstance(self.config.sampling, LangevinConfig):
            sample_function = sample_langevin_batch
        elif isinstance(self.config.sampling, PseudoLangevinConfig):
            sample_function = sample_pseudo_langevin_batch
        else:
            raise ValueError("Unknown sampling!")
        sampling_config = self.config.sampling

        def score_function(y: torch.Tensor, stats: bool = False):
            return self.negative_energy_function_grad_y(batched_x, y, stats=stats)

        return sample_function(
            score_function=score_function,
            y=batched_init_y,
            step_size=sampling_config.step_size,
            noise=sampling_config.noise,
            num_iterations=sampling_config.num_iterations,
            decay=sampling_config.decay,
            thresh=sampling_config.thresh,
            data_projector=sampling_config.projection.data_projector,
            compute_stats=compute_stats,
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
                    y_samples = self.sample_buffer.noise_gen.sample((x_samples.size(0),)).to(x_samples)
                    # y_samples = torch.randn_like(x_samples) * init_sigma
                else:  # sample from Sampler
                    y_samples = init_sampler.sample(x_samples.size(0)).to(x_samples)
            output_samples = self.get_samples_energy(x_samples, y_samples)

            return output_samples

    def store(self, path: str) -> None:
        directory_path = os.path.dirname(path)
        os.makedirs(directory_path, exist_ok=True)

        torch.save(
            {
                "potential_state_dict": self.potential.state_dict(),
                "cost_state_dict": self.cost.state_dict(),
                "config_dict": self.config.model_dump(),
            },
            path,
        )

    def forward(self, x_samples: torch.Tensor) -> torch.Tensor:  # -> [bs]
        with torch.no_grad():
            y_samples = self.sample_buffer.noise_gen.sample((x_samples.size(0),)).to(x_samples)
            output_samples = self.get_samples_energy(x_samples, y_samples)

            return output_samples

    def compute_unpaired_loss(
        self, X_unpaired: torch.Tensor, Y_unpaired: torch.Tensor, compute_stats: bool = False
    ) -> dict[str, torch.Tensor]:
        """
        You can find details about training at https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial8/Deep_Energy_Models.html.
        """
        # slightly noise the data
        if self.config.reference_data_noise_sigma > 0.0:
            Y_unpaired += self.config.reference_data_noise_sigma * torch.randn_like(Y_unpaired)

        x_samples, y_samples_0, indices = self.sample_buffer(X_unpaired)

        with torch.no_grad():
            if compute_stats:
                Y_sampled, neg_energy_t, cost_t, potential_t, noise_norm = self.get_samples_energy(
                    x_samples, y_samples_0, compute_stats=compute_stats
                )
                output = {
                    "neg_energy_t": neg_energy_t,
                    "cost_t": cost_t,
                    "potential_t": potential_t,
                    "noise": noise_norm,
                }
            else:
                Y_sampled = self.get_samples_energy(x_samples, y_samples_0, compute_stats=compute_stats)
                output = {}

        self.sample_buffer.push(x_samples, Y_sampled, indices)
        pos_out = self.potential.forward(Y_unpaired)
        pos_out_mean = pos_out.mean()
        neg_out = self.negative_energy_function(x_samples, Y_sampled)
        neg_out_mean = neg_out.mean()
        loss = -(pos_out_mean - neg_out_mean)  # we maximize this loss
        loss += self.config.alpha * (pos_out.pow(2) + neg_out.pow(2)).mean()

        return output | {"loss": loss, "int_potential": pos_out_mean, "int_log_Z": neg_out_mean}

    def compute_paired_loss(self, X_paired: torch.Tensor, Y_paired: torch.Tensor) -> dict[str, torch.Tensor]:  # -> [1]
        output = {}
        cost = self.cost.forward(X_paired, Y_paired)
        loss = cost.mean()
        loss += self.config.alpha * (cost.pow(2)).mean()

        return output | {"loss": loss}

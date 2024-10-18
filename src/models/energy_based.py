import inspect
import os

import torch
import torch.nn as nn

from src.configs.energy_based.model import EBMConfig
from src.costs.base import BaseCost
from src.models.base import BaseModel
from src.samplers.base import Sampler
from src.samplers.energy_based.sample_buffer import SampleBuffer
from src.utils.energy_based import computePotGrad, evaluating


# The code of this class is based on https://github.com/PetrMokrov/Energy-guided-Entropic-OT/tree/main
class EGEOT(BaseModel):
    """
    Energy-guided entropic optimal transport (EOT) with general cost function class
    """

    def __init__(self, potential: nn.Module, cost: BaseCost, sample_buffer: SampleBuffer, config: EBMConfig):
        super(EGEOT, self).__init__()
        self.potential = potential
        self.cost = cost
        self.sample_buffer = sample_buffer
        self.config = config

    def cond_score(
        self, y: torch.Tensor, x: torch.Tensor, ret_stats: bool = False
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor
    ):  # -> ([bs x y_dim], [bs x y_dim], [bs x y_dim]) | [bs x y_dim]
        with torch.enable_grad():
            y.requires_grad_(True)
            proto_s = self.potential.forward(y)
            score = computePotGrad(y, proto_s)
            assert score.shape == y.shape  # [bs x y_dim]

        cost_coeff = (1 / self.config.epsilon) * self.config.sampling.cost_coefficient / self.config.sampling.step_size
        cost_part = self.cost.grad_y(x, y) * cost_coeff  # [bs x y_dim]
        score_part = score * self.config.sampling.score_coefficient  # [bs x y_dim]

        if not ret_stats:
            return score_part - cost_part
        return score_part - cost_part, cost_part, score_part

    def get_samples_energy(
        self,
        x_samples: torch.Tensor,
        init_y_samples: torch.Tensor,
        compute_stats: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor:

        def score_function(y, ret_stats=False):
            return self.cond_score(y, x_samples, ret_stats=ret_stats)

        sample_function = self.config.sampling.function
        signature = inspect.signature(sample_function)
        valid_args = signature.parameters
        filtered_args = {k: v for k, v in iter(self.config.sampling) if k in valid_args}

        return sample_function(
            score_function=score_function,
            y=init_y_samples,
            data_projector=self.config.projection.data_projector,
            compute_stats=compute_stats,
            **filtered_args,
        )

    # TODO: why this function is so universal for sampling?
    # WIP: current function takes arguments from config for Langevin sampling
    def sample(
        self,
        x_samples: torch.Tensor,
        init_y_samples: torch.Tensor | None = None,
        init_sampler: Sampler | None = None,
    ) -> torch.Tensor:
        with evaluating(self.potential), evaluating(self.cost):
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
        with evaluating(self.potential):
            with torch.no_grad():
                y_samples = self.sample_buffer.noise_gen.sample((x_samples.size(0),)).to(x_samples)
                output_samples = self.get_samples_energy(x_samples, y_samples)

                return output_samples

    def compute_unpaired_loss(self, X_unpaired: torch.Tensor, Y_unpaired: torch.Tensor) -> dict[str, torch.Tensor]:
        # slightly noise the data
        if self.config.reference_data_noise_sigma > 0.0:
            Y_unpaired += self.config.reference_data_noise_sigma * torch.randn_like(Y_unpaired)

        x_samples, neg_y_samples_0, indices = self.sample_buffer(X_unpaired)

        with evaluating(self.potential), evaluating(self.cost):
            with torch.no_grad():
                neg_y_samples, r_t, cost_r_t, score_r_t, noise_norm = self.get_samples_energy(
                    x_samples, neg_y_samples_0, compute_stats=True
                )

        self.sample_buffer.push(x_samples, neg_y_samples, indices)
        pos_out = self.potential.forward(Y_unpaired)
        pos_out_mean = pos_out.mean()
        neg_out = self.potential.forward(neg_y_samples)
        neg_out_mean = neg_out.mean()
        loss = -pos_out_mean + neg_out_mean
        loss += self.config.alpha * (pos_out.pow(2) + neg_out.pow(2)).mean()
        # self.sample_buffer.push(x_samples, neg_y_samples, indices)
        return {
            "pos_out": pos_out_mean,
            "neg_out": neg_out_mean,
            "r_t": r_t,
            "cost_r_t": cost_r_t,
            "score_r_t": score_r_t,
            "noise": noise_norm,
            "loss": loss,
        }

    def compute_paired_loss(self, X_paired: torch.Tensor, Y_paired: torch.Tensor) -> torch.Tensor:  # -> [1]
        c = self.cost(X_paired, Y_paired)

        return c.mean()

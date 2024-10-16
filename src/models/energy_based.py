import os
from abc import ABC, abstractmethod
from typing import Callable, Literal

import torch
import torch.nn as nn
from pydantic import BaseModel, model_validator

from samplers.energy_based.base import SampleBuffer
from src.samplers.base import Sampler
from src.utils.energy_based import computePotGrad, evaluating
from src.utils.langevin import sample_langevin_batch, sample_pseudo_langevin_batch


class LangevinConfig(BaseModel):
    thresh: float | None = None
    noise: float = 0.05
    step_size: float = 0.05
    decay: float = 1.0
    score_coefficient: float = 1.0
    num_iterations: int = 10
    cost_coefficient = None

    # Init cost_coefficients = sampling_noise^2
    @model_validator(pre=True)
    def set_cost_coefficient(cls, values):
        if "cost_coefficient" not in values or values["cost_coefficient"] is None:
            values["cost_coefficient"] = values["sampling_noise"] ** 2
            return values


class PseudoLangevinConfig(LangevinConfig):
    grad_proj_type: Literal["value", "norm", "none"] = "none"
    norm_thresh: float = 1.0
    value_thresh: float = 0.01
    noise: float = 0.005


class ProjectionDataConfig(BaseModel):
    min: float = 0.0
    max: float = 1.0
    is_projected: bool = False
    data_projector: Callable[[torch.Tensor], torch.Tensor] | None = None

    @model_validator(pre=True)
    def set_data_projector(cls, values):
        if "data_projector" not in values or values["data_projector"] is None:
            if values["is_projected"] == False:
                values["data_projector"] = lambda x: x
            else:
                values["data_projector"] = lambda x: x.clamp_(values["min"], values["max"])
            return values


# make binding between method and config
class EBMConfig(BaseModel):
    sampling: LangevinConfig | PseudoLangevinConfig = LangevinConfig()
    projection: ProjectionDataConfig = ProjectionDataConfig()


# The code of this class is based on https://github.com/PetrMokrov/Energy-guided-Entropic-OT/tree/main
class EGEOTBase(nn.Module, ABC):
    """
    EGEOT with general cost function generic class
    """

    def __init__(self, sample_buffer: SampleBuffer, config: EBMConfig, *args, **kwargs):
        self.sample_buffer = sample_buffer
        self.config = config
        super().__init__(*args, config=config, **kwargs)

    @abstractmethod
    def cost_grad_y(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:  # -> [bs x y_dim]
        """
        returns \nabla_y c(x, y)
        """
        raise NotImplementedError()

    def cond_score(
        self, y: torch.Tensor, x: torch.Tensor, ret_stats: bool = False
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor
    ):  # -> ([bs x y_dim], [bs x y_dim], [bs x y_dim]) | [bs x y_dim]
        with torch.enable_grad():
            y.requires_grad_(True)
            proto_s = self.forward(y)
            score = computePotGrad(y, proto_s)
            assert score.shape == y.shape  # [bs x y_dim]

        cost_coeff = self.config.sampling.cost_coefficient / self.config.sampling.step_size
        cost_part = self.cost_grad_y(y, x) * cost_coeff  # [bs x y_dim]
        score_part = score * self.config.sampling.score_coefficient  # [bs x y_dim]

        if not ret_stats:
            return score_part - cost_part
        return score_part - cost_part, cost_part, score_part

    def get_samples_energy(
        self,
        init_y_samples: torch.Tensor,
        x_samples: torch.Tensor,
        decay=1.0,
        compute_stats=False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:

        def score(y, ret_stats=False):
            return self.cond_score(y, x_samples, ret_stats=ret_stats)

        if isinstance(self.config.sampling, LangevinConfig):
            return sample_langevin_batch(
                score_function=score,
                y=init_y_samples,
                step_size=self.config.sampling.step_size,
                noise=self.sampling_config.noise,
                noise: float = 0.005,
                num_iterations: int = 100,
                decay: float = 1.0,
                thresh: float | None = None,
                data_projector: Callable[[torch.Tensor], torch.Tensor] = lambda x: x.clamp_(0.0, 1.0),
                compute_stats: bool = False,
            )
            return sample_langevin_batch(
                score,
                init_y_samples,
                eps=eps,
                n_steps=n_steps,
                decay=decay,
                thresh=self.config.LANGEVIN_THRESH,
                noise=self.config.LANGEVIN_SAMPLING_NOISE,
                data_projector=data_projector,
                compute_stats=compute_stats,
            )
        elif isinstance(self.config.sampling, PseudoLangevinConfig):
            return sample_pseudo_langevin_batch(
                score,
                init_y_samples,
                eps=eps,
                n_steps=n_steps,
                decay=decay,
                grad_proj_type=self.config.PSEUDO_LANGEVIN_GRAD_PROJ_TYPE,
                norm_thresh=self.config.PSEUDO_LANGEVIN_NORM_THRESH,
                value_thresh=self.config.PSEUDO_LANGEVIN_VALUE_THRESH,
                noise=self.config.PSEUDO_LANGEVIN_NOISE,
                data_projector=data_projector,
                compute_stats=compute_stats,
            )
        

    def loss(self, xy_samples):
        """
        x_samples : (bs, *shape)
        y_samples : (bs, *shape)
        """
        x_samples = xy_samples[0]
        pos_y_samples = xy_samples[1]

        # slightly noise the data
        if self.config.REFERENCE_DATA_NOISE_SIGMA > 0.0:
            pos_y_samples += self.config.REFERENCE_DATA_NOISE_SIGMA * torch.randn_like(pos_y_samples)

        x_samples, neg_y_samples_0, indices = self.sample_buffer(x_samples)

        with evaluating(self):
            with torch.no_grad():
                neg_y_samples, r_t, cost_r_t, score_r_t, noise_norm = self.get_samples_energy(
                    neg_y_samples_0,
                    x_samples,
                    self.config.ENERGY_SAMPLING_STEP,
                    self.config.ENERGY_SAMPLING_ITERATIONS,
                    decay=self.config.LANGEVIN_DECAY,
                    compute_stats=True,
                )

        self.sample_buffer.push(x_samples, neg_y_samples, indices)
        pos_out = self.forward(pos_y_samples)
        pos_out_mean = pos_out.mean()
        neg_out = self.forward(neg_y_samples)
        neg_out_mean = neg_out.mean()
        loss = -pos_out_mean + neg_out_mean
        loss += self.config.ALPHA * (pos_out.pow(2) + neg_out.pow(2)).mean()
        self.sample_buffer.push(x_samples, neg_y_samples, indices)
        return {
            "pos_out": pos_out_mean,
            "neg_out": neg_out_mean,
            "loss": loss,
            "r_t": r_t,
            "cost_r_t": cost_r_t,
            "score_r_t": score_r_t,
            "noise": noise_norm,
        }

    def sample(
        self,
        x_samples: torch.Tensor,
        y_init: torch.Tensor | None = None,
        num_iterations: int | None = None,
        step_size: float | None = None,
        decay: float | None = None,
        init_sigma: float = 1.0,
        init_sampler: Sampler | None = None,
    ) -> torch.Tensor:
        n_iterations = self.sampling_config.num_iterations if num_iterations is None else num_iterations
        step_size = self.config.ENERGY_SAMPLING_STEP if step_size is None else step_size
        decay = self.config.LANGEVIN_DECAY if decay is None else decay

        with evaluating(self):
            with torch.no_grad():
                # sample from initial distribution
                if y_init is not None:
                    z = y_init
                else:
                    if init_sampler is None:
                        z = torch.randn_like(x_samples) * init_sigma
                    else:
                        z = init_sampler.sample(x_samples.size(0)).to(x_samples)
                z = self.get_samples_energy(z, x_samples, eps=step_size, n_steps=n_iterations, decay=decay)
                assert isinstance(z, torch.Tensor)
                return z

    def store(self, path: str):
        directory_path = os.path.dirname(path)
        os.makedirs(directory_path, exist_ok=True)

        torch.save({"state_dict": self.state_dict(), "config_dict": self.sampling_config.model_dump()}, path)

    @staticmethod
    def load():
        raise NotImplementedError()


class EGEOTl2Sq(EGEOTBase):
    """
    EgEOT for squared l2 loss $0.5 \\Vert x - y \\Vert_2^2$
    """

    def cost_grad_y(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return y - x


class EGEOTNoCost(EGEOTBase):
    """
    EgEOT for zero cost (i.e. recovers simple energy based model)
    """

    def cost_grad_y(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.tensor(0.0).to(y.device)


# class EGEOTl2SqAmbient(EGEOTBase):
#     """ """

#     def __init__(self, latent2data_gen, sample_buffer, config, *args, **kwargs):
#         self.latent2data_gen = latent2data_gen
#         super().__init__(sample_buffer, config, *args, **kwargs)

#     def cost_grad_y(self, y, x):
#         with torch.enable_grad():
#             y.requires_grad_(True)
#             cost = 0.5 * torch.flatten(self.latent2data_gen(y) - x, start_dim=1).pow(2).sum(dim=1, keepdim=True)
#             assert cost.shape == torch.Size([y.size(0), 1])
#             res = computePotGrad(y, cost)
#         return res

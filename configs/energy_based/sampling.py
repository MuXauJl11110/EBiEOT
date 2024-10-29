from typing import Callable, Literal

from pydantic import BaseModel, model_validator

from src.samplers.energy_based.langevin import (
    sample_langevin_batch,
    sample_pseudo_langevin_batch,
)


class LangevinConfig(BaseModel):
    function: Callable = sample_langevin_batch
    thresh: float | None = None
    step_size: float = 0.05
    noise: float = 0.05
    num_iterations: int = 100
    decay: float = 1.0
    score_coefficient: float = 1.0
    cost_coefficient: float | None = None

    # Init cost_coefficients = sampling_noise^2
    @model_validator(mode="after")
    def set_cost_coefficient(self):
        self.cost_coefficient = self.noise**2
        return self


class PseudoLangevinConfig(LangevinConfig):
    function: Callable = sample_pseudo_langevin_batch
    grad_proj_type: Literal["value", "norm", "none"] = "none"
    norm_thresh: float = 1.0
    value_thresh: float = 0.01
    noise: float = 0.005

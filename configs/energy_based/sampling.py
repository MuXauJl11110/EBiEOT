from typing import Literal

from configs.energy_based.projection import ProjectionDataConfig
from pydantic import BaseModel, model_validator


class LangevinConfig(BaseModel):
    thresh: float | None = None
    step_size: float = 0.05
    noise: float = 1e-2
    num_iterations: int = 100
    decay: float = 1.0
    score_coefficient: float = 1.0
    cost_coefficient: float | None = None
    projection: ProjectionDataConfig = ProjectionDataConfig()

    # Init cost_coefficients = sampling_noise^2
    @model_validator(mode="after")
    def set_cost_coefficient(self):
        self.cost_coefficient = self.noise**2
        return self


class PseudoLangevinConfig(LangevinConfig):
    grad_proj_type: Literal["value", "norm", "none"] = "none"
    norm_thresh: float = 1.0
    value_thresh: float = 0.01
    noise: float = 0.005

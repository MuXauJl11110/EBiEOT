from typing import Callable

import torch
from pydantic import BaseModel, model_validator


class ProjectionDataConfig(BaseModel):
    min: float = 0.0
    max: float = 1.0
    is_projected: bool = False
    data_projector: Callable[[torch.Tensor], torch.Tensor] | None = None

    @model_validator(mode="after")
    def set_data_projector(self):
        if self.is_projected == True:
            self.data_projector = lambda x: x.clamp_(self.min, self.max)
        else:
            self.data_projector = lambda x: x
        return self

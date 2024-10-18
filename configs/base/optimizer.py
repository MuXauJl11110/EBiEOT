from pydantic import BaseModel


class BaseOptimizerConfig(BaseModel):
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)

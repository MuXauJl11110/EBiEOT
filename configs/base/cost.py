from pydantic import BaseModel


class BaseCostConfig(BaseModel):
    x_dim: int = 2
    y_dim: int = 2
    m_potentials: int = 25
    epsilon: float = 1.0

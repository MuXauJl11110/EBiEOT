from dataclasses import dataclass


@dataclass
class Model:
    x_dim: int
    y_dim: int
    n_potentials: int
    m_potentials: int
    epsilon: float
    sampling_batch_size: int
    A_diagonal_init: float
    cost_function: str

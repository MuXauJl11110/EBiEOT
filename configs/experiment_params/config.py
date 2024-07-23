from dataclasses import dataclass


@dataclass
class ExperimentParams:
    project_name: str
    seed: int
    device: str
    double_precision: bool
    init_by_samples: bool
    cost_included: bool
    gradient_max_norm: float

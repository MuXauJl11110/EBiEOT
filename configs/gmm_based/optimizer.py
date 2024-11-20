from configs.base.optimizer import BaseOptimizerConfig


class OptPairedConfig(BaseOptimizerConfig):
    lr: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)


class OptUnpairedConfig(BaseOptimizerConfig):
    lr: float = 2e-4
    betas: tuple[float, float] = (0.9, 0.999)

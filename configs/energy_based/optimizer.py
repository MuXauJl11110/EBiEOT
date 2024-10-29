from configs.base.optimizer import BaseOptimizerConfig


class OptPairedConfig(BaseOptimizerConfig):
    lr: float = 1e-4
    betas: tuple[float, float] = (0.2, 0.99)


class OptUnpairedConfig(BaseOptimizerConfig):
    lr: float = 2e-4
    betas: tuple[float, float] = (0.2, 0.99)

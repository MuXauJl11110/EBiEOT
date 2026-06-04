from typing import Protocol

from omegaconf import DictConfig
from torch.optim import Optimizer


class BaselineTrainer(Protocol):
    def build_optimizers(self, cfg: DictConfig) -> tuple[Optimizer, ...]: ...

    def train_step(self, step: int, usd, utd, pd) -> dict[str, float]: ...

    def state_dict(self) -> dict: ...

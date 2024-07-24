from abc import ABC, abstractmethod

import torch


class Sampler(ABC):
    def __init__(
        self,
        device: str = "cuda",
    ):
        self.device = device

    @abstractmethod
    def sample(self, size: int = 5) -> torch.Tensor:
        pass

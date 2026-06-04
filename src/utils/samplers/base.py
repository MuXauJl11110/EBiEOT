from abc import ABC, abstractmethod

import torch
from torch import Generator


class Sampler(ABC):
    def __init__(
        self,
        generator: Generator | None = None,
        device: str = "cuda",
    ):
        self.device = device

        if generator is not None:
            self.generator = generator
        else:
            self.generator = torch.Generator(device=device)

    @abstractmethod
    def sample(self, size: int = 5) -> torch.Tensor:
        pass

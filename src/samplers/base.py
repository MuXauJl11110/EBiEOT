from abc import ABC, abstractmethod

import numpy as np
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


class TensorSampler(Sampler):
    def __init__(self, tensor: torch.Tensor, device="cuda"):
        super(TensorSampler, self).__init__(device)
        self.tensor = torch.clone(tensor).to(device)

    def sample(self, size=5):
        assert size <= self.tensor.shape[0]

        ind = torch.tensor(
            np.random.choice(np.arange(self.tensor.shape[0]), size=size, replace=False), device=self.device
        )
        return torch.clone(self.tensor[ind]).detach().to(self.device)

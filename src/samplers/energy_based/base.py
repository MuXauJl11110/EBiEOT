from abc import ABC, abstractmethod

import torch


# TODO: annotate __init__ method
class SampleBuffer(ABC):

    def __init__(self, noise_gen) -> None:
        self.noise_gen = noise_gen

    @abstractmethod
    def push(self, Xs: torch.Tensor, samples: torch.Tensor, ids: list[int] | None) -> None:
        raise NotImplementedError()

    @abstractmethod
    def get(self, n_samples: int):
        raise NotImplementedError()

    @abstractmethod
    def __len__(self):
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, Xs: torch.Tensor):
        raise NotImplementedError()

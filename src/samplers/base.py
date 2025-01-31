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
    def __init__(self, tensor: torch.Tensor, device: str = "cuda"):
        super(TensorSampler, self).__init__(device)
        self.tensor = torch.clone(tensor).to(device)

    def sample(self, size: int = 5):
        assert size <= self.tensor.shape[0]

        ind = torch.tensor(
            np.random.choice(np.arange(self.tensor.shape[0]), size=size, replace=False), device=self.device
        )
        return torch.clone(self.tensor[ind]).detach().to(self.device)


class TensorLabeledSampler:
    def __init__(self, tensor: torch.Tensor, labels: list[str], device: str = "cuda"):
        assert len(tensor) == len(labels)
        self.device = device
        self.tensor = torch.clone(tensor).to(device)
        self.labels = labels

    def sample(self, size: int = 5) -> tuple[torch.Tensor, list[str]]:
        assert size <= self.tensor.shape[0]

        ind = np.random.choice(np.arange(self.tensor.shape[0]), size=size, replace=False)
        ind_tensor = torch.tensor(ind, device=self.device)
        return torch.clone(self.tensor[ind_tensor]).detach().to(self.device), [self.labels[i] for i in ind]


class PairedLabeledSampler:
    def __init__(
        self,
        source_tensor: torch.Tensor,
        source_labels: list[str],
        target_tensor: torch.Tensor,
        target_labels: list[str],
        device: str = "cuda",
    ):
        assert len(source_tensor) == len(target_tensor)
        assert len(source_tensor) == len(source_labels)
        assert len(target_tensor) == len(target_labels)
        self.device = device
        self.source_tensor = source_tensor
        self.source_labels = source_labels
        self.target_tensor = target_tensor
        self.target_labels = target_labels
        self.n = source_tensor.shape[0]

    def sample(self, size: int = 5) -> tuple[torch.Tensor, list[str], torch.Tensor, list[str]]:
        assert size <= self.n

        ind = np.random.choice(np.arange(self.n), size=size, replace=False)
        ind_tensor = torch.tensor(ind, device=self.device)
        return (
            self.source_tensor[ind_tensor].detach().to(self.device),
            [self.source_labels[i] for i in ind],
            self.target_tensor[ind_tensor].detach().to(self.device),
            [self.target_labels[i] for i in ind],
        )

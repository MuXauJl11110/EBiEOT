import random
from typing import Generator

import torch
from src.utils.samplers.base import Sampler
from torch.utils.data import DataLoader


class TensorSampler(Sampler):
    def __init__(self, tensor: torch.Tensor, device: str = "cuda") -> None:
        super().__init__(device)
        self.tensor = torch.clone(tensor).to(device)

    def sample(self, size: int = 5) -> torch.Tensor:
        assert size <= self.tensor.shape[0]

        ind = torch.randperm(self.tensor.shape[0], device=self.tensor.device)[:size]
        return torch.clone(self.tensor[ind]).detach().to(self.device)


class PairedTensorBatchSampler:
    """Sample independent minibatches of paired rows from fixed ``(X, Y)`` tensors."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.y = y

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        n = self.x.shape[0]
        idx = torch.randint(0, n, (batch_size,), device=self.x.device)
        return self.x[idx], self.y[idx]


class DatasetSampler(Sampler):
    def __init__(
        self,
        dataset: torch.Tensor,
        device: str = "cuda",
    ) -> None:
        super().__init__(device=device)
        self.dataset = dataset

    def sample(self, batch_size: int = 16) -> torch.Tensor:
        ind = random.choices(range(len(self.dataset)), k=batch_size)
        with torch.no_grad():
            batch = self.dataset[ind].clone().to(self.device)
        return batch


class LoaderSampler(Sampler):
    def __init__(self, loader: DataLoader, device: str = "cuda") -> None:
        super().__init__(device)
        self.loader = loader
        self.it = iter(self.loader)

    def sample(self, size: int = 5) -> torch.Tensor:
        assert size <= self.loader.batch_size
        try:
            batch, _ = next(self.it)
        except StopIteration:
            self.it = iter(self.loader)
            return self.sample(size)
        if len(batch) < size:
            return self.sample(size)

        return batch[:size].to(self.device)


class PairedLoaderSampler(Sampler):
    def __init__(self, loader: DataLoader, device: str = "cuda") -> None:
        super().__init__(device)
        self.loader = loader
        self.generator = iter(self.loader)

    def sample(self, size: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
        X_list, Y_list = [], []
        batch_size = self.loader.batch_size
        num_sampling_iterations = size // batch_size
        num_remaining_samples = size % batch_size
        for _ in range(num_sampling_iterations):
            X, Y = self.sample_from_generator(self.generator, self.loader)
            X_list.append(X)
            Y_list.append(Y)
        if num_remaining_samples > 0:
            X, Y = self.sample_from_generator(self.generator, self.loader)
            X_list.append(X[:num_remaining_samples])
            Y_list.append(Y[:num_remaining_samples])
        return torch.cat(X_list).to(self.device), torch.cat(Y_list).to(self.device)

    def sample_from_generator(
        self,
        generator: Generator[tuple[torch.Tensor, torch.Tensor], None, None],
        loader: DataLoader,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            try:
                X, Y = next(generator)
            except StopIteration:
                generator = iter(loader)
                X, Y = next(generator)
        return X, Y

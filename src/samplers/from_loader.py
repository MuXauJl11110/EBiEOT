from typing import Generator

import torch
from torch.utils.data import DataLoader

from src.samplers.base import Sampler


class LoaderSampler(Sampler):
    def __init__(self, loader: DataLoader, device: str = "cuda"):
        super(LoaderSampler, self).__init__(device)
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
    def __init__(self, loader: DataLoader, device: str = "cuda"):
        super(PairedLoaderSampler, self).__init__(device)
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
        self, generator: Generator[tuple[torch.Tensor, torch.Tensor], None, None], loader: DataLoader
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            try:
                X, Y = next(generator)
            except StopIteration:
                generator = iter(loader)
                X, Y = next(generator)
        return X, Y

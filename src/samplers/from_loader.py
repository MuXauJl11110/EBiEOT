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
        self.it = iter(self.loader)

    def sample(self, size: int = 5) -> torch.Tensor:
        assert size <= self.loader.batch_size
        try:
            batch_x, batch_y = next(self.it)
        except StopIteration:
            self.it = iter(self.loader)
            return self.sample(size)
        if len(batch_x) < size:
            return self.sample(size)

        return batch_x[:size].to(self.device), batch_y[:size].to(self.device)

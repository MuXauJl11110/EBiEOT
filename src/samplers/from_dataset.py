import random

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from src.samplers.base import Sampler


class DatasetSampler(Sampler):
    def __init__(
        self,
        dataset: torch.Tensor,
        device: str = "cuda",
    ):
        super(DatasetSampler, self).__init__(device=device)
        self.dataset = dataset

    def sample(self, batch_size: int = 16):
        ind = random.choices(range(len(self.dataset)), k=batch_size)
        with torch.no_grad():
            batch = self.dataset[ind].clone().to(self.device).float()
        return batch


class DatasetSamplerLabeled(Sampler):
    def __init__(self, dataset: Dataset, batch_size: int = 32, num_workers: int = 32, device: str = "cuda"):
        super(DatasetSamplerLabeled, self).__init__(device=device)
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

        with torch.no_grad():
            self.dataset = torch.cat([X for (X, _) in loader])
            self.labels = torch.cat([y for (_, y) in loader])

    def sample(self, batch_size: int = 16):
        ind = random.choices(range(len(self.dataset)), k=batch_size)
        with torch.no_grad():
            batch_x = self.dataset[ind].clone().to(self.device).float()
            batch_y = self.labels[ind].clone().to(self.device)
        return batch_x, batch_y

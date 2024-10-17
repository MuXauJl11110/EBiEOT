import random

import numpy as np
import torch
from torch.distributions.distribution import Distribution

from src.samplers.energy_based.base import SampleBuffer


# TODO: which type is ids?
# TODO: annotate return parameters of the class
# TODO: annotate return parameters for the get_random method
class SampleBufferEgEOT(SampleBuffer):
    def __init__(self, noise_gen: Distribution, p: float = 0.95, max_samples: int = 10000, device: str = "cpu"):
        self.max_samples = max_samples
        self.buffer_samples = torch.empty(0)  # Tensor for samples
        self.buffer_Xs = torch.empty(0)  # Tensor for Xs
        self.device = device
        self.p = p
        super().__init__(noise_gen)

    def push(self, Xs: torch.Tensor, samples: torch.Tensor, ids=None) -> None:
        samples = samples.detach().cpu()
        Xs = Xs.detach().cpu()

        if ids is None:
            if self.buffer_samples.numel() == 0:  # Buffer is empty, initialize it
                self.buffer_samples = samples
                self.buffer_Xs = Xs
            else:
                self.buffer_samples = torch.cat((self.buffer_samples, samples), dim=0)
                self.buffer_Xs = torch.cat((self.buffer_Xs, Xs), dim=0)

            # Keep buffer within max_samples
            if self.buffer_samples.size(0) > self.max_samples:
                excess = self.buffer_samples.size(0) - self.max_samples
                self.buffer_samples = self.buffer_samples[excess:]
                self.buffer_Xs = self.buffer_Xs[excess:]
        else:
            assert len(ids) == len(samples)
            assert max(ids) < self.buffer_samples.size(0)
            self.buffer_samples[ids] = samples
            self.buffer_Xs[ids] = Xs

    def get(self, n_samples: int):
        indices = random.choices(range(self.buffer_samples.size(0)), k=n_samples)
        samples = self.buffer_samples[indices].to(self.device)
        Xs = self.buffer_Xs[indices].to(self.device)
        return Xs, samples, indices

    def get_random(self, Xs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        samples = self.noise_gen.sample((Xs.size(0),)).to(Xs)
        return Xs, samples, None

    def __len__(self) -> int:
        return self.buffer_samples.size(0)

    def __call__(self, Xs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        batch_size = Xs.size(0)

        if len(self) < 1:
            return self.get_random(Xs)

        n_replay = (np.random.rand(batch_size) < self.p).sum()

        if n_replay == 0:
            Xs, samples, _ = self.get_random(Xs)
        elif n_replay == batch_size:
            Xs, samples, _ = self.get(n_replay)
        else:
            replay_Xs, replay_samples, _ = self.get(n_replay)
            random_Xs, random_samples, _ = self.get_random(Xs[n_replay:])
            Xs, samples = torch.cat([replay_Xs, random_Xs], 0), torch.cat([replay_samples, random_samples], 0)

        return Xs, samples, None

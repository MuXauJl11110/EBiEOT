import random

import numpy as np
import torch

from src.samplers.energy_based.base import SampleBuffer


# TODO: which type is ids, noise_gen?
# TODO: annotate return parameters of the class
# TODO: annotate return parameters for the get_random method
class SampleBufferEgEOT(SampleBuffer):

    def __init__(self, noise_gen, p: float = 0.95, max_samples: int = 10000, device: str = "cpu"):
        self.max_samples = max_samples
        self.buffer = []
        self.device = device
        self.p = p
        super().__init__(noise_gen)

    def push(self, Xs: torch.Tensor, samples: torch.Tensor, ids) -> None:
        samples = samples.detach().cpu()
        Xs = Xs.detach().cpu()

        if ids is None:
            for sample, X in zip(samples, Xs):
                self.buffer.append((sample, X))

                if len(self.buffer) > self.max_samples:
                    self.buffer.pop(0)
        else:
            assert len(ids) == len(samples)
            assert max(ids) < len(self.buffer)
            samp_Xs = [(sample, X) for sample, X in zip(samples, Xs)]
            for i, _id in enumerate(ids):
                self.buffer[_id] = samp_Xs[i]

    def get(self, n_samples: int):
        indices = random.choices(range(len(self.buffer)), k=n_samples)
        items = [self.buffer[i] for i in indices]
        samples, Xs = zip(*items)
        samples = torch.stack(samples, 0).to(self.device)
        Xs = torch.stack(Xs, 0).to(self.device)
        return Xs, samples, indices

    def get_random(self, Xs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        samples = self.noise_gen.sample((Xs.size(0),)).to(Xs)
        return Xs, samples, None

    def __len__(self) -> int:
        return len(self.buffer)

    def __call__(self, Xs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        batch_size = Xs.size(0)
        if len(self) < 1:
            return self.get_random(Xs)

        n_replay = (np.random.rand(batch_size) < self.p).sum()

        if n_replay == 0:
            Xs, samples, _ = self.get_random(Xs)
        elif n_replay == batch_size:
            Xs, samples, _ = self.get(n_replay, device=Xs.device)
        else:
            replay_Xs, replay_samples, _ = self.get(n_replay)
            random_Xs, random_samples, _ = self.get_random(Xs[n_replay:])
            Xs, samples = torch.cat([replay_Xs, random_Xs], 0), torch.cat([replay_samples, random_samples], 0)

        return Xs, samples, None


class SampleBufferStatic(SampleBuffer):

    def __init__(self, noise_gen, Xs: torch.Tensor, device: str = "cpu"):
        self.Xs = Xs
        self.noise_gen = noise_gen
        self.Ys = self.noise_gen((Xs.size(0),)).cpu()
        self.device = device

    def __len__(self):
        return len(self.Xs)

    def push(self, Xs: torch.Tensor, samples: torch.Tensor, ids):
        self.Xs[ids] = Xs.detach().cpu()
        self.Ys[ids] = samples.detach().cpu()
        del Xs
        del samples

    def get(self, n_samples: int):
        indices = np.random.choice(len(self), n_samples)
        return self.Xs[indices].to(self.device), self.Ys[indices].to(self.device), indices

    def __call__(self, Xs: torch.Tensor):
        return self.get(len(Xs))

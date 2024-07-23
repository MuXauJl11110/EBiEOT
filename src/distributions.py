from typing import Generator

import numpy as np
import torch
from sklearn import datasets
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import DataLoader, TensorDataset

from src.discrete_ot import OTPlanSampler


class Sampler:
    def __init__(
        self,
        device="cuda",
    ):
        self.device = device

    def sample(self, batch_size: int = 5):
        pass


class SwissRollSampler(Sampler):
    def __init__(self, dim=2, device: str = "cuda"):
        super(SwissRollSampler, self).__init__(device=device)
        assert dim == 2
        self.dim = 2

    def sample(self, batch_size: int = 10):
        # batch = datasets.make_swiss_roll(n_samples=batch_size, noise=0.8)[0].astype("float32")[:, [0, 2]] / 7.5
        batch = datasets.make_swiss_roll(n_samples=batch_size, noise=0.8)[0][:, [0, 2]] / 7.5
        return torch.tensor(batch, device=self.device)


class StandardNormalSampler(Sampler):
    def __init__(self, dim: int = 1, device: str = "cuda"):
        super(StandardNormalSampler, self).__init__(device=device)
        self.dim = dim

    def sample(self, batch_size: int = 10):
        return torch.randn(batch_size, self.dim, device=self.device)


class StandardNormalOnCircleSampler(Sampler):
    def __init__(self, R: float, D: torch.Tensor, device: str = "cuda"):
        super(StandardNormalOnCircleSampler, self).__init__(device=device)
        self.R = R
        self.D = D
        self.dim = 2

    def compute(self, t: torch.Tensor, diag: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        assert len(t.shape) == 1  # t shape batch*1

        c, s = torch.cos(2 * torch.pi * t.squeeze()), torch.sin(2 * torch.pi * t.squeeze())
        x, y = self.R * c, self.R * s
        a = torch.stack([x, y]).T
        Q = torch.stack([torch.stack([c, -s]), torch.stack([s, c])]).permute(2, 0, 1)

        QT_D = torch.bmm(Q.permute(0, 2, 1), self.D.unsqueeze(0).repeat(t.shape[0], 1, 1))
        QTDQ = torch.bmm(QT_D, Q)

        if diag:
            QTDQ = torch.diagonal(QTDQ, dim1=1, dim2=2)

        return a, QTDQ

    def sample(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        loc, covariance_matrix = self.compute(t)
        mn = MultivariateNormal(loc=loc, covariance_matrix=covariance_matrix)

        return mn.sample()


class GridGaussiansSampler(Sampler):
    def __init__(
        self,
        dim: int = 2,
        x_mode: int = 2,
        y_mode: int = 2,
        x_from: float = -2.0,
        x_to: float = 2.0,
        y_from: float = -2.0,
        y_to: float = 2.0,
        std: float = 0.15,
        shuffle: bool = True,
        device: str = "cuda",
    ):
        super(GridGaussiansSampler, self).__init__(device=device)
        self.dim = dim
        self.std = std

        assert x_from < x_to
        assert y_from < y_to
        mu_x = torch.from_numpy(np.linspace(x_from, x_to, x_mode))
        mu_y = torch.from_numpy(np.linspace(y_from, y_to, y_mode))
        self.mu = torch.cartesian_prod(mu_x, mu_y).to(device=device)
        if shuffle:
            perm = torch.randperm(x_mode * y_mode)
            self.mu = self.mu[perm, :]
        self.cov = torch.diag(std * torch.ones(x_mode * y_mode, device=device))
        self.distribution = MultivariateNormal(loc=self.mu.T, scale_tril=self.cov)

    def sample(self, batch_size: int = 10):
        # return self.distribution.sample((batch_size,)).swapaxes(0, 1).flatten(1).T
        return self.distribution.sample((batch_size,)).swapaxes(1, 2).reshape(batch_size * len(self.mu), self.dim)


class PairedSampler(Sampler):
    def __init__(
        self,
        X_sampler: Sampler,
        Y_sampler: Sampler,
        batch_size: int = 128,
        n_paired_samples: int | None = None,
        m_unpaired_samples: int | None = None,
        mini_batch_size: int | None = None,
        otp_sampler: OTPlanSampler | None = None,
        device: str = "cuda",
    ):
        super(PairedSampler, self).__init__(device=device)
        self.batch_size = batch_size
        self.device = device

        if mini_batch_size is not None and otp_sampler is None:
            raise ValueError("OTPlanSampler must initialized during mini-batch sampling! But is None.")
        self.paired_loader = self._init_paired_loader(
            X_sampler, Y_sampler, n_paired_samples, mini_batch_size, otp_sampler
        )
        self.paired_generator = iter(self.paired_loader)

        self.unpaired_loader = self._init_unpaired_loader(X_sampler, Y_sampler, m_unpaired_samples)
        self.unpaired_generator = iter(self.unpaired_loader)

    def sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sample_from_generator(self.unpaired_generator, self.unpaired_loader)

    def sample_pair(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sample_from_generator(self.paired_generator, self.paired_loader)

    def sample_from_generator(self, generator: Generator, loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            try:
                X, Y = next(generator)
            except StopIteration:
                generator = iter(loader)
                X, Y = next(generator)
        return X, Y

    def _init_paired_loader(
        self,
        X_sampler: Sampler,
        Y_sampler: Sampler,
        n_paired_samples: int,
        mini_batch_size: int | None,
        otp_sampler: OTPlanSampler | None,
    ) -> DataLoader:
        X_paired, Y_paired = torch.empty((0, X_sampler.dim)), torch.empty((0, Y_sampler.dim))
        if mini_batch_size is not None:
            num_sampling_iterations = n_paired_samples // mini_batch_size
            num_remaining_samples = n_paired_samples % mini_batch_size
            for _ in range(num_sampling_iterations):
                _X, _Y = X_sampler.sample(mini_batch_size), Y_sampler.sample(mini_batch_size)
                X, Y = otp_sampler.sample_plan(_X, _Y)
                X_paired, Y_paired = torch.cat((X_paired, X), 0), torch.cat((Y_paired, Y), 0)
            if num_remaining_samples > 0:
                _X, _Y = X_sampler.sample(num_remaining_samples), Y_sampler.sample(num_remaining_samples)
                X, Y = otp_sampler.sample_plan(_X, _Y)
                X_paired, Y_paired = torch.cat((X_paired, X), 0), torch.cat((Y_paired, Y), 0)
        else:
            num_gaussians = len(X_sampler.mu)
            X, Y = X_sampler.sample(n_paired_samples), Y_sampler.sample(n_paired_samples)
            for i in range(num_gaussians):
                indices = np.arange(i, len(X), num_gaussians)
                for x in X[indices]:
                    for y in Y[indices]:
                        X_paired, Y_paired = torch.cat((X_paired, x[None, :]), 0), torch.cat((Y_paired, y[None, :]), 0)

        dataset = TensorDataset(X_paired, Y_paired)
        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, generator=torch.Generator(device=self.device)
        )

    def _init_unpaired_loader(
        self,
        X_sampler: Sampler,
        Y_sampler: Sampler,
        m_unpaired_samples: int,
    ) -> DataLoader:
        X, Y = X_sampler.sample(m_unpaired_samples), Y_sampler.sample(m_unpaired_samples)
        dataset = TensorDataset(X, Y)
        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, generator=torch.Generator(device=self.device)
        )

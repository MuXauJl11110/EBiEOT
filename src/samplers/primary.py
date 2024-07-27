import numpy as np
import torch
from sklearn import datasets
from torch.distributions.multivariate_normal import MultivariateNormal

from src.samplers.base import Sampler


class SwissRollSampler(Sampler):
    def __init__(self, dim: int = 2, device: str = "cuda"):
        super(SwissRollSampler, self).__init__(device=device)
        assert dim == 2
        self.dim = 2

    def sample(self, batch_size: int = 10):
        batch = datasets.make_swiss_roll(n_samples=batch_size, noise=0.8)[0].astype("float32")[:, [0, 2]] / 7.5
        # batch = datasets.make_swiss_roll(n_samples=batch_size, noise=0.8)[0][:, [0, 2]] / 7.5
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
        mu_x = torch.from_numpy(np.linspace(x_from, x_to, x_mode, dtype=np.float32))
        mu_y = torch.from_numpy(np.linspace(y_from, y_to, y_mode, dtype=np.float32))
        self.mu = torch.cartesian_prod(mu_x, mu_y).to(device=device)
        if shuffle:
            perm = torch.randperm(x_mode * y_mode)
            self.mu = self.mu[perm, :]
        self.cov = torch.diag(std * torch.ones(x_mode * y_mode, device=device))
        self.distribution = MultivariateNormal(loc=self.mu.T, scale_tril=self.cov)

    def sample(self, batch_size: int = 10):
        return self.distribution.sample((batch_size,)).swapaxes(1, 2).reshape(batch_size * len(self.mu), self.dim)

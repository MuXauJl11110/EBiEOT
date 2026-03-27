import numpy as np
import torch
from src.samplers.base import Sampler
from torch import Generator
from torch.distributions.multivariate_normal import MultivariateNormal


def swiss_roll_transform(
    t: torch.Tensor,
    generator: Generator,
    noise: float = 0.0,
) -> torch.Tensor:
    """
    Maps parameter t → 2D swiss roll with optional isotropic noise.
    """
    x = t * torch.cos(t)
    z = t * torch.sin(t)

    X = torch.stack((x, z), dim=1)

    if noise > 0:
        X = X + noise * torch.randn(size=X.shape, generator=generator)

    return X


class SwissRollSampler(Sampler):
    def __init__(
        self,
        dim: int = 2,
        noise: float = 0.8,
        scale: float = 7.5,
        t_min: float = 1.5 * np.pi,
        t_max: float = 4.5 * np.pi,
        generator: Generator | None = None,
        device: str = "cuda",
    ):
        super().__init__(generator=generator, device=device)
        assert dim == 2

        self.noise = noise
        self.scale = scale
        self.t_min = t_min
        self.t_max = t_max

    def sample(self, batch_size: int = 10):
        diff = self.t_max - self.t_min

        t = self.t_min + diff * torch.rand((batch_size,), generator=self.generator, device=self.device)

        batch = swiss_roll_transform(t=t, generator=self.generator, noise=self.noise) / self.scale

        return batch


class StandardNormalSampler(Sampler):
    def __init__(self, dim: int = 1, generator: Generator | None = None, device: str = "cuda"):
        super(StandardNormalSampler, self).__init__(generator=generator, device=device)
        self.dim = dim

    def sample(self, batch_size: int = 10):
        return torch.randn(batch_size, self.dim, device=self.device)


class StandardNormalOnCircleSampler(Sampler):
    def __init__(self, R: float, D: torch.Tensor, device: str = "cuda"):
        super(StandardNormalOnCircleSampler, self).__init__(device=device)
        self.R = R  # radius
        self.D = D  # rotation matrix
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

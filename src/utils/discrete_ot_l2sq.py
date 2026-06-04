"""Entropic discrete OT with squared L2 ground cost (mnist2to3 DOT initialization)."""


import numpy as np
import ot
import torch
import torch.distributions as TD


class DiscreteEOT_l2sq_sampler:
    @staticmethod
    def discrete_sample_conditional(
        Y: torch.Tensor,
        G: torch.Tensor,
        i_x: int,
        n_pts: int,
        return_indices: bool = False,
    ):
        probs = G[i_x] / torch.sum(G[i_x])
        distrib = TD.Categorical(probs=probs)
        numbers = distrib.sample((n_pts,))
        if not return_indices:
            return Y[numbers]
        return numbers

    def __init__(self, X, Y, G, device: str | torch.device = "cpu"):
        self.device = device
        self.X = torch.tensor(X).float().clone().detach().to(self.device)
        self.Y = torch.tensor(Y).float().clone().detach().to(self.device)
        self.G = torch.tensor(G).float().clone().detach().to(self.device)

    def sample_by_indices(self, x_indices, return_indices: bool = False):
        spls = []
        for x_idx in x_indices:
            spls.append(
                self.discrete_sample_conditional(
                    self.Y, self.G, x_idx, 1, return_indices=return_indices
                )
            )
        return torch.cat(spls, dim=0)


class DiscreteEOT_l2sq:
    def _cast(self, x):
        if self.dtype == "torch32":
            return torch.tensor(x).float().clone().detach().to(self.device)
        if self.dtype == "torch64":
            return torch.tensor(x).double().clone().detach().to(self.device)
        raise ValueError(f"Unknown dtype: {self.dtype}")

    def __init__(
        self,
        verbose: bool = False,
        method: str = "sinkhorn_log",
        stopThr: float = 1e-09,
        numItermax: int = 10000,
        dtype: str = "torch32",
        device: str | torch.device = "cpu",
    ):
        self.verbose = verbose
        self.method = method
        self.stopThr = stopThr
        self.numItermax = numItermax
        self.dtype = dtype
        self.device = device

    def solve(self, X: torch.Tensor, Y: torch.Tensor, eps: float):
        _X, _Y = self._cast(X), self._cast(Y)
        M = 0.5 * ot.dist(_X, _Y)
        xL, yL = X.shape[0], Y.shape[0]
        wX = self._cast(np.ones(xL) / xL)
        wY = self._cast(np.ones(yL) / yL)
        G = ot.sinkhorn(
            wX,
            wY,
            M,
            eps,
            method=self.method,
            numItermax=self.numItermax,
            stopThr=self.stopThr,
            verbose=self.verbose,
        )
        return DiscreteEOT_l2sq_sampler(_X, _Y, G, device=self.device)

import torch

from src.costs.base import BaseCost


class SquareCost(BaseCost):
    def func(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # [1]
        return (x - y).square().mean()

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseGenerativeModel(ABC, nn.Module):
    @abstractmethod
    def forward(self, batched_x: torch.Tensor) -> torch.Tensor:  # -> [bs]
        pass

    @abstractmethod
    def compute_paired_loss(self, X_paired: torch.Tensor, Y_paired: torch.Tensor) -> torch.Tensor:  # -> [1]
        pass

    @abstractmethod
    def compute_unpaired_loss(self, X_unpaired: torch.Tensor, Y_unpaired: torch.Tensor) -> dict[str, torch.Tensor]:
        pass

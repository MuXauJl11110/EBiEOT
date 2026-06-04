from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseEBiEOT(ABC, nn.Module):
    """Abstract base for EBiEOT (Energy-Based Entropic Optimal Transport).

    Concrete subclasses implement batched ``forward`` (e.g. maps or samples conditioned on ``x``)
    and two training hooks: `compute_paired_loss` for aligned `(x_i, y_i)` batches, and
    `compute_unpaired_loss` when only marginal batches of `x` and `y` are observed.
    """

    @abstractmethod
    def forward(self, batched_x: torch.Tensor) -> torch.Tensor:  # -> [bs]
        """Run the model on a batch of ``x`` (e.g. push-forward or sampling).

        Args:
            batched_x: Batch of inputs, shape ``(bs, x_dim)`` (or project-specific layout with
                leading batch dimension ``bs``).

        Returns:
            Tensor with leading dimension ``bs``; trailing dimensions depend on the subclass.
        """
        pass

    @abstractmethod
    def compute_paired_loss(
        self, X_paired: torch.Tensor, Y_paired: torch.Tensor
    ) -> torch.Tensor:  # -> [1]
        """Loss from fully observed pairs ``(x_i, y_i)``.

        Args:
            X_paired: Batch of ``x``, shape ``(bs, x_dim)``.
            Y_paired: Matching ``y``, same ``bs``, shape ``(bs, y_dim)``.

        Returns:
            Scalar training objective (typically shape ``()`` or ``(1,)``).
        """
        pass

    @abstractmethod
    def compute_unpaired_loss(
        self, X_unpaired: torch.Tensor, Y_unpaired: torch.Tensor
    ) -> dict[str, torch.Tensor]:  # -> [1]
        """Loss when only marginal batches of ``x`` and ``y`` are available (no pairing).

        Args:
            X_unpaired: Batch from the ``x`` marginal, shape ``(bs_x, x_dim)``.
            Y_unpaired: Batch from the ``y`` marginal, shape ``(bs_y, y_dim)`` (sizes may follow
                the concrete method; see subclass).

        Returns:
            Mapping that must include the optimization target; subclasses typically store the scalar
            objective under the key ``loss`` and may add diagnostic tensors under other keys.
        """
        pass

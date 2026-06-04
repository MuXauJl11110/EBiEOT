"""Log-sum-exponential costs with MLP parameterizations (:class:`BaseLSECost`)."""

import torch
import torch.nn as nn

from src.ebieot.costs.base import BaseLSECost


def _build_mlp_backbone(
    x_dim: int,
    hidden_channels: list[int],
    activation_layer: type[nn.Module],
) -> tuple[nn.Module, int]:
    if not hidden_channels:
        return nn.Identity(), x_dim
    layers: list[nn.Module] = []
    in_dim = x_dim
    for hidden_dim in hidden_channels:
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(activation_layer())
        in_dim = hidden_dim
    return nn.Sequential(*layers), in_dim


class MLPLSECost(BaseLSECost):
    """LSE cost with separate MLP backbones and linear heads for ``log v_m(x)`` and ``b_m(x)``."""

    def __init__(
        self,
        log_v_m_hidden_channels: list[int],
        b_m_hidden_channels: list[int],
        x_dim: int = 2,
        y_dim: int = 2,
        m_potentials: int = 25,
        epsilon: float = 1.0,
        log_v_m_activation_layer: nn.Module = nn.ReLU,
        b_m_activation_layer: nn.Module = nn.ReLU,
    ):
        """
        Args:
            log_v_m_hidden_channels: Hidden layer widths for the ``log v_m`` tower (output head size is ``M``).
            b_m_hidden_channels: Hidden layer widths for the ``b_m`` tower (output head size is ``M * y_dim``).
            x_dim: Input dimension for :math:`x`. Default ``2``.
            y_dim: Dimension of :math:`y`. Default ``2``.
            m_potentials: Number of mixture terms :math:`M`. Default ``25``.
            epsilon: Same role as in :class:`BaseLSECost`. Default ``1.0``.
            log_v_m_activation_layer: Class for activations in the :math:`v_m` tower.
            b_m_activation_layer: Class for activations in the :math:`b_m` tower.
        """
        super().__init__(x_dim, y_dim)
        self.m_potentials = m_potentials
        self.register_buffer("epsilon", torch.tensor(epsilon))

        log_v_activation = log_v_m_activation_layer
        b_m_activation = b_m_activation_layer
        self._log_v_backbone, log_v_latent = _build_mlp_backbone(
            x_dim, log_v_m_hidden_channels, log_v_activation
        )
        self._log_v_head = nn.Sequential(
            nn.Linear(log_v_latent, m_potentials), nn.LogSoftmax(dim=-1)
        )

        self._b_backbone, b_latent = _build_mlp_backbone(x_dim, b_m_hidden_channels, b_m_activation)
        self._b_head = nn.Linear(b_latent, m_potentials * y_dim)
        nn.init.normal_(self._b_head.weight, std=0.01)
        nn.init.zeros_(self._b_head.bias)

    def compute_log_v_m(self, x: torch.Tensor) -> torch.Tensor:  # [M]
        """``log v_m(x)`` for :math:`m=1..M`; shape ``(M,)``."""
        features = self._log_v_backbone(x[None, :])
        return self._log_v_head(features).reshape(self.m_potentials)

    def compute_b_m(self, x: torch.Tensor) -> torch.Tensor:  # [M x y_dim]
        """Coefficients :math:`b_m(x)`; shape ``(M, y_dim)``."""
        features = self._b_backbone(x[None, :])
        return self._b_head(features).reshape(self.m_potentials, self.y_dim)


class SharedMLPLSECost(BaseLSECost):
    """LSE cost with one shared MLP backbone and linear heads for ``log v_m`` and ``b_m``."""

    def __init__(
        self,
        shared_hidden_channels: list[int],
        x_dim: int = 2,
        y_dim: int = 2,
        m_potentials: int = 25,
        epsilon: float = 1.0,
        activation_layer: nn.Module = nn.SiLU,
        use_layer_norm: bool = True,
    ):
        """
        Args:
            shared_hidden_channels: Widths of the shared ``Linear`` stack after ``x``. If empty, heads read
                directly from ``x`` (``latent_dim == x_dim``).
            x_dim: Input dimension for :math:`x`. Default ``2``.
            y_dim: Dimension of :math:`y`. Default ``2``.
            m_potentials: Number of mixture terms :math:`M`. Default ``25``.
            epsilon: Same role as in :class:`BaseLSECost`. Default ``1.0``.
            activation_layer: Class used after each hidden ``Linear`` (except inside ``LayerNorm`` blocks).
            use_layer_norm: If True, apply ``LayerNorm`` after each hidden linear layer.
        """
        super().__init__(x_dim, y_dim)
        self.m_potentials = m_potentials
        self.register_buffer("epsilon", torch.tensor(epsilon))

        # 1. Shared Feature Extractor
        layers = []
        in_dim = x_dim
        for hidden_dim in shared_hidden_channels:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(activation_layer())
            in_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)

        # 2. Heads (Mapping latent to targets)
        latent_dim = shared_hidden_channels[-1] if shared_hidden_channels else x_dim

        self.log_v_m_head = nn.Sequential(
            nn.Linear(latent_dim, m_potentials), nn.LogSoftmax(dim=-1)
        )

        self.b_m_head = nn.Linear(latent_dim, m_potentials * y_dim)

        # 3. Safe Initialization to prevent early LogSumExp saturation
        nn.init.normal_(self.b_m_head.weight, std=0.01)
        nn.init.zeros_(self.b_m_head.bias)

    def compute_log_v_m(self, x: torch.Tensor) -> torch.Tensor:  # [M]
        features = self.backbone(x[None, :])
        return self.log_v_m_head(features).reshape(self.m_potentials)

    def compute_b_m(self, x: torch.Tensor) -> torch.Tensor:  # [M, y_dim]
        features = self.backbone(x[None, :])
        return self.b_m_head(features).reshape(self.m_potentials, self.y_dim)

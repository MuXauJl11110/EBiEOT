import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from src.baselines import baseline_registry

_DEVICE = torch.device("cpu")
_BATCH_SIZE = 8
_POOL = 32


class _MockUnpairedSampler:
    def __init__(self, dim: int, device: torch.device, pool: int = _POOL) -> None:
        self._data = torch.randn(pool, dim, device=device)

    def sample(self, batch_size: int) -> torch.Tensor:
        idx = torch.randint(0, self._data.shape[0], (batch_size,), device=self._data.device)
        return self._data[idx]


class _MockPairedSampler:
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        device: torch.device,
        pool: int = _POOL,
    ) -> None:
        self._x = torch.randn(pool, x_dim, device=device)
        self._y = torch.randn(pool, y_dim, device=device)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = torch.randint(0, self._x.shape[0], (batch_size,), device=self._x.device)
        return self._x[idx], self._y[idx]


def _base_cfg(method: str) -> DictConfig:
    return OmegaConf.create(
        {
            "method": method,
            "dataset": {"x_dim": 2, "y_dim": 2},
            "train": {
                "batch_size": _BATCH_SIZE,
                "z_dim": 1,
                "num_timesteps": 1,
                "layers_g": [32, 32],
                "layers_d": [32, 32],
                "lr_g": 1e-3,
                "lr_d": 3e-4,
                "beta1": 0.5,
                "beta2": 0.9,
                "r1_gamma": 0.01,
                "lazy_reg": 1,
                "lr": 3e-4,
                "weight_decay": 0.01,
            },
            "model": {
                "input_size": 2,
                "hidden_size": 32,
                "num_hidden_layers": 2,
                "features": 2,
                "context_features": 2,
                "hidden_features": [16],
                "hidden_context_features": 32,
                "num_layers": 2,
                "num_blocks_per_layer": 1,
            },
        }
    )


def _samplers(x_dim: int = 2, y_dim: int = 2):
    usd = _MockUnpairedSampler(x_dim, _DEVICE)
    utd = _MockUnpairedSampler(y_dim, _DEVICE)
    pd = _MockPairedSampler(x_dim, y_dim, _DEVICE)
    return usd, utd, pd


@pytest.mark.parametrize(
    "method,expected_keys",
    [
        ("cgan", {"d_loss", "g_loss"}),
        ("ugan", {"d_loss", "g_loss"}),
        ("regression", {"loss"}),
        ("cnf", {"loss"}),
        ("cnf_semi", {"loss", "paired_nll", "unp_loss"}),
    ],
)
def test_baseline_train_step(method: str, expected_keys: set[str]) -> None:
    cfg = _base_cfg(method)
    trainer = baseline_registry[method](cfg, _DEVICE)
    trainer.build_optimizers(cfg)
    usd, utd, pd = _samplers()
    metrics = trainer.train_step(0, usd, utd, pd)
    assert set(metrics) == expected_keys
    assert all(isinstance(v, float) for v in metrics.values())
    assert trainer.state_dict()

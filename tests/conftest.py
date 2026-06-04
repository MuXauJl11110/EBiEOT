import os
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict

from src.ebieot.costs.base import BaseLSECost
from src.ebieot.ebieot_gmm import EbieotGmm


class FixedLSECost(BaseLSECost):
    """Constant :math:`b_m` and ``log v_m`` for every ``x`` (tests only)."""

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        m_potentials: int,
        epsilon: float,
        *,
        generator: torch.Generator | None = None,
    ):
        super().__init__(x_dim, y_dim, m_potentials, epsilon)
        gen = generator or torch.Generator()
        self.register_buffer("_b_m", torch.randn(m_potentials, y_dim, generator=gen))
        self.register_buffer(
            "_log_v_m",
            torch.log_softmax(torch.randn(m_potentials, generator=gen), dim=0),
        )

    def compute_b_m(self, x: torch.Tensor) -> torch.Tensor:
        return self._b_m

    def compute_log_v_m(self, x: torch.Tensor) -> torch.Tensor:
        return self._log_v_m


@pytest.fixture
def x_dim() -> int:
    return 4


@pytest.fixture
def y_dim() -> int:
    return 3


@pytest.fixture
def n_potentials() -> int:
    return 2


@pytest.fixture
def m_potentials() -> int:
    return 2


@pytest.fixture
def batch_size() -> int:
    return 5


@pytest.fixture
def epsilon() -> float:
    return 1.0


@pytest.fixture
def cost(x_dim: int, y_dim: int, m_potentials: int, epsilon: float) -> FixedLSECost:
    g = torch.Generator().manual_seed(0)
    return FixedLSECost(x_dim, y_dim, m_potentials, epsilon, generator=g)


@pytest.fixture
def model(
    y_dim: int,
    n_potentials: int,
    cost: FixedLSECost,
    epsilon: float,
) -> EbieotGmm:
    torch.manual_seed(1)
    return EbieotGmm(
        y_dim=y_dim,
        n_potentials=n_potentials,
        cost=cost,
        epsilon=epsilon,
        sampling_batch_size=128,
        A_diagonal_init=0.1,
    )


@pytest.fixture
def batched_x(batch_size: int, x_dim: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(2)
    return torch.randn(batch_size, x_dim, generator=g)


@pytest.fixture
def batched_y(batch_size: int, y_dim: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(3)
    return torch.randn(batch_size, y_dim, generator=g)


# --- Hydra compose fixtures (see lightning-hydra-template/tests/conftest.py) ---


@pytest.fixture(scope="session")
def conf_dir() -> str:
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, "conf")
    )


@pytest.fixture(scope="package")
def cfg_hydra_global(conf_dir: str) -> DictConfig:
    """Default composed config for Hydra tests (quiet extras, no logger)."""
    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=egeot_swiss_roll",
                "train.steps_to=2",
                "extras.print_config=false",
                "extras.enforce_tags=false",
            ],
        )

    with open_dict(cfg):
        cfg.extras.print_config = False
        cfg.extras.enforce_tags = False

    return cfg


@pytest.fixture(scope="function")
def cfg_hydra(cfg_hydra_global: DictConfig, tmp_path: Path) -> DictConfig:
    """Per-test copy with writable ``paths.output_dir`` / ``paths.log_dir``."""
    cfg = cfg_hydra_global.copy()

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()

import importlib
from collections.abc import Iterator, Mapping
from typing import Callable

from omegaconf import DictConfig

from src.baselines.data import build_baseline_samplers
from src.baselines.protocol import BaselineTrainer

BaselineTrainerFactory = Callable[[DictConfig, object], BaselineTrainer]

_TRAINER_SPECS: dict[str, tuple[str, str]] = {
    "cgan": ("src.baselines.cgan", "CganTrainer"),
    "ugan": ("src.baselines.ugan", "UganTrainer"),
    "regression": ("src.baselines.regression", "RegressionTrainer"),
    "cnf": ("src.baselines.cnf", "CnfTrainer"),
    "cnf_semi": ("src.baselines.cnf_semi", "CnfSemiTrainer"),
}

_LOADED: dict[str, BaselineTrainerFactory] = {}


def _get_trainer_factory(method: str) -> BaselineTrainerFactory:
    if method not in _TRAINER_SPECS:
        raise KeyError(method)
    if method not in _LOADED:
        module_name, class_name = _TRAINER_SPECS[method]
        module = importlib.import_module(module_name)
        _LOADED[method] = getattr(module, class_name)
    return _LOADED[method]


class _LazyBaselineRegistry(Mapping[str, BaselineTrainerFactory]):
    def __getitem__(self, key: str) -> BaselineTrainerFactory:
        return _get_trainer_factory(key)

    def __iter__(self) -> Iterator[str]:
        return iter(_TRAINER_SPECS)

    def __len__(self) -> int:
        return len(_TRAINER_SPECS)

    def get(
        self, key: str, default: BaselineTrainerFactory | None = None
    ) -> BaselineTrainerFactory | None:
        if key not in _TRAINER_SPECS:
            return default
        return _get_trainer_factory(key)

    def __contains__(self, key: object) -> bool:
        return key in _TRAINER_SPECS


baseline_registry: Mapping[str, BaselineTrainerFactory] = _LazyBaselineRegistry()


def __getattr__(name: str):
    for method, (_, class_name) in _TRAINER_SPECS.items():
        if name == class_name:
            return _get_trainer_factory(method)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaselineTrainer",
    "baseline_registry",
    "build_baseline_samplers",
    "CganTrainer",
    "UganTrainer",
    "RegressionTrainer",
    "CnfTrainer",
    "CnfSemiTrainer",
]

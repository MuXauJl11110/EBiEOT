from functools import partial
from typing import Any, Callable

import torch.nn as nn
from omegaconf import DictConfig, OmegaConf


def default_active() -> Callable[[], nn.Module]:
    return partial(nn.LeakyReLU, 0.2)


def cfg_int(node: Any, key: str, default: int) -> int:
    if node is None or not OmegaConf.is_config(node):
        return default
    return int(OmegaConf.select(node, key, default=default))


def cfg_float(node: Any, key: str, default: float) -> float:
    if node is None or not OmegaConf.is_config(node):
        return default
    return float(OmegaConf.select(node, key, default=default))


def cfg_list(node: Any, key: str, default: list) -> list:
    if node is None or not OmegaConf.is_config(node):
        return list(default)
    value = OmegaConf.select(node, key, default=default)
    return list(value) if value is not None else list(default)


def batch_size(cfg: DictConfig) -> int:
    return cfg_int(cfg.train, "batch_size", 128)


def x_dim(cfg: DictConfig) -> int:
    return cfg_int(cfg.dataset, "x_dim", 2)


def y_dim(cfg: DictConfig) -> int:
    return cfg_int(cfg.dataset, "y_dim", 2)

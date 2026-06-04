"""Reusable weather notebook helpers for config, data prep, and samplers."""


import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from src.utils.core.seed import set_seed
from src.utils.samplers.data import DatasetSampler, PairedTensorBatchSampler
from src.utils.training.swiss_roll_notebook import CometExperiment, normalize_experiment_key

__all__ = [
    "CometExperiment",
    "build_weather_samplers",
    "compose_weather_cfg",
    "load_weather_tensors",
    "paired_sampler_weather",
    "unpaired_sampler_weather",
]


def compose_weather_cfg(
    repo_root: str,
    experiment: str,
    overrides: list[str],
    aliases: dict[str, str] | None = None,
):
    conf_dir = os.path.abspath(os.path.join(repo_root, "conf"))
    experiment_key = normalize_experiment_key(experiment, aliases=aliases)
    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        cfg = compose(
            config_name="config",
            overrides=[f"experiment={experiment_key}", *overrides],
        )
    seed = int(cfg.seed) if cfg.get("seed") is not None else int(cfg.train.seed)
    set_seed(seed)
    return cfg, experiment_key, seed


def load_weather_tensors(
    data_root: str | Path,
    *,
    n_chosen_locations: int = 200,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, list[np.ndarray]]:
    """Load and split weather tensors (same logic as archived notebooks)."""
    root = Path(data_root)
    data = np.load(root / "X_num.npy")
    data = np.stack([d for d in data if sum(np.isnan(d)) == 0])
    data_csv = pd.read_csv(root / "csv" / "X_num.csv")
    target = np.load(root / "Y.npy")
    meta = np.load(root / "X_meta.npy")
    meta = np.stack([meta[i] for i, d in enumerate(data) if sum(np.isnan(d)) == 0])

    data_new = np.concatenate((data, meta[:, -2].reshape(-1, 1)), axis=1)

    dict_location_src: dict[float, list[np.ndarray]] = {}
    for d in data_new:
        if d[-2] == 1.0:
            d_new = d[:-7]
            dict_location_src.setdefault(d[-1], []).append(d_new)

    dict_location_src_new: dict[float, np.ndarray] = {}
    for key, item in dict_location_src.items():
        item_arr = np.stack(item)
        if item_arr.shape[0] > 1:
            item_arr = (item_arr - np.min(item_arr, axis=0)) / (
                np.max(item_arr, axis=0) - np.min(item_arr, axis=0) + 1e-1
            )
            dict_location_src_new[key] = item_arr
    dict_location_src = dict_location_src_new

    dict_location_trg: dict[float, list[np.ndarray]] = {}
    for d in data_new:
        if d[-2] == 6.0:
            d_new = d[:-7]
            dict_location_trg.setdefault(d[-1], []).append(d_new)

    dict_location_trg_new: dict[float, np.ndarray] = {}
    for key, item in dict_location_trg.items():
        item_arr = np.stack(item)
        if item_arr.shape[0] > 1:
            item_arr = (item_arr - np.min(item_arr, axis=0)) / (
                np.max(item_arr, axis=0) - np.min(item_arr, axis=0) + 1e-1
            )
            dict_location_trg_new[key] = item_arr
    dict_location_trg = dict_location_trg_new

    chosen_locs = list(dict_location_trg.keys())[:n_chosen_locations]
    x_pair_orig, y_pair_orig = [], []
    for key in dict_location_src.keys():
        if key not in chosen_locs:
            continue
        item_src = dict_location_src[key]
        x = np.concatenate([np.mean(item_src, axis=0), np.std(item_src, axis=0)])
        x_pair_orig.append(x)
        y_pair_orig.append(dict_location_trg[key])

    x_pair_orig = np.stack(x_pair_orig)

    x_orig = []
    for key in dict_location_src.keys():
        if key in chosen_locs:
            continue
        item_src = dict_location_src[key]
        x = np.concatenate([np.mean(item_src, axis=0), np.std(item_src, axis=0)])
        x_orig.append(x)
    x_orig = np.stack(x_orig)

    y_orig = []
    for key in dict_location_trg.keys():
        if key in chosen_locs:
            continue
        y_orig.append(dict_location_trg[key])

    return x_orig, y_orig, x_pair_orig, y_pair_orig


def paired_sampler_weather(
    x_pair: np.ndarray,
    y_pair: list[np.ndarray],
    b_size: int,
    *,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    idxs = np.random.randint(low=0, high=len(x_pair) - 1, size=b_size)
    x_pair_batch = torch.tensor(x_pair[idxs], device=device, dtype=dtype)
    y_pair_batch = np.stack(
        [y_pair[idx][random.randint(0, len(y_pair[idx]) - 1)] for idx in idxs]
    )
    y_pair_batch = torch.tensor(y_pair_batch, device=device, dtype=dtype)
    return x_pair_batch, y_pair_batch


def unpaired_sampler_weather(
    x: np.ndarray,
    y: list[np.ndarray],
    b_size: int,
    *,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    idxs = np.random.randint(low=0, high=len(x) - 1, size=b_size)
    idxs_y = np.array([len(x) - idx - 1 for idx in idxs])
    x_batch = torch.tensor(x[idxs], device=device, dtype=dtype)
    y_batch = np.stack(
        [y[idx][random.randint(0, len(y[idx]) - 1)] for idx in idxs_y]
    )
    y_batch = torch.tensor(y_batch, device=device, dtype=dtype)
    return x_batch, y_batch


def build_weather_samplers(
    x_unpaired: torch.Tensor,
    y_unpaired: torch.Tensor,
    x_paired: torch.Tensor,
    y_paired: torch.Tensor,
    *,
    device: str | torch.device,
) -> tuple[DatasetSampler, DatasetSampler, PairedTensorBatchSampler]:
    dev = str(device)
    return (
        DatasetSampler(x_unpaired, device=dev),
        DatasetSampler(y_unpaired, device=dev),
        PairedTensorBatchSampler(x_paired, y_paired),
    )

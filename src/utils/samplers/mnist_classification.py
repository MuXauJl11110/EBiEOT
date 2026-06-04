"""DataLoader builders for MNIST semi-supervised classification."""


import random
from dataclasses import dataclass

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset

from src.utils.datasets.mnist_classification import (
    build_mnist_datasets,
    build_paired_marginal_val_indices,
    indices_by_class_from_dataset,
    val_per_class_from_paired,
)


@dataclass
class MnistClassificationLoaders:
    paired: DataLoader
    marginal: DataLoader
    val: DataLoader
    test: DataLoader
    train_dataset: torch.utils.data.Dataset
    num_classes: int
    in_channels: int
    image_size: int


def build_mnist_classification_loaders(
    cfg: DictConfig,
    *,
    paired_per_class: int | None = None,
    unpaired_per_class: int | None = None,
    seed: int | None = None,
) -> MnistClassificationLoaders:
    ds = cfg.dataset
    data_root = str(ds.data_root)
    paired_pc = int(
        paired_per_class if paired_per_class is not None else ds.paired_per_class
    )
    unpaired_pc = int(
        unpaired_per_class
        if unpaired_per_class is not None
        else ds.unpaired_per_class
    )
    val_ratio = float(ds.val_ratio)
    run_seed = int(seed if seed is not None else cfg.train.seed)

    train_dataset, test_dataset, num_classes, in_channels, image_size = (
        build_mnist_datasets(data_root)
    )
    by_class = indices_by_class_from_dataset(train_dataset, num_classes)
    val_pc = val_per_class_from_paired(paired_pc, val_ratio)

    rng = random.Random(run_seed + 10_000)
    paired_idx, marginal_idx, val_idx = build_paired_marginal_val_indices(
        by_class,
        paired_per_class=paired_pc,
        marginal_per_class=unpaired_pc,
        val_per_class=val_pc,
        rng=rng,
    )
    if len(marginal_idx) == 0:
        marginal_idx = paired_idx.copy()

    pin_mem = torch.cuda.is_available()
    num_workers = int(ds.get("num_workers", 0))

    def _loader(subset: Subset, batch_size: int, shuffle: bool) -> DataLoader:
        return DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_mem,
        )

    tcfg = cfg.train
    return MnistClassificationLoaders(
        paired=_loader(
            Subset(train_dataset, paired_idx),
            int(tcfg.paired_batch_size),
            True,
        ),
        marginal=_loader(
            Subset(train_dataset, marginal_idx),
            int(tcfg.unpaired_batch_size),
            True,
        ),
        val=_loader(
            Subset(train_dataset, val_idx),
            int(tcfg.get("eval_batch_size", tcfg.unpaired_batch_size)),
            False,
        ),
        test=_loader(
            test_dataset,
            int(tcfg.get("eval_batch_size", tcfg.unpaired_batch_size)),
            False,
        ),
        train_dataset=train_dataset,
        num_classes=num_classes,
        in_channels=in_channels,
        image_size=image_size,
    )

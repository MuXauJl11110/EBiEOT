"""MNIST loaders and semi-supervised index splits for energy-based classification."""


import gzip
import math
import os
import random
import struct
import typing as tp
import urllib.request
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
MNIST_URL_BASE = "https://storage.googleapis.com/cvdf-datasets/mnist"


class TensorLabelDataset(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        return int(self.images.size(0))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx], self.labels[idx]

    @property
    def targets(self) -> torch.Tensor:
        return self.labels


def read_idx_images_gz(path: str) -> torch.Tensor:
    with gzip.open(path, "rb") as f:
        header = f.read(16)
        magic, num, rows, cols = struct.unpack(">IIII", header)
        if magic != 2051:
            raise ValueError(f"Invalid image IDX magic number in {path}: {magic}")
        data = f.read()
    arr = np.frombuffer(data, dtype=np.uint8).reshape(num, rows, cols)
    x = torch.from_numpy(arr.copy()).float().unsqueeze(1) / 255.0
    x = (x - 0.1307) / 0.3081
    return x


def read_idx_labels_gz(path: str) -> torch.Tensor:
    with gzip.open(path, "rb") as f:
        header = f.read(8)
        magic, num = struct.unpack(">II", header)
        if magic != 2049:
            raise ValueError(f"Invalid label IDX magic number in {path}: {magic}")
        data = f.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    if arr.shape[0] != num:
        raise ValueError(
            f"Label count mismatch in {path}: header={num}, actual={arr.shape[0]}"
        )
    return torch.from_numpy(arr.copy()).long()


def maybe_download_mnist_files(data_root: str) -> None:
    os.makedirs(data_root, exist_ok=True)
    for filename in MNIST_FILES.values():
        path = os.path.join(data_root, filename)
        if os.path.exists(path):
            continue
        url = f"{MNIST_URL_BASE}/{filename}"
        print(f"Downloading {url} -> {path}")
        try:
            urllib.request.urlretrieve(url, path)
        except Exception as exc:
            raise RuntimeError(
                "Could not download MNIST files. Either enable network access or place "
                f"these files manually in {data_root}: {', '.join(MNIST_FILES.values())}"
            ) from exc


def build_mnist_datasets(
    data_root: str,
) -> tuple[TensorLabelDataset, TensorLabelDataset, int, int, int]:
    needed = [os.path.join(data_root, n) for n in MNIST_FILES.values()]
    if not all(os.path.exists(p) for p in needed):
        maybe_download_mnist_files(data_root)
    train_images = read_idx_images_gz(
        os.path.join(data_root, MNIST_FILES["train_images"])
    )
    train_labels = read_idx_labels_gz(
        os.path.join(data_root, MNIST_FILES["train_labels"])
    )
    test_images = read_idx_images_gz(
        os.path.join(data_root, MNIST_FILES["test_images"])
    )
    test_labels = read_idx_labels_gz(
        os.path.join(data_root, MNIST_FILES["test_labels"])
    )
    train = TensorLabelDataset(train_images, train_labels)
    test = TensorLabelDataset(test_images, test_labels)
    return train, test, 10, 1, 28


def indices_by_class_from_dataset(
    dataset: tp.Any, num_classes: int
) -> dict[int, list[int]]:
    if hasattr(dataset, "targets"):
        labels = np.array(dataset.targets)
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])
    return {c: np.where(labels == c)[0].tolist() for c in range(num_classes)}


def build_paired_marginal_val_indices(
    indices_by_class: dict[int, list[int]],
    paired_per_class: int,
    marginal_per_class: int,
    val_per_class: int,
    rng: random.Random,
) -> tuple[list[int], list[int], list[int]]:
    paired: list[int] = []
    for _c, class_indices in indices_by_class.items():
        idxs = class_indices.copy()
        rng.shuffle(idxs)
        paired.extend(idxs[:paired_per_class])

    paired_set = set(paired)
    marginal: list[int] = []
    for _c, class_indices in indices_by_class.items():
        leftovers = [i for i in class_indices if i not in paired_set]
        rng.shuffle(leftovers)
        marginal.extend(leftovers[:marginal_per_class])

    used = paired_set | set(marginal)
    val: list[int] = []
    for _c, class_indices in indices_by_class.items():
        leftovers = [i for i in class_indices if i not in used]
        rng.shuffle(leftovers)
        val.extend(leftovers[:val_per_class])
    return paired, marginal, val


def val_per_class_from_paired(paired_per_class: int, val_ratio: float) -> int:
    return max(1, int(math.ceil(paired_per_class * val_ratio)))


class EarlyStopping:
    def __init__(self, patience: int = 6, mode: str = "max") -> None:
        self.patience = patience
        self.mode = mode
        self.best: float | None = None
        self.num_bad = 0
        self.should_stop = False

    def step(self, metric: float) -> None:
        if self.best is None:
            self.best = metric
            return
        improved = (metric > self.best) if self.mode == "max" else (metric < self.best)
        if improved:
            self.best = metric
            self.num_bad = 0
        else:
            self.num_bad += 1
        if self.num_bad >= self.patience:
            self.should_stop = True


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module, step: int, start_from: int = 10) -> None:
        if step < start_from:
            return
        for name, param in model.state_dict().items():
            self.shadow[name].mul_(self.decay)
            self.shadow[name].add_((1 - self.decay) * param.detach())

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow)


def normalize_arch(arch: str) -> str:
    key = arch.lower().replace("_embed", "")
    if key in ("mlp", "cnn"):
        return key
    if key == "mlp_embed":
        return "mlp"
    if key == "cnn_embed":
        return "cnn"
    raise ValueError(f"Unknown arch: {arch}. Expected mlp, cnn, mlp_embed, or cnn_embed.")


@torch.no_grad()
def evaluate_classification(
    model: tp.Any,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    loss_vals: list[float] = []
    joint_vals: list[float] = []
    marg_y_vals: list[float] = []
    logz_vals: list[float] = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        probs = model.posterior(x)
        preds = probs.argmax(dim=1)
        correct += int((preds == y).sum().item())
        total += int(x.size(0))
        parts = model.loss_with_parts(x, y, x)
        loss_vals.append(float(parts["loss"].item()))
        joint_vals.append(float(parts["jointTerm"].item()))
        marg_y_vals.append(float(parts["margYTerm"].item()))
        logz_vals.append(float(parts["logzTerm"].item()))
    return {
        "acc": (correct / total) if total > 0 else 0.0,
        "mean_loss": float(np.mean(loss_vals)) if loss_vals else 0.0,
        "mean_joint_term": float(np.mean(joint_vals)) if joint_vals else 0.0,
        "mean_marg_y_term": float(np.mean(marg_y_vals)) if marg_y_vals else 0.0,
        "mean_logz_term": float(np.mean(logz_vals)) if logz_vals else 0.0,
    }


def train_classification_epoch(
    model: tp.Any,
    paired_loader: DataLoader,
    marginal_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    ema: EMA | None,
    global_step: int,
    grad_clip: float,
) -> tuple[dict[str, float], int]:
    from itertools import cycle

    model.train()
    epoch_steps = max(1, max(len(paired_loader), len(marginal_loader)))
    paired_iter = cycle(paired_loader)
    marginal_iter = cycle(marginal_loader)
    running_loss = 0.0
    running_joint = 0.0
    running_margy = 0.0
    running_logz = 0.0

    for _ in range(epoch_steps):
        x_p, y_p = next(paired_iter)
        x_m, _ = next(marginal_iter)
        x_p = x_p.to(device)
        y_p = y_p.to(device)
        x_m = x_m.to(device)

        optimizer.zero_grad(set_to_none=True)
        parts = model.loss_with_parts(x_p, y_p, x_m)
        parts["loss"].backward()
        if grad_clip > 0 and math.isfinite(grad_clip):
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if ema is not None:
            ema.update(model, global_step)
        global_step += 1

        running_loss += float(parts["loss"].item())
        running_joint += float(parts["jointTerm"].item())
        running_margy += float(parts["margYTerm"].item())
        running_logz += float(parts["logzTerm"].item())

    return (
        {
            "train_loss": running_loss / epoch_steps,
            "train_joint": running_joint / epoch_steps,
            "train_marg_y": running_margy / epoch_steps,
            "train_logz": running_logz / epoch_steps,
        },
        global_step,
    )


def run_classification_training(
    model: tp.Any,
    paired_loader: DataLoader,
    marginal_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    *,
    epochs_max: int,
    patience: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    ema_decay: float,
    use_ema: bool = True,
) -> dict[str, tp.Any]:
    ema = EMA(model, decay=ema_decay) if use_ema else None
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    stopper = EarlyStopping(patience=patience, mode="max")
    best_val_acc = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    global_step = 0
    epoch_records: list[dict[str, tp.Any]] = []
    last_epoch = 0

    for epoch in range(1, epochs_max + 1):
        last_epoch = epoch
        train_metrics, global_step = train_classification_epoch(
            model,
            paired_loader,
            marginal_loader,
            optimizer,
            device,
            ema,
            global_step,
            grad_clip,
        )

        backup_state = {k: v.clone() for k, v in model.state_dict().items()}
        if ema is not None:
            ema.copy_to(model)
        val_metrics = evaluate_classification(model, val_loader, device)
        model.load_state_dict(backup_state)

        val_acc = val_metrics["acc"]
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = deepcopy(backup_state)

        stopper.step(val_acc)
        record = {
            "epoch": epoch,
            **train_metrics,
            "val_acc": float(val_acc),
            "val_loss": float(val_metrics["mean_loss"]),
            "stopped_here": False,
        }
        epoch_records.append(record)

        if stopper.should_stop:
            record["stopped_here"] = True
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate_classification(model, test_loader, device)

    return {
        "best_val_acc": float(best_val_acc),
        "test_acc": float(test_metrics["acc"]),
        "test_loss": float(test_metrics["mean_loss"]),
        "epochs_ran": last_epoch,
        "epoch_records": epoch_records,
    }

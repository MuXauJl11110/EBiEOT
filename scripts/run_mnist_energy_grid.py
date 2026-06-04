#!/usr/bin/env python3
"""Grid search for semi-supervised energy-based MNIST classification."""


import argparse
import os
import random
import sys
import time
import typing as tp
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.ebieot.classification_based import ClassificationBasedEBiEOT
from src.utils.datasets.mnist_classification import (
    build_mnist_datasets,
    build_paired_marginal_val_indices,
    indices_by_class_from_dataset,
    normalize_arch,
    run_classification_training,
    val_per_class_from_paired,
)

try:
    from comet_ml import Experiment
except Exception:
    Experiment = None


CONFIG: dict[str, tp.Any] = {
    "data_root": "./data/mnist",
    "paired_grid": [20, 50],
    "unpaired_grid": [0, 200],
    "val_ratio": 0.25,
    "batch_paired": 64,
    "batch_marginal": 128,
    "batch_eval": 512,
    "epochs_max": 25,
    "patience": 6,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "grad_clip": 5.0,
    "epsilon": 0.5,
    "ema_decay": 0.999,
    "arch": "cnn",
    "mlp_embed_dim": 32,
    "cnn_embed_dim": 64,
    "mlp_hidden_dim": 256,
    "cnn_hidden_dim": 128,
    "seed": 42,
    "parallel_workers": 1,
    "num_workers": 0,
    "use_cuda": True,
    "log_to_comet": False,
    "results_csv": "grid_results.csv",
}


def parse_int_list(csv_values: str) -> list[int]:
    out: list[int] = []
    for raw in csv_values.split(","):
        raw = raw.strip()
        if raw:
            out.append(int(raw))
    return out


def maybe_create_experiment(config: dict[str, tp.Any], run_label: str):
    if not config["log_to_comet"]:
        return None
    if Experiment is None:
        print("Comet disabled: comet_ml not installed.")
        return None
    if not os.environ.get("COMET_API_KEY"):
        print("Comet disabled: COMET_API_KEY not set.")
        return None
    exp = Experiment(
        project_name="mnist-energy-grid",
        parse_args=False,
        auto_output_logging="simple",
    )
    exp.set_name(run_label)
    return exp


def build_model(
    config: dict[str, tp.Any], num_classes: int, in_channels: int, image_size: int
) -> ClassificationBasedEBiEOT:
    arch = normalize_arch(config["arch"])
    return ClassificationBasedEBiEOT(
        num_classes=num_classes,
        in_channels=in_channels,
        image_size=image_size,
        epsilon=float(config["epsilon"]),
        arch=arch,
        mlp_embed_dim=int(config["mlp_embed_dim"]),
        cnn_embed_dim=int(config["cnn_embed_dim"]),
        mlp_hidden_dim=int(config["mlp_hidden_dim"]),
        cnn_hidden_dim=int(config["cnn_hidden_dim"]),
    )


def run_experiment_cell(args: dict[str, tp.Any]) -> dict[str, tp.Any]:
    paired_per_class = args["paired_per_class"]
    unpaired_per_class = args["unpaired_per_class"]
    config = args["config"]
    run_id = args["run_id"]
    seed = args["seed"]
    device_name = args["device"]

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device(device_name)

    train_dataset, test_dataset, num_classes, in_channels, image_size = (
        build_mnist_datasets(config["data_root"])
    )
    by_class = indices_by_class_from_dataset(train_dataset, num_classes)
    val_per_class = val_per_class_from_paired(
        paired_per_class, float(config["val_ratio"])
    )

    rng = random.Random(seed + 10_000 + run_id)
    paired_idx, marginal_idx, val_idx = build_paired_marginal_val_indices(
        by_class,
        paired_per_class=paired_per_class,
        marginal_per_class=unpaired_per_class,
        val_per_class=val_per_class,
        rng=rng,
    )
    if len(marginal_idx) == 0:
        marginal_idx = paired_idx.copy()

    pin_mem = bool(config["use_cuda"] and torch.cuda.is_available())
    nw = int(config["num_workers"])

    def _loader(indices: list[int], batch_size: int, shuffle: bool) -> DataLoader:
        return DataLoader(
            Subset(train_dataset, indices),
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False,
            num_workers=nw,
            pin_memory=pin_mem,
        )

    paired_loader = _loader(paired_idx, int(config["batch_paired"]), True)
    marginal_loader = _loader(marginal_idx, int(config["batch_marginal"]), True)
    val_loader = _loader(val_idx, int(config["batch_eval"]), False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(config["batch_eval"]),
        shuffle=False,
        num_workers=nw,
        pin_memory=pin_mem,
    )

    model = build_model(config, num_classes, in_channels, image_size).to(device)

    run_label = (
        f"mnist-{config['arch']}-p{paired_per_class}-u{unpaired_per_class}"
        f"-seed{seed}-run{run_id}"
    )
    experiment = maybe_create_experiment(config, run_label)
    if experiment is not None:
        experiment.log_parameters(
            {
                "paired_per_class": paired_per_class,
                "unpaired_per_class": unpaired_per_class,
                "seed": seed,
                "epsilon": config["epsilon"],
                "arch": config["arch"],
            }
        )

    result = run_classification_training(
        model,
        paired_loader,
        marginal_loader,
        val_loader,
        test_loader,
        device,
        epochs_max=int(config["epochs_max"]),
        patience=int(config["patience"]),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
        grad_clip=float(config["grad_clip"]),
        ema_decay=float(config["ema_decay"]),
        use_ema=True,
    )

    if experiment is not None:
        experiment.log_metric("test_acc", result["test_acc"])
        experiment.log_metric("test_loss", result["test_loss"])
        experiment.end()

    epoch_records = result.pop("epoch_records", [])
    for rec in epoch_records:
        rec.update(
            {
                "run_id": run_id,
                "seed": seed,
                "arch": config["arch"],
                "paired_per_class": paired_per_class,
                "unpaired_per_class": unpaired_per_class,
            }
        )

    return {
        "run_id": run_id,
        "seed": seed,
        "arch": config["arch"],
        "paired_per_class": paired_per_class,
        "unpaired_per_class": unpaired_per_class,
        "best_val_acc": result["best_val_acc"],
        "test_acc": result["test_acc"],
        "test_loss": result["test_loss"],
        "epochs_ran": result["epochs_ran"],
        "epoch_records": epoch_records,
    }


def launch_grid_search(config: dict[str, tp.Any]) -> pd.DataFrame:
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device_name = (
        "cuda" if (config["use_cuda"] and torch.cuda.is_available()) else "cpu"
    )
    cells: list[dict[str, tp.Any]] = []
    run_id = 0
    for paired in config["paired_grid"]:
        for unpaired in config["unpaired_grid"]:
            run_id += 1
            cells.append(
                {
                    "paired_per_class": paired,
                    "unpaired_per_class": unpaired,
                    "config": config,
                    "device": device_name,
                    "run_id": run_id,
                    "seed": seed + run_id * 11,
                }
            )

    print(
        f"Prepared {len(cells)} runs with arch={config['arch']} "
        f"(device={device_name}, workers={config['parallel_workers']})."
    )

    if int(config["parallel_workers"]) == 1:
        raw_results = [run_experiment_cell(c) for c in cells]
    else:
        with Pool(processes=int(config["parallel_workers"])) as pool:
            raw_results = pool.map(run_experiment_cell, cells)

    epoch_rows: list[dict[str, tp.Any]] = []
    results: list[dict[str, tp.Any]] = []
    for row in raw_results:
        epoch_rows.extend(row.pop("epoch_records", []))
        results.append(row)

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(config["results_csv"]) or ".", exist_ok=True)
    df.to_csv(config["results_csv"], index=False)
    print(f"Saved results to {config['results_csv']}")
    if epoch_rows:
        epoch_csv = config["results_csv"].replace(".csv", "_epochs.csv")
        pd.DataFrame(epoch_rows).to_csv(epoch_csv, index=False)
        print(f"Saved epoch metrics to {epoch_csv}")
    return df


def build_config_from_args() -> dict[str, tp.Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--arch", type=str, default="cnn_embed")
    parser.add_argument("--paired-grid", type=str, default="20,50")
    parser.add_argument("--unpaired-grid", type=str, default="0,200")
    parser.add_argument("--epochs-max", type=int, default=25)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epsilon", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--parallel-workers", type=int, default=1)
    parser.add_argument("--data-root", type=str, default="./data/mnist")
    parser.add_argument("--log-to-comet", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="Force CPU.")
    args = parser.parse_args()

    config = dict(CONFIG)
    config["run_name"] = args.run_name
    config["arch"] = args.arch
    config["paired_grid"] = parse_int_list(args.paired_grid)
    config["unpaired_grid"] = parse_int_list(args.unpaired_grid)
    config["epochs_max"] = args.epochs_max
    config["patience"] = args.patience
    config["lr"] = args.lr
    config["epsilon"] = args.epsilon
    config["seed"] = args.seed
    config["parallel_workers"] = args.parallel_workers
    config["data_root"] = args.data_root
    config["log_to_comet"] = bool(args.log_to_comet)
    if args.cpu:
        config["use_cuda"] = False
    arch_slug = normalize_arch(args.arch)
    config["results_csv"] = (
        f"./results/grid_results_seed_{args.seed}_{args.run_name}_{arch_slug}.csv"
    )
    return config


if __name__ == "__main__":
    cfg = build_config_from_args()
    t0 = time.time()
    df_out = launch_grid_search(cfg)
    elapsed = time.time() - t0
    print(f"Elapsed: {elapsed:.1f}s")
    cols = [
        "arch",
        "paired_per_class",
        "unpaired_per_class",
        "best_val_acc",
        "test_acc",
        "epochs_ran",
    ]
    print(df_out[cols].sort_values(["paired_per_class", "unpaired_per_class"]))

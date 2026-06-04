import os

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.utils.samplers.base import Sampler
from src.utils.samplers.data import PairedLoaderSampler
from src.utils.samplers.discrete_ot import OTPlanSampler


def generate_paired_data(
    X_sampler: Sampler,
    Y_sampler: Sampler,
    mini_batch_sampler: OTPlanSampler,
    num_samples: int,
    save_dir: str,
    file_postfix: str,
    mini_batch_size: int = 64,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not os.path.exists(os.path.join(save_dir, f"X_paired_train_{file_postfix}.pt")):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        X_paired_list, Y_paired_list = [], []

        for _ in tqdm(
            range(2 * num_samples)
        ):  # the first part for train, another for test
            _X_paired, _Y_paired = X_sampler.sample(mini_batch_size), Y_sampler.sample(
                mini_batch_size
            )
            _X_paired, _Y_paired = mini_batch_sampler.sample_plan(_X_paired, _Y_paired)
            X_paired_list.append(_X_paired[0])
            Y_paired_list.append(_Y_paired[0])

        X_paired, Y_paired = torch.stack(X_paired_list), torch.stack(Y_paired_list)

        torch.save(
            X_paired[:num_samples],
            os.path.join(save_dir, f"X_paired_train_{file_postfix}.pt"),
        )
        torch.save(
            Y_paired[:num_samples],
            os.path.join(save_dir, f"Y_paired_train_{file_postfix}.pt"),
        )
        torch.save(
            X_paired[num_samples:],
            os.path.join(save_dir, f"X_paired_test_{file_postfix}.pt"),
        )
        torch.save(
            Y_paired[num_samples:],
            os.path.join(save_dir, f"Y_paired_test_{file_postfix}.pt"),
        )

        X_paired_train = X_paired[:num_samples]
        Y_paired_train = Y_paired[:num_samples]
        X_paired_test = X_paired[num_samples:]
        Y_paired_test = Y_paired[num_samples:]
    else:
        X_paired_train = torch.load(
            os.path.join(save_dir, f"X_paired_train_{file_postfix}.pt"),
            map_location=device,
            weights_only=True,
        )
        Y_paired_train = torch.load(
            os.path.join(save_dir, f"Y_paired_train_{file_postfix}.pt"),
            map_location=device,
            weights_only=True,
        )
        X_paired_test = torch.load(
            os.path.join(save_dir, f"X_paired_test_{file_postfix}.pt"),
            map_location=device,
            weights_only=True,
        )
        Y_paired_test = torch.load(
            os.path.join(save_dir, f"Y_paired_test_{file_postfix}.pt"),
            map_location=device,
            weights_only=True,
        )

    return X_paired_train, Y_paired_train, X_paired_test, Y_paired_test


def get_paired_sampler(
    X_paired: torch.Tensor,
    Y_paired: torch.Tensor,
    batch_size: int,
    num_samples: int,
    device: str = "cuda",
) -> PairedLoaderSampler:
    assert len(X_paired) == len(Y_paired)
    loader_kwargs = {
        "num_workers": 0,
        "generator": torch.Generator(device=X_paired.device),
    }
    ind = torch.randperm(len(X_paired), device=X_paired.device)[:num_samples]
    paired_loader = DataLoader(
        TensorDataset(X_paired[ind], Y_paired[ind]),
        batch_size=min(batch_size, num_samples),
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    return PairedLoaderSampler(paired_loader, device=device)

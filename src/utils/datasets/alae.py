import os

import numpy as np
import torch


def get_latents(
    data_type: str,
    train_size: int = 60000,
    test_size: int = 10000,
    from_dir: str = "./datasets/FFHQ",
    dtype: torch.Type = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    latents = torch.from_numpy(np.load(os.path.join(from_dir, "latents.npy")))
    train_latents, test_latents = latents[:train_size], latents[train_size:]

    if data_type in {"MAN", "WOMAN"}:
        gender = np.load(os.path.join(from_dir, "gender.npy"))
        train_gender, test_gender = gender[:train_size], gender[train_size:]
    elif data_type in {"ADULT", "CHILDREN"}:
        age = torch.from_numpy(np.load(os.path.join(from_dir, "age.npy")))
        train_age, test_age = age[:train_size], age[train_size:]

    if data_type == "MAN":
        mask_train = torch.from_numpy((train_gender == "male").reshape(-1))
        mask_test = torch.from_numpy((test_gender == "male").reshape(-1))
    elif data_type == "WOMAN":
        mask_train = torch.from_numpy((train_gender == "female").reshape(-1))
        mask_test = torch.from_numpy((test_gender == "female").reshape(-1))
    elif data_type == "ADULT":
        mask_train = (train_age >= 18).reshape(-1) & (train_age != -1).reshape(-1)
        mask_test = (test_age >= 18).reshape(-1) & (test_age != -1).reshape(-1)
    elif data_type == "CHILDREN":
        mask_train = (train_age < 18).reshape(-1) & (train_age != -1).reshape(-1)
        mask_test = (test_age < 18).reshape(-1) & (test_age != -1).reshape(-1)
    else:
        raise ValueError(f"Unsupported data type: {data_type}")

    data_train = train_latents[mask_train]
    data_test = test_latents[mask_test]

    return data_train.to(dtype=dtype), data_test.to(dtype=dtype)

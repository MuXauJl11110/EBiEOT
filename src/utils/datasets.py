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
    latents = np.load(os.path.join(from_dir, "latents.npy"))
    train_latents, test_latents = latents[:train_size], latents[train_size:]

    if data_type in {"MAN", "WOMAN"}:
        gender = np.load(os.path.join(from_dir, "gender.npy"))
        train_gender, test_gender = gender[:train_size], gender[train_size:]
    elif data_type in {"ADULT", "CHILDREN"}:
        age = np.load(os.path.join(from_dir, "age.npy"))
        train_age, test_age = age[:train_size], age[train_size:]

    if data_type == "MAN":
        inds_train = np.arange(train_size)[(train_gender == "male").reshape(-1)]
        inds_test = np.arange(test_size)[(test_gender == "male").reshape(-1)]
    elif data_type == "WOMAN":
        inds_train = np.arange(train_size)[(train_gender == "female").reshape(-1)]
        inds_test = np.arange(test_size)[(test_gender == "female").reshape(-1)]
    elif data_type == "ADULT":
        inds_train = np.arange(train_size)[(train_age >= 18).reshape(-1) * (train_age != -1).reshape(-1)]
        inds_test = np.arange(test_size)[(test_age >= 18).reshape(-1) * (test_age != -1).reshape(-1)]
    elif data_type == "CHILDREN":
        inds_train = np.arange(train_size)[(train_age < 18).reshape(-1) * (train_age != -1).reshape(-1)]
        inds_test = np.arange(test_size)[(test_age < 18).reshape(-1) * (test_age != -1).reshape(-1)]
    data_train = train_latents[inds_train]
    data_test = test_latents[inds_test]

    return torch.tensor(data_train, dtype=dtype), torch.tensor(data_test, dtype=dtype)

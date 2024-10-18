import gc
import os
import sys

sys.path.append("..")

import random

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm

from src.auxiliary_models.generative import MLPnet
from src.samplers.from_dataset import DatasetSampler
from src.samplers.primary import StandardNormalSampler, SwissRollSampler
from src.utils.discrete_ot import OTPlanSampler
from src.utils.paired import generate_paired_data, get_GT_points, get_paired_sampler

if __name__ == "__main__":
    device = torch.device("cuda")
    os.system("wandb login <your token>")

    X_DIM = 2
    Y_DIM = 2
    assert X_DIM > 1
    assert Y_DIM > 1

    OUTPUT_SEED = 42

    N_POTENTIALS = 50
    M_POTENTIALS = 10
    EPSILON = 1.0
    INIT_BY_SAMPLES = True
    A_DIAGONAL_INIT = 0.1

    BATCH_SIZE = 128
    SAMPLING_BATCH_SIZE = 128

    D_LR_PAIRED = 3e-4  # 1e-3 for eps 0.1, 0.01 and 3e-4 for eps 0.002
    D_LR_UNPAIRED = 1e-3
    D_GRADIENT_MAX_NORM = float("inf")

    M_X_UNPAIRED_SAMPLES = 0
    N_Y_UNPAIRED_SAMPLES = 1024
    L_PAIRED_SAMPLES = 16000  # 128

    SAVE_EVERY = 25000
    MAX_STEPS = 250000
    CONTINUE = -1

    EXP_COST = "MLP"
    EXP_COST_INCLUDED = True
    MINIBATCH_COST = "rotation-v2"

    EMA_UPDATE = False

    Z_SIZE = 4
    EPS = 0.1

    torch.manual_seed(OUTPUT_SEED)
    np.random.seed(OUTPUT_SEED)

    torch.set_default_device(device)
    dtype = torch.float32
    torch.torch.set_default_dtype(dtype)

    EXP_META_INFO = ""
    EXP_NAME = (
        "MLP_Swiss_Roll_"
        + f"EPSILON_{EPSILON}_"
        + f"MAX_STEPS_{MAX_STEPS}_"
        + f"M_X_UNPAIRED_{M_X_UNPAIRED_SAMPLES}_"
        + f"N_Y_UNPAIRED_{N_Y_UNPAIRED_SAMPLES}_"
        + f"L_PAIRED_{L_PAIRED_SAMPLES}_"
        + f"LR_PAIRED_{D_LR_PAIRED}_"
        + f"MINIBATCH_COST_{MINIBATCH_COST}_"
        + EXP_META_INFO
    )

    config = dict(
        X_DIM=X_DIM,
        Y_DIM=Y_DIM,
        D_LR_PAIRED=D_LR_PAIRED,
        D_LR_UNPAIRED=D_LR_UNPAIRED,
        BATCH_SIZE=BATCH_SIZE,
        EPSILON=EPSILON,
        D_GRADIENT_MAX_NORM=D_GRADIENT_MAX_NORM,
        N_POTENTIALS=N_POTENTIALS,
        M_POTENTIALS=M_POTENTIALS,
        INIT_BY_SAMPLES=INIT_BY_SAMPLES,
        A_DIAGONAL_INIT=A_DIAGONAL_INIT,
        M_X_UNPAIRED_SAMPLES=M_X_UNPAIRED_SAMPLES,
        N_Y_UNPAIRED_SAMPELS=N_Y_UNPAIRED_SAMPLES,
        L_PAIRED_SAMPLES=L_PAIRED_SAMPLES,
    )

    wandb.init(name=EXP_NAME, project="inverse_ot", config=config)

    X_sampler = StandardNormalSampler(dim=2, device=device)
    Y_sampler = SwissRollSampler(dim=2, device=device, dtype=dtype)

    otp_sampler = OTPlanSampler("sinkhorn", cost_function=MINIBATCH_COST)

    data_dir = "./checkpoints/Tensors"
    file_postfix = f"{MINIBATCH_COST}_{L_PAIRED_SAMPLES}"

    X_paired_train, Y_paired_train, X_paired_test, Y_paired_test = generate_paired_data(
        X_sampler, Y_sampler, otp_sampler, L_PAIRED_SAMPLES, data_dir, file_postfix, device=device
    )

    pd_train_sampler = get_paired_sampler(X_paired_train, Y_paired_train, BATCH_SIZE, L_PAIRED_SAMPLES, device)

    X_unpaired_test = X_sampler.sample(L_PAIRED_SAMPLES)
    Y_unpaired_test = Y_sampler.sample(L_PAIRED_SAMPLES)

    if M_X_UNPAIRED_SAMPLES > 0:
        source_data = X_sampler.sample(M_X_UNPAIRED_SAMPLES)
        usd_sampler = DatasetSampler(source_data, device=device)  # usd - unpaired source data
    else:
        usd_sampler = DatasetSampler(X_paired_train, device=device)

    if N_Y_UNPAIRED_SAMPLES > 0:
        target_data = Y_sampler.sample(N_Y_UNPAIRED_SAMPLES)
        utd_sampler = DatasetSampler(target_data, device=device)  # utd - unpaired target data
    else:
        utd_sampler = DatasetSampler(Y_paired_train, device=device)

    starting_points = torch.tensor([[-2.0, 0.0], [2.0, 2.0], [-0.5, -0.75]])
    num_ending_points = 64

    num_starting_points_paired = 5
    indices = random.choices(range(L_PAIRED_SAMPLES), k=num_starting_points_paired)
    starting_points_paired = X_paired_train[indices]
    ending_points_paired = Y_paired_train[indices]

    gt_Y_points = get_GT_points(X_sampler, Y_sampler, otp_sampler, starting_points)

    T = MLPnet(input_size=2, hidden_size=256, num_hidden_layers=4).to(device)

    T_opt_paired = torch.optim.Adam(T.parameters(), lr=D_LR_PAIRED, weight_decay=0.01)

    OUTPUT_MODEL_PATH = "./checkpoints/models/reg_16k_full"
    if not os.path.exists(OUTPUT_MODEL_PATH):
        os.makedirs(OUTPUT_MODEL_PATH)

    for step in tqdm(range(CONTINUE + 1, MAX_STEPS)):

        T_opt_paired.zero_grad()
        X_paired, Y_paired = pd_train_sampler.sample(BATCH_SIZE)

        T_loss = F.mse_loss(Y_paired, T(X_paired))

        T_loss.backward()
        T_opt_paired.step()

        wandb.log({f"Loss": T_loss}, step=step)

        if (step + 1) % SAVE_EVERY == 0:
            torch.save(T.state_dict(), os.path.join(OUTPUT_MODEL_PATH, f"T_{step}.pt"))
        gc.collect()
        torch.cuda.empty_cache()

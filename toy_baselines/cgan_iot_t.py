import gc
import os
import sys

sys.path.append("..")

import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

import wandb
from src.models.models import MyCDiscriminator, MyCGenerator
from src.samplers.from_dataset import DatasetSampler
from src.samplers.primary import StandardNormalSampler, SwissRollSampler
from src.utils.discrete_ot import OTPlanSampler
from src.utils.paired import generate_paired_data, get_GT_points, get_paired_sampler


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


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

M_X_UNPAIRED_SAMPLES = 16000  # 1024
N_Y_UNPAIRED_SAMPLES = 0
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


EXP_META_INFO = ""
EXP_NAME = (
    "CGAN_Swiss_Roll_"
    + f"MAX_STEPS_{MAX_STEPS}_"
    + f"M_X_UNPAIRED_{M_X_UNPAIRED_SAMPLES}_"
    + f"L_PAIRED_{L_PAIRED_SAMPLES}_"
    + f"LR_PAIRED_{D_LR_PAIRED}_"
    + f"LR_UNPAIRED_{D_LR_UNPAIRED}_"
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


args = {
    "nz": 1,
    "num_timesteps": 1,
    "x_dim": 2,
    "t_dim": 2,
    "out_dim": 2,
    "beta_min": 0.1,
    "beta_max": 20.0,
    "layers_G": [256, 256, 256],
    "layers_D": [256, 256, 256],
    "num_iterations": 200000,
    "batch_size": 128,
    "lr_d": 1e-4,
    "lr_g": 1e-4,
    "beta1": 0.5,
    "beta2": 0.9,
    "r1_gamma": 0.01,
    "lazy_reg": 1,
    "use_ema": False,
    "ema_decay": 0.999,
    "sampler_precalc": 1000,
    "sampler_gen_params": {},
    "exp_path": "./swiss_roll/",
    "save_ckpt": True,
    "save_ckpt_every": 5000,
    "save_content": True,
    "save_content_every": 5000,
    "visualize": True,
    "visualize_every": 1000,
    "print": True,
    "print_every": 100,
    "resume": False,
}
args = dotdict(args)


if __name__ == "__main__":
    device = torch.device("cuda")
    os.system("wandb login <your token>")

    torch.set_default_device(device)
    dtype = torch.float32
    torch.torch.set_default_dtype(dtype)

    torch.manual_seed(OUTPUT_SEED)
    np.random.seed(OUTPUT_SEED)

    wandb.init(name=EXP_NAME, project="inverse_ot", config=config)

    X_sampler = StandardNormalSampler(dim=2, device=device)
    Y_sampler = SwissRollSampler(dim=2, device=device, dtype=dtype)

    otp_sampler = OTPlanSampler("sinkhorn", cost_function=MINIBATCH_COST)

    data_dir = "checkpoints/Tensors"
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

    batch_size = args.batch_size
    nz = args.nz  # latent dimension

    netG = MyCGenerator(
        x_dim=args.x_dim,
        t_dim=args.t_dim,
        n_t=args.num_timesteps,
        out_dim=args.out_dim,
        z_dim=nz,
        layers=args.layers_G,
    ).to(device)

    netD = MyCDiscriminator(x_dim=args.x_dim, t_dim=args.t_dim, n_t=args.num_timesteps, layers=args.layers_D).to(
        device
    )

    optimizerD = optim.Adam(netD.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))

    schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG, args.num_iterations, eta_min=1e-5)
    schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerD, args.num_iterations, eta_min=1e-5)

    history = {
        "D_loss": [],
        "G_loss": [],
    }
    history = dotdict(history)

    OUTPUT_MODEL_PATH = "./checkpoints/models/cgan_16k_full"
    if not os.path.exists(OUTPUT_MODEL_PATH):
        os.makedirs(OUTPUT_MODEL_PATH)

    # MAIN CYCLE

    for step in tqdm(range(CONTINUE + 1, MAX_STEPS)):
        #########################
        # Discriminator training
        #########################
        for p in netD.parameters():
            p.requires_grad = True

        netD.zero_grad()

        ###################################
        # Sample real data
        X_paired, Y_paired = pd_train_sampler.sample(BATCH_SIZE)

        ###################################
        # Sample timesteps
        t = torch.randint(0, args.num_timesteps, (X_paired.size(0),), device=device)

        Y_paired.requires_grad = True

        ###################################
        # Optimizing loss on real data
        D_real = netD(Y_paired, t, X_paired.detach())

        errD_real = F.softplus(-D_real)
        errD_real = errD_real.mean()

        errD_real.backward(retain_graph=True)

        ###################################
        # R_1(\phi) regularization
        if args.lazy_reg is None or step % args.lazy_reg == 0:
            grad_real = torch.autograd.grad(
                outputs=D_real.sum(),
                inputs=Y_paired,
                create_graph=True,
            )[0]
            grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()

            grad_penalty = args.r1_gamma / 2 * grad_penalty
            grad_penalty.backward()

        ###################################
        # Sample vector from latent space for generation
        latent_z = torch.randn(batch_size, nz, device=device)

        ###################################
        # Sample fake output
        x_0_predict = netG(X_paired.detach(), t, latent_z)

        ###################################
        # Optimize loss on fake data
        output = netD(x_0_predict, t, X_paired.detach()).view(-1)

        errD_fake = F.softplus(output)
        errD_fake = errD_fake.mean()
        errD_fake.backward()

        errD = errD_real + errD_fake

        history.D_loss.append(errD.item())
        wandb.log({f"D Loss": errD}, step=step)

        ###################################
        # Update weights of netD
        optimizerD.step()

        #############################################################

        #########################
        # Generator training
        #########################
        for p in netD.parameters():
            p.requires_grad = False
        netG.zero_grad()

        ###################################
        # Sample timesteps
        t = torch.randint(0, args.num_timesteps, (X_paired.size(0),), device=device)

        ###################################
        # Sample pairs for training
        unp_sample = usd_sampler.sample(BATCH_SIZE)

        ###################################
        # Sample vector from latent space for generation
        latent_z = torch.randn(batch_size, nz, device=device)

        ###################################
        # Sample fake output
        x_0_predict = netG(unp_sample.detach(), t, latent_z)

        ###################################
        # Optimize loss on fake data
        output = netD(x_0_predict, t, unp_sample.detach()).view(-1)

        ###################################
        # Update weights of netG
        errG = F.softplus(-output)
        errG = errG.mean()

        errG.backward()
        optimizerG.step()
        history.G_loss.append(errG.item())

        wandb.log({f"G Loss": errG}, step=step)
        # LR-Scheduling step
        schedulerG.step()
        schedulerD.step()

        if (step + 1) % SAVE_EVERY == 0:
            torch.save(netG.state_dict(), os.path.join(OUTPUT_MODEL_PATH, f"G_{step}.pt"))

        gc.collect()
        torch.cuda.empty_cache()

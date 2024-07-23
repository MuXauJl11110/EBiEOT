import logging
import os

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch import optim
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import wandb
from src.discrete_ot import OTPlanSampler
from src.distributions import PairedSampler, Sampler
from src.light_gcot import LightGCOT
from src.plotting import (
    plot_A_parameters,
    plot_B_parameters,
    plot_distributions,
    plot_gaussians,
)

logger = logging.getLogger(__name__)


def create_samplers(cfg: DictConfig, device: str) -> tuple[Sampler, Sampler]:
    X_sampler: Sampler = hydra.utils.instantiate(cfg.samplers.X, device=device)
    Y_sampler: Sampler = hydra.utils.instantiate(cfg.samplers.Y, device=device)

    return X_sampler, Y_sampler


def setup_accelerator(cfg: DictConfig) -> str:
    logger.info("Setting up accelerator...")
    device = device = (
        torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
        if cfg.experiment_params.device == "auto"
        else cfg.experiment_params.device
    )
    torch.set_default_device(device)
    if cfg.experiment_params.double_precision:
        torch.torch.set_default_dtype(torch.float64)
    torch.manual_seed(cfg.experiment_params.seed)
    np.random.seed(cfg.experiment_params.seed)
    return device


def setup_wandb_config(cfg: DictConfig) -> tuple[str, dict, str]:
    logger.info("Setting up wandb config...")
    exp_name = (
        f"{cfg.experiment_params.project_name}_{cfg.name}_"
        + f"EPSILON_{cfg.model.epsilon}_"
        + f"N_{cfg.model.n_potentials}_"
        + f"M_{cfg.model.m_potentials}_"
        + f"with_{cfg.model.cost_function}_"
        + f"cost_included_{cfg.experiment_params.cost_included}_"
        + f"N_PAIRED_{cfg.train_params.n_paired_samples}_"
        + f"M_UNPAIRED_{cfg.train_params.m_unpaired_samples}_"
        + cfg.train_params.meta_info
    )
    exp_config = dict(
        X_DIM=cfg.samplers.X.dim,
        Y_DIM=cfg.samplers.Y.dim,
        N_POTENTIALS=cfg.model.n_potentials,
        M_POTENTIALS=cfg.model.m_potentials,
        EPSILON=cfg.model.epsilon,
        A_DIAGONAL_INIT=cfg.model.A_diagonal_init,
        INIT_BY_SAMPLES=cfg.experiment_params.init_by_samples,
        BATCH_SIZE=cfg.train_params.batch_size,
        D_LR=cfg.optimizer.lr,
    )
    output_path = "../checkpoints/{}".format(exp_name)

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    return exp_name, exp_config, output_path


@hydra.main(config_path="./configs", config_name="grid_gaussians", version_base="1.3")
def main(cfg: DictConfig):
    known_experiments = {"Swiss_Roll", "Grid_Gaussians"}
    if cfg.name not in known_experiments:
        raise ValueError(f"Unknown experiment: {cfg.name}! Known experiments: {known_experiments}")

    device = setup_accelerator(cfg)
    exp_name, exp_config, output_path = setup_wandb_config(cfg)

    logger.info("Creating samplers...")
    X_sampler, Y_sampler = create_samplers(cfg, device)

    logger.info("Creating model...")
    model: LightGCOT = hydra.utils.instantiate(cfg.model)
    if cfg.experiment_params.init_by_samples:
        model.init_a_by_samples(Y_sampler.sample(model.n_potentials))

    logger.info("Setting up optimizer...")
    optimizer: optim.Optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    if cfg.train_params.begin > -1:
        model.load_state_dict(torch.load(os.path.join(output_path, f"D_{cfg.train_params.begin}.pt")))
        optimizer.load_state_dict(torch.load(os.path.join(output_path, f"D_opt_{cfg.train_params.begin}.pt")))

    if cfg.train_params.mini_batch_size is not None:
        logger.info("Setting up OT plan sampler...")
        otp_sampler: OTPlanSampler = hydra.utils.instantiate(cfg.plan_sampler)
    else:
        otp_sampler = None

    starting_points = torch.tensor([[-1.5, 1.5], [0.0, 0.0], [1.5, -1.5]])
    sampler = PairedSampler(
        X_sampler,
        Y_sampler,
        n_paired_samples=cfg.train_params.n_paired_samples,
        m_unpaired_samples=cfg.train_params.m_unpaired_samples,
        mini_batch_size=cfg.train_params.mini_batch_size,
        otp_sampler=otp_sampler,
    )
    wandb.init(name=exp_name, config=exp_config)
    with logging_redirect_tqdm():
        for step in tqdm(range(cfg.train_params.begin + 1, cfg.train_params.end)):
            # training loop
            optimizer.zero_grad()

            X, Y = sampler.sample()

            log_v_m = model.compute_log_v_m(X)  # [bs x M]
            b_m = model.compute_b_m(X)  # [bs x M x y_dim]

            log_w_n = model.compute_log_w_n()  # [N]
            a_n = model.compute_a_n()  # [N x y_dim]
            A_n = model.compute_A_n()  # [N x y_dim]

            f_c = model.compute_dual_potential(log_w_n, a_n, A_n, log_v_m, b_m)
            f = model.compute_primal_potential(Y, log_w_n, a_n, A_n)

            if cfg.experiment_params.cost_included:
                X_paired, Y_paired = sampler.sample_pair()
                log_v_m_paired = model.compute_log_v_m(X_paired)  # [bs x M]
                b_m_paired = model.compute_b_m(X_paired)  # [bs x M x y_dim]

                c = model.compute_cost(Y_paired, log_v_m_paired, b_m_paired)
                D_loss = c.mean() - (f_c + f).mean()
                D_loss.backward()
                wandb.log({r"$c(x, y)$": c.mean().item()}, step=step)
            else:
                D_loss = -(f_c + f).mean()
                D_loss.backward()
            D_gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=cfg.experiment_params.gradient_max_norm
            )
            optimizer.step()

            wandb.log({f"D gradient norm": D_gradient_norm.item()}, step=step)
            wandb.log({f"D_loss": D_loss.item()}, step=step)
            wandb.log({r"$-f^c(x)$": -f_c.mean().item()}, step=step)
            wandb.log({r"$-f(y)$": -f.mean().item()}, step=step)
            wandb.log({r"$-f(y)-f^c(x)$": -(f_c + f).mean().item()}, step=step)
            wandb.log({f"lam_min(A_n)": torch.min(A_n)}, step=step)
            wandb.log({f"lam_max(A_n)": torch.max(A_n)}, step=step)

            if step % cfg.train_params.plot_every == 0:
                A_dict = plot_A_parameters(model, log=True)
                B_dict = plot_B_parameters(model, starting_points, log=True)
                if cfg.name == "Swiss_Roll":
                    distr_dict = plot_distributions(
                        model, X_sampler, Y_sampler, X_paired, Y_paired, starting_points, log=True
                    )
                elif cfg.name == "Grid_Gaussians":
                    distr_dict = plot_gaussians(model, X_sampler, Y_sampler, X_paired, Y_paired, log=True)
                wandb.log(A_dict | B_dict | distr_dict)

            torch.save(model.state_dict(), os.path.join(output_path, f"D_{step}.pt"))
            torch.save(optimizer.state_dict(), os.path.join(output_path, f"D_opt_{step}.pt"))

    wandb.finish()


if __name__ == "__main__":
    main()

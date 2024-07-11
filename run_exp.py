import os

import click
import numpy as np
import torch
from tqdm import tqdm

import wandb
from src.discrete_ot import OTPlanSampler
from src.distributions import Sampler, StandardNormalSampler, SwissRollSampler
from src.light_gcot import LightGCOT
from src.plotting import plot_A_parameters, plot_B_parameters, plot_distributions

# Sampler parameters
x_dim_option = click.option("--x_dim", default=2)
y_dim_option = click.option("--y_dim", default=2)

# Model parameters
n_potentials_option = click.option("--n_potentials", default=500)
m_potentials_option = click.option("--m_potentials", default=500)
epsilon_option = click.option("--epsilon", default=0.002)
sampling_batch_size_option = click.option("--sampling_batch_size", default=128)
A_diagonal_init_option = click.option("--A_diagonal_init", default=0.1)
cost_function_option = click.option("--cost_function", default="MLP")

# Optimizer parameters
d_lr_option = click.option("--d_lr", default=3e-4)
d_gradient_max_norm_option = click.option("--d_gradient_max_norm", default=float("inf"))

# Mini-batch sampler parameters
method_option = click.option("--method", default="sinkhorn")
reg_option = click.option("--reg", default=0.01)
reg_m_option = click.option("--reg_m", default=0.01)
normalize_cost_option = click.option("--normalize_cost", default=False)

# Experiment parameters
init_by_samples_option = click.option("--init_by_samples", default=False)
cost_included_option = click.option("--cost_included", default=True)
meta_info_option = click.option("--meta_info", default="_")
double_precision_option = click.option("--double_precision", default=True)
mini_batch_option = click.option("--mini_batch", default=True)
seed_option = click.option("--seed", default=30)
begin_option = click.option("--begin", default=-1)
end_option = click.option("--end", default=20000)
batch_size_option = click.option("--batch_size", default=128)
plot_every_option = click.option("--plot_every", default=500)


@click.command()
@x_dim_option
@y_dim_option
@click.pass_context
def setup_samplers(ctx: click.Context, x_dim: int, y_dim: int) -> tuple[Sampler, Sampler]:
    X_sampler = StandardNormalSampler(dim=x_dim, device=ctx.obj["DEVICE"])
    Y_sampler = SwissRollSampler(dim=y_dim, device=ctx.obj["DEVICE"])

    return X_sampler, Y_sampler


@click.command()
@method_option
@reg_option
@reg_m_option
@normalize_cost_option
def setup_ot_plan_sampler(method: str, reg: float, reg_m: float, normalize_cost: bool) -> OTPlanSampler:
    return OTPlanSampler(method, reg, reg_m, normalize_cost)


@click.command()
@x_dim_option
@y_dim_option
@n_potentials_option
@m_potentials_option
@epsilon_option
@sampling_batch_size_option
@A_diagonal_init_option
@cost_function_option
def setup_model(
    x_dim: int,
    y_dim: int,
    n_potentials: int,
    m_potentials: int,
    epsilon: float,
    sampling_batch_size: int,
    A_diagonal_init: float,
    cost_function: str,
) -> LightGCOT:
    return LightGCOT(
        x_dim=x_dim,
        y_dim=y_dim,
        n_potentials=n_potentials,
        m_potentials=m_potentials,
        epsilon=epsilon,
        sampling_batch_size=sampling_batch_size,
        A_diagonal_init=A_diagonal_init,
        cost_function=cost_function,
    )


@click.command()
@d_lr_option
@click.pass_context
def setup_optimizer(ctx: click.Context, d_lr: float) -> torch.optim.Optimizer:
    model = ctx.obj["MODEL"]
    optimizer = torch.optim.Adam(model.parameters(), lr=d_lr)
    return optimizer


@click.command()
@x_dim_option
@y_dim_option
@n_potentials_option
@m_potentials_option
@epsilon_option
@A_diagonal_init_option
@init_by_samples_option
@batch_size_option
@d_lr_option
@d_gradient_max_norm_option
def setup_exp_config(
    x_dim: int,
    y_dim: int,
    n_potentials: int,
    m_potentials: int,
    epsilon: float,
    A_diagonal_init: float,
    init_by_samples: bool,
    batch_size: int,
    d_lr: float,
    d_gradient_max_norm: float,
) -> dict:
    return dict(
        X_DIM=x_dim,
        Y_DIM=y_dim,
        N_POTENTIALS=n_potentials,
        M_POTENTIALS=m_potentials,
        EPSILON=epsilon,
        A_DIAGONAL_INIT=A_diagonal_init,
        INIT_BY_SAMPLES=init_by_samples,
        BATCH_SIZE=batch_size,
        D_LR=d_lr,
        D_GRADIENT_MAX_NORM=d_gradient_max_norm,
    )


@click.command()
@n_potentials_option
@m_potentials_option
@epsilon_option
@cost_function_option
@cost_included_option
@meta_info_option
def setup_exp_name(
    n_potentials: int, m_potentials: int, epsilon: float, cost_function: str, cost_included: bool, meta_info: str
) -> str:
    return (
        f"LightGCOT_Swiss_Roll_EPSILON_{epsilon}_N_{n_potentials}_M_{m_potentials}_with_{cost_function}_cost_included_{cost_included}"
        + meta_info
    )


@click.command()
@double_precision_option
@mini_batch_option
@seed_option
@init_by_samples_option
@begin_option
@end_option
@cost_included_option
@batch_size_option
@plot_every_option
@d_gradient_max_norm_option
@click.pass_context
def run(
    ctx: click.Context,
    double_precision: bool,
    mini_batch: bool,
    seed: int,
    init_by_samples: bool,
    begin: int,
    end: int,
    cost_included: bool,
    batch_size: int,
    plot_every: int,
    d_gradient_max_norm: float,
):
    ctx.ensure_object(dict)
    # Setup device
    print("Setting up device...")
    device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    if double_precision:
        torch.torch.set_default_dtype(torch.float64)
    ctx.obj["DEVICE"] = device

    # Setup config
    print("Setting up config...")
    exp_name = setup_exp_name()
    exp_config = setup_exp_config()
    torch.manual_seed(seed)
    np.random.seed(seed)
    output_path = "../checkpoints/{}".format(exp_name)

    # Create samplers
    print("Creating samplers...")
    X_sampler, Y_sampler = setup_samplers()

    # Create model
    print("Creating model...")
    model: LightGCOT = setup_model()
    if init_by_samples:
        model.init_a_by_samples(Y_sampler.sample(model.n_potentials))
    ctx.obj["MODEL"] = model

    # Setup optimizer
    print("Setting up optimizer...")
    optimizer: torch.optim.Optimizer = setup_optimizer()
    if begin > -1:
        optimizer.load_state_dict(torch.load(os.path.join(output_path, f"D_opt_{begin}.pt")))

    # Mini batch sampler setup
    if mini_batch:
        print("Setting up OT plan sampler...", end=" ")
        otp_sampler = setup_ot_plan_sampler()

    starting_points = torch.tensor([[-1.5, 1.5], [0.0, 0.0], [1.5, -1.5]])

    wandb.init(name=exp_name, config=exp_config)
    print("Run...", end=" ")
    for step in tqdm(range(begin + 1, end)):
        # training loop
        optimizer.zero_grad()

        if mini_batch:
            _X, _Y = X_sampler.sample(batch_size), Y_sampler.sample(batch_size)
            X, Y = otp_sampler.sample_plan(_X, _Y)
        else:
            X, Y = X_sampler.sample(batch_size), Y_sampler.sample(batch_size)

        log_v_m = model.compute_log_v_m(X)  # [bs x M]
        b_m = model.compute_b_m(X)  # [bs x M x y_dim]

        log_w_n = model.compute_log_w_n()  # [N]
        a_n = model.compute_a_n()  # [N x y_dim]
        A_n = model.compute_A_n()  # [N x y_dim]

        f_c = model.compute_dual_potential(log_w_n, a_n, A_n, log_v_m, b_m)
        f = model.compute_primal_potential(Y, log_w_n, a_n, A_n)

        if cost_included:
            c = model.compute_cost(Y, log_v_m, b_m)
            D_loss = (c - f_c - f).mean()
            D_loss.backward()
            wandb.log({r"$c(x, y)$": c.mean().item()}, step=step)
        else:
            D_loss = -(f_c + f).mean()
            D_loss.backward()
        D_gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=d_gradient_max_norm)
        optimizer.step()

        wandb.log({f"D gradient norm": D_gradient_norm.item()}, step=step)
        wandb.log({f"D_loss": D_loss.item()}, step=step)
        wandb.log({r"$-f^c(x)$": -f_c.mean().item()}, step=step)
        wandb.log({r"$-f(y)$": -f.mean().item()}, step=step)
        wandb.log({r"$-f(y)-f^c(x)$": -(f_c + f).mean().item()}, step=step)
        wandb.log({f"lam_min(A_n)": torch.min(A_n)}, step=step)
        wandb.log({f"lam_max(A_n)": torch.max(A_n)}, step=step)

        if step % plot_every == 0:
            A_dict = plot_A_parameters(model, log=True)
            B_dict = plot_B_parameters(model, starting_points, log=True)
            distr_dict = plot_distributions(model, X_sampler, Y_sampler, starting_points, log=True)
            wandb.log(A_dict | B_dict | distr_dict)

        torch.save(model.state_dict(), os.path.join(output_path, f"D_{step}.pt"))
        torch.save(optimizer.state_dict(), os.path.join(output_path, f"D_opt_{step}.pt"))

    wandb.finish()


if __name__ == "__main__":
    # Sounds good, doesn't work
    run(obj={})

"""Reusable Swiss-roll notebook helpers for config, setup and logging."""


import os
from typing import Callable

import torch
from comet_ml import Experiment
from hydra import compose, initialize_config_dir

from src.utils.core.seed import set_seed
from src.utils.samplers.discrete_ot import OTPlanSampler
from src.utils.samplers.synthetic import (
    StandardNormalSampler,
    SwissRollSampler,
    build_swiss_roll_samplers,
)
from src.utils.training.helpers import compute_loss


class CometExperiment:
    """Adapter so plotting helpers can log figures and metrics to Comet."""

    def __init__(self, experiment: Experiment):
        self.experiment = experiment

    def log_figure(
        self,
        figure_name: str,
        figure,
        *,
        step: int | None = None,
    ) -> None:
        kwargs: dict = {"figure_name": figure_name, "figure": figure}
        if step is not None:
            kwargs["step"] = step
        self.experiment.log_figure(**kwargs)

    def log_metrics(self, metrics: dict, prefix: str = "") -> None:
        payload = {f"{prefix}{k}": v for k, v in metrics.items()}
        self.experiment.log_metrics(payload)


def normalize_experiment_key(experiment: str, aliases: dict[str, str] | None = None) -> str:
    aliases = aliases or {}
    return aliases.get(experiment, experiment.replace("-", "_"))


def compose_swiss_roll_cfg(
    repo_root: str,
    experiment: str,
    overrides: list[str],
    aliases: dict[str, str] | None = None,
):
    conf_dir = os.path.abspath(os.path.join(repo_root, "conf"))
    experiment_key = normalize_experiment_key(experiment, aliases=aliases)
    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        cfg = compose(
            config_name="config",
            overrides=[f"experiment={experiment_key}", *overrides],
        )
    seed = int(cfg.seed) if cfg.get("seed") is not None else int(cfg.train.seed)
    set_seed(seed)
    return cfg, experiment_key, seed


def infer_cost_label(cfg) -> str:
    target = str(cfg.ebieot.cost.get("_target_", "")).lower()
    if "shared" in target:
        return "SharedMLPLSE"
    if "mlplse" in target:
        return "MLPLSE"
    if "mlpl2" in target:
        return "MLPL2"
    return "MLP"


def build_run_metadata(prefix: str, experiment_key: str, cfg) -> dict:
    cost_label = infer_cost_label(cfg)
    shared_preset = "shared" in cost_label.lower()
    return {
        "experiment_key": experiment_key,
        "cost_function_label": cost_label,
        "shared_preset": shared_preset,
        "run_name": f"{prefix}-{experiment_key}",
    }


def build_swiss_roll_context(cfg, device):
    usd_sampler, utd_sampler, pd_sampler = build_swiss_roll_samplers(cfg, device)
    ds = cfg.dataset
    x_sampler = StandardNormalSampler(dim=int(ds.x_dim), device=str(device))
    y_sampler = SwissRollSampler(dim=int(ds.y_dim), device=str(device))
    mb = ds.minibatch
    otp_sampler = OTPlanSampler(
        method=str(mb.method),
        reg=float(mb.reg),
        reg_m=float(mb.get("reg_m", 1.0)),
        cost_function=str(mb.cost_function),
        normalize_cost=bool(mb.get("normalize_cost", False)),
    )

    n_test = int(ds.P_XY_paired)
    x0 = x_sampler.sample(n_test).to(device)
    x1 = y_sampler.sample(n_test).to(device)
    X_paired_test, Y_paired_test = otp_sampler.sample_plan(x0.cpu(), x1.cpu())
    X_paired_test, Y_paired_test = X_paired_test.to(device), Y_paired_test.to(device)

    X_unpaired_test = x_sampler.sample(n_test).to(device)
    Y_unpaired_test = y_sampler.sample(n_test).to(device)

    return {
        "usd_sampler": usd_sampler,
        "utd_sampler": utd_sampler,
        "pd_sampler": pd_sampler,
        "x_sampler": x_sampler,
        "y_sampler": y_sampler,
        "otp_sampler": otp_sampler,
        "X_paired_train": pd_sampler.x,
        "Y_paired_train": pd_sampler.y,
        "X_paired_test": X_paired_test,
        "Y_paired_test": Y_paired_test,
        "X_unpaired_test": X_unpaired_test,
        "Y_unpaired_test": Y_unpaired_test,
    }


def build_ema_model_copy(model_builder: Callable, cfg, device, model, enabled: bool):
    if not enabled:
        return None
    model_copy = model_builder(cfg, device)
    model_copy.load_state_dict(model.state_dict())
    return model_copy


def make_adam(params, opt_cfg):
    kwargs = {
        "lr": float(opt_cfg.lr),
        "betas": tuple(float(b) for b in opt_cfg.betas),
    }
    if opt_cfg.get("weight_decay") is not None:
        kwargs["weight_decay"] = float(opt_cfg.weight_decay)
    return torch.optim.Adam(params, **kwargs)


def should_run(step: int, every: int) -> bool:
    return every > 0 and step % every == 0


def build_periodic_loss_payload(
    model,
    output_unpaired,
    output_paired,
    total_loss,
    X_paired_train,
    Y_paired_train,
    X_paired_test,
    Y_paired_test,
    X_unpaired_test,
    Y_unpaired_test,
):
    return {
        "Unpaired loss": output_unpaired["loss"].item(),
        "Paired loss": output_paired["loss"].item(),
        "Loss": total_loss.item(),
        "Train paired loss": compute_loss(
            model, X_paired_train, Y_paired_train, X_paired_train, Y_paired_train
        ),
        "Test paired loss": compute_loss(model, X_paired_test, Y_paired_test, X_paired_test, Y_paired_test),
        "Test unpaired loss": compute_loss(
            model, X_unpaired_test, Y_unpaired_test, X_paired_test, Y_paired_test
        ),
    }


def log_optional_metrics(
    experiment: Experiment,
    output: dict,
    key_to_metric: dict[str, tuple[str, Callable[[torch.Tensor], float]]],
    step: int,
) -> None:
    for key, (name, transform) in key_to_metric.items():
        if key in output:
            experiment.log_metric(name, transform(output[key]), step=step)

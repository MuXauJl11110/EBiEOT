"""Hydra training for colored MNIST with MLP or CNN EBiEOT."""


import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from scripts.train import _maybe_clip, _optimizer, _resolve_seed, build_neural_model
from src.utils.core.pylogger import RankedLogger
from src.utils.core.seed import set_seed
from src.utils.experiment.hydra_utils import extras, task_wrapper
from src.utils.experiment.logging_utils import (
    instantiate_loggers,
    log_training_metrics,
    save_resolved_config,
)
from src.utils.samplers.colored_mnist import (
    build_colored_mnist_image_samplers,
    build_colored_mnist_samplers,
)
from tqdm import tqdm

log = RankedLogger(__name__)


def _build_samplers(cfg: DictConfig, device: torch.device):
    if bool(cfg.dataset.get("use_images", False)):
        return build_colored_mnist_image_samplers(cfg, device)
    return build_colored_mnist_samplers(cfg, device)


@task_wrapper
def train(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(_resolve_seed(cfg))

    output_dir = Path(cfg.paths.output_dir)
    save_resolved_config(cfg, output_dir)

    loggers = instantiate_loggers(cfg.get("logger"), full_cfg=cfg)

    model = build_neural_model(cfg, device)
    usd, utd, pd, _, _ = _build_samplers(cfg, device)
    opt_u = _optimizer(model, cfg.train.optimizer.unpaired)
    opt_p = _optimizer(model, cfg.train.optimizer.paired)
    tcfg = cfg.train

    for step in tqdm(
        range(int(tcfg.steps_from), int(tcfg.steps_to)),
        desc=tcfg.name,
    ):
        opt_u.zero_grad()
        x_u = usd.sample(int(tcfg.unpaired_batch_size))
        y_u = utd.sample(int(tcfg.unpaired_batch_size))
        out_u = model.compute_unpaired_loss(x_u, y_u)
        out_u["loss"].backward()
        _maybe_clip(model, float(tcfg.gradient_max_norm))
        opt_u.step()

        opt_p.zero_grad()
        x_p, y_p = pd.sample(int(tcfg.paired_batch_size))
        out_p = model.compute_paired_loss(x_p, y_p)
        out_p["loss"].backward()
        _maybe_clip(model, float(tcfg.gradient_max_norm))
        opt_p.step()

        log_every = int(tcfg.get("log_every", 0))
        if log_every > 0 and step % log_every == 0:
            metrics = {
                "train/unpaired_loss": float(out_u["loss"]),
                "train/paired_loss": float(out_p["loss"]),
            }
            log.info(
                f"step {step} unpaired={metrics['train/unpaired_loss']:.4f} "
                f"paired={metrics['train/paired_loss']:.4f}"
            )
            log_training_metrics(step, metrics, loggers)

    ckpt_path = output_dir / "checkpoint.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "cfg": OmegaConf.to_container(cfg, resolve=True),
        },
        ckpt_path,
    )
    log.info(f"Saved checkpoint to {ckpt_path}")


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    extras(cfg)
    train(cfg)


if __name__ == "__main__":
    main()

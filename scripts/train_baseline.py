"""CLI entry: inverse-OT baseline training from Hydra configs."""


import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from src.baselines import baseline_registry, build_baseline_samplers
from src.utils.core.pylogger import RankedLogger
from src.utils.core.seed import set_seed
from src.utils.experiment.hydra_utils import extras, task_wrapper
from src.utils.experiment.logging_utils import (
    instantiate_loggers,
    log_training_metrics,
    save_resolved_config,
)
from tqdm import tqdm

log = RankedLogger(__name__)


def _resolve_seed(cfg: DictConfig) -> int:
    if cfg.get("seed") is not None:
        return int(cfg.seed)
    return int(cfg.train.seed)


def _save_checkpoint(
    path: Path,
    trainer,
    cfg: DictConfig,
    step: int,
) -> None:
    torch.save(
        {
            "step": step,
            "state_dict": trainer.state_dict(),
            "cfg": OmegaConf.to_container(cfg, resolve=True),
        },
        path,
    )
    log.info(f"Saved checkpoint to {path}")


@task_wrapper
def train(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(_resolve_seed(cfg))

    output_dir = Path(cfg.paths.output_dir)
    save_resolved_config(cfg, output_dir)

    loggers = instantiate_loggers(cfg.get("logger"), full_cfg=cfg)

    method = str(cfg.method)
    if method not in baseline_registry:
        raise ValueError(
            f"Unknown baseline method: {method!r}. "
            f"Choose from {sorted(baseline_registry)}"
        )
    trainer = baseline_registry[method](cfg, device)
    trainer.build_optimizers(cfg)
    usd, utd, pd = build_baseline_samplers(cfg, device)

    tcfg = cfg.train
    steps_from = int(tcfg.steps_from)
    steps_to = int(tcfg.steps_to)
    save_every = int(tcfg.get("save_every", 0))
    log_every = int(tcfg.get("log_every", 0))

    for step in tqdm(range(steps_from, steps_to), desc=str(tcfg.name)):
        metrics = trainer.train_step(step, usd, utd, pd)

        if log_every > 0 and step % log_every == 0:
            log.info(" ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
            log_training_metrics(
                step,
                {f"train/{k}": v for k, v in metrics.items()},
                loggers,
            )

        if save_every > 0 and step > 0 and step % save_every == 0:
            _save_checkpoint(
                output_dir / f"checkpoint_step_{step}.pt",
                trainer,
                cfg,
                step,
            )

    _save_checkpoint(output_dir / "checkpoint.pt", trainer, cfg, max(steps_to - 1, 0))


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    extras(cfg)
    train(cfg)


if __name__ == "__main__":
    main()

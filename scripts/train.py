"""CLI entry: Swiss-roll toy training from Hydra configs."""

import math
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from src.ebieot.classification_based import ClassificationBasedEBiEOT
from src.ebieot.ebieot_gmm import EbieotGmm
from src.ebieot.ebieot_nn import EbieotNn
from src.ebieot.sampling.sample_buffer import SampleBuffer
from src.utils.core.pylogger import RankedLogger
from src.utils.core.seed import set_seed
from src.utils.experiment.hydra_utils import extras, task_wrapper
from src.utils.experiment.logging_utils import (
    instantiate_loggers,
    log_training_metrics,
    save_resolved_config,
)
from src.utils.datasets.mnist_classification import run_classification_training
from src.utils.samplers.mnist_classification import build_mnist_classification_loaders
from src.utils.samplers.synthetic import build_swiss_roll_samplers
from torch import optim
from torch.distributions import Independent, Normal
from tqdm import tqdm

log = RankedLogger(__name__)


def _make_sample_buffer(
    buf_cfg: Any,
    y_dim: int,
    device: torch.device,
    *,
    event_shape: tuple[int, ...] | None = None,
) -> SampleBuffer:
    scale = float(buf_cfg.noise_std)
    if event_shape is not None:
        loc = torch.zeros(event_shape, device=device)
        scale_t = torch.full(event_shape, scale, device=device)
        noise_gen = Independent(Normal(loc, scale_t), len(event_shape))
    else:
        noise_gen = Independent(
            Normal(
                torch.zeros(y_dim, device=device),
                torch.full((y_dim,), scale, device=device),
            ),
            1,
        )
    return SampleBuffer(
        noise_gen=noise_gen,
        p=float(buf_cfg.replay_p),
        max_samples=int(buf_cfg.max_samples),
    )


def build_neural_model(cfg: DictConfig, device: torch.device) -> EbieotNn:
    cost = instantiate(cfg.ebieot.cost).to(device)
    potential = instantiate(cfg.ebieot.potential).to(device)
    m = cfg.ebieot.model
    p = m.projection
    ds = cfg.dataset
    event_shape = None
    if bool(ds.get("use_images", False)):
        event_shape = (int(ds.channels), int(ds.img_size), int(ds.img_size))
    buffer = _make_sample_buffer(
        m.buffer,
        y_dim=int(ds.y_dim),
        device=device,
        event_shape=event_shape,
    )
    return EbieotNn(
        potential=potential,
        cost=cost,
        sample_buffer=buffer,
        sampler_type=str(m.sampler_type),
        epsilon=float(m.epsilon),
        alpha=float(m.alpha),
        reference_data_noise_sigma=float(m.reference_data_noise_sigma),
        step_size=float(m.step_size),
        noise=float(m.noise),
        num_iterations=int(m.num_iterations),
        decay=float(m.decay),
        thresh=m.thresh,
        grad_proj_type=str(m.grad_proj_type),
        norm_thresh=float(m.norm_thresh),
        value_thresh=float(m.value_thresh),
        projection_min=float(p.min),
        projection_max=float(p.max),
        is_projected=bool(p.is_projected),
        compute_stats=bool(m.compute_stats),
    )


def build_gmm_model(cfg: DictConfig, device: torch.device) -> EbieotGmm:
    cost = instantiate(cfg.ebieot.cost).to(device)
    gm = cfg.ebieot.model
    a_init = gm.get("A_diagonal_init")
    return EbieotGmm(
        cost=cost,
        y_dim=int(gm.y_dim),
        n_potentials=int(gm.n_potentials),
        epsilon=float(gm.epsilon),
        sampling_batch_size=int(gm.sampling_batch_size),
        A_diagonal_init=float(a_init) if a_init is not None else None,
    )


def build_classification_model(
    cfg: DictConfig, device: torch.device
) -> ClassificationBasedEBiEOT:
    cm = cfg.ebieot.model
    mlp_hd = cm.get("mlp_hidden_dim")
    cnn_hd = cm.get("cnn_hidden_dim")
    model = ClassificationBasedEBiEOT(
        num_classes=int(cm.num_classes),
        in_channels=int(cm.in_channels),
        image_size=int(cm.image_size),
        epsilon=float(cm.epsilon),
        arch=str(cm.arch),
        mlp_embed_dim=int(cm.mlp_embed_dim),
        cnn_embed_dim=int(cm.cnn_embed_dim),
        hidden_dim=int(cm.hidden_dim),
        mlp_hidden_dim=int(mlp_hd) if mlp_hd is not None else None,
        cnn_hidden_dim=int(cnn_hd) if cnn_hd is not None else None,
    )
    return model.to(device)


def _optimizer(model: torch.nn.Module, cfg: DictConfig) -> optim.Optimizer:
    return optim.Adam(
        model.parameters(),
        lr=float(cfg.lr),
        betas=tuple(float(b) for b in cfg.betas),
    )


def _maybe_clip(model: torch.nn.Module, max_norm: float) -> None:
    if not math.isfinite(max_norm):
        return
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def _resolve_seed(cfg: DictConfig) -> int:
    if cfg.get("seed") is not None:
        return int(cfg.seed)
    return int(cfg.train.seed)


def _build_model(cfg: DictConfig, device: torch.device, method: str):
    if method == "neural":
        return build_neural_model(cfg, device)
    if method == "gmm":
        return build_gmm_model(cfg, device)
    if method == "classification":
        return build_classification_model(cfg, device)
    raise ValueError(f"Unknown method: {method}")


def _train_classification(cfg: DictConfig, device: torch.device) -> None:
    output_dir = Path(cfg.paths.output_dir)
    save_resolved_config(cfg, output_dir)
    loggers = instantiate_loggers(cfg.get("logger"), full_cfg=cfg)

    model = build_classification_model(cfg, device)
    loaders = build_mnist_classification_loaders(cfg)
    tcfg = cfg.train

    result = run_classification_training(
        model,
        loaders.paired,
        loaders.marginal,
        loaders.val,
        loaders.test,
        device,
        epochs_max=int(tcfg.epochs_max),
        patience=int(tcfg.patience),
        lr=float(tcfg.lr),
        weight_decay=float(tcfg.weight_decay),
        grad_clip=float(tcfg.grad_clip),
        ema_decay=float(tcfg.ema_decay),
        use_ema=True,
    )

    log_every = int(tcfg.get("log_every", 0))
    for rec in result.get("epoch_records", []):
        epoch = int(rec["epoch"])
        if log_every > 0 and epoch % log_every == 0:
            metrics = {
                "train/loss": float(rec["train_loss"]),
                "train/joint": float(rec["train_joint"]),
                "train/marg_y": float(rec["train_marg_y"]),
                "train/logz": float(rec["train_logz"]),
                "val/acc": float(rec["val_acc"]),
                "val/loss": float(rec["val_loss"]),
            }
            log.info(
                f"epoch {epoch} train_loss={metrics['train/loss']:.4f} "
                f"val_acc={metrics['val/acc']:.4f}"
            )
            log_training_metrics(epoch, metrics, loggers)

    log.info(
        f"best_val_acc={result['best_val_acc']:.4f} "
        f"test_acc={result['test_acc']:.4f} test_loss={result['test_loss']:.4f}"
    )
    log_training_metrics(
        int(result["epochs_ran"]),
        {
            "best_val_acc": float(result["best_val_acc"]),
            "test/acc": float(result["test_acc"]),
            "test/loss": float(result["test_loss"]),
        },
        loggers,
    )

    ckpt_path = output_dir / "checkpoint.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "cfg": OmegaConf.to_container(cfg, resolve=True),
            "metrics": {
                "best_val_acc": result["best_val_acc"],
                "test_acc": result["test_acc"],
                "test_loss": result["test_loss"],
            },
        },
        ckpt_path,
    )
    log.info(f"Saved checkpoint to {ckpt_path}")


@task_wrapper
def train(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(_resolve_seed(cfg))

    method = str(cfg.get("method", "neural"))
    if method == "classification":
        _train_classification(cfg, device)
        return

    output_dir = Path(cfg.paths.output_dir)
    save_resolved_config(cfg, output_dir)

    loggers = instantiate_loggers(cfg.get("logger"), full_cfg=cfg)

    model = _build_model(cfg, device, method)
    usd, utd, pd = build_swiss_roll_samplers(cfg, device)
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

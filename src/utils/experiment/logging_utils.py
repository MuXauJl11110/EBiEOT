"""Experiment logging: comet init, metrics, resolved config dump."""

from importlib.util import find_spec
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional, Union

import hydra
from omegaconf import DictConfig, OmegaConf

from src.utils.core.pylogger import RankedLogger

log = RankedLogger(__name__)

LoggerHandle = Any
Loggers = Union[LoggerHandle, list[LoggerHandle], None]
_ACTIVE_LOGGERS: list[LoggerHandle] = []


def is_empty_logger_cfg(logger_cfg: Optional[DictConfig]) -> bool:
    if logger_cfg is None:
        return True
    if isinstance(logger_cfg, DictConfig) and len(logger_cfg) == 0:
        return True
    return False


def cfg_to_loggable_container(
    cfg: DictConfig,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Serialize a Hydra config for experiment loggers (notebook-safe).

    ``hydra compose()`` in Jupyter does not set ``HydraConfig``, so interpolations
    like ``${hydra:runtime.output_dir}`` cannot be resolved unless ``output_dir``
    is supplied explicitly (e.g. the notebook checkpoint path).
    """
    from omegaconf import open_dict

    working = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    if output_dir is not None:
        out = str(output_dir)
        with open_dict(working):
            if working.get("paths") is None:
                working.paths = OmegaConf.create({})
            with open_dict(working.paths):
                working.paths.output_dir = out
                working.paths.work_dir = out

    try:
        return OmegaConf.to_container(working, resolve=True, throw_on_missing=False)
    except Exception:
        log.warning(
            "Could not fully resolve Hydra config for logging; logging an unresolved copy."
        )
        return OmegaConf.to_container(working, resolve=False)


def _resolve_save_dir(
    logger_cfg: DictConfig,
    full_cfg: DictConfig,
    output_dir: str | Path | None,
) -> str:
    if output_dir is not None:
        return str(output_dir)
    try:
        return str(OmegaConf.select(full_cfg, "paths.output_dir", resolve=True))
    except Exception:
        pass
    try:
        return str(OmegaConf.select(logger_cfg, "save_dir", resolve=True))
    except Exception:
        return "."


def merge_tags(
    logger_tags: Optional[list],
    cfg_tags: Optional[list],
) -> list[str]:
    merged: list[str] = []
    for source in (logger_tags, cfg_tags):
        if not source:
            continue
        for tag in source:
            s = str(tag).strip()
            if s and s not in merged:
                merged.append(s)
    return merged


def init_comet_from_cfg(
    logger_cfg: DictConfig,
    *,
    full_cfg: DictConfig,
    name: Optional[str] = None,
    output_dir: str | Path | None = None,
) -> Any:
    """Initialize native Comet from ``conf/logger/comet.yaml``.

    Pass ``output_dir`` when calling from a notebook that uses ``hydra.compose``
    (no ``@hydra.main`` runtime).
    """
    if find_spec("comet_ml") is None:
        log.warning("comet_ml is not installed; skipping logger init.")
        return None

    from comet_ml import Experiment, OfflineExperiment

    run_name = name
    if run_name is None and full_cfg.get("train") and full_cfg.train.get("name"):
        run_name = str(full_cfg.train.name)

    project_name = str(logger_cfg.project)
    workspace = logger_cfg.get("entity")
    save_dir = _resolve_save_dir(logger_cfg, full_cfg, output_dir)
    offline = bool(logger_cfg.get("offline", False))

    init_kwargs: dict[str, Any] = {"project_name": project_name}
    if workspace:
        init_kwargs["workspace"] = str(workspace)

    log.info(f"Initializing comet_ml <project={project_name}>")
    if offline:
        init_kwargs["offline_directory"] = save_dir
        experiment = OfflineExperiment(**init_kwargs)
    else:
        experiment = Experiment(**init_kwargs)

    def _plain_tags(value: Any) -> Optional[list]:
        if value is None:
            return None
        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=False)
        return value

    tags = merge_tags(_plain_tags(logger_cfg.get("tags")), _plain_tags(full_cfg.get("tags")))
    if tags:
        experiment.add_tags(tags)
    if run_name:
        experiment.set_name(run_name)

    experiment.log_parameters(
        {"hydra_config": cfg_to_loggable_container(full_cfg, output_dir=output_dir)}
    )

    group = logger_cfg.get("group")
    if group:
        experiment.log_parameter("group", str(group))
    job_type = logger_cfg.get("job_type")
    if job_type:
        experiment.log_parameter("job_type", str(job_type))
    run_id = logger_cfg.get("id")
    if run_id:
        experiment.log_parameter("id", str(run_id))

    return experiment


def init_comet(
    project: str,
    save_dir: str,
    *,
    full_cfg: Optional[DictConfig] = None,
    entity: Optional[str] = None,
    offline: bool = False,
    id: Optional[str] = None,
    tags: Optional[list] = None,
    group: str = "",
    job_type: str = "",
    name: Optional[str] = None,
) -> Any:
    """Hydra ``_target_`` entry point; pass ``full_cfg`` from ``instantiate_loggers``."""
    if full_cfg is None:
        raise ValueError("init_comet requires full_cfg when used from Hydra wiring.")

    logger_cfg = OmegaConf.create(
        {
            "project": project,
            "save_dir": save_dir,
            "entity": entity,
            "offline": offline,
            "id": id,
            "tags": tags or [],
            "group": group,
            "job_type": job_type,
        }
    )
    return init_comet_from_cfg(logger_cfg, full_cfg=full_cfg, name=name)


def instantiate_loggers(
    logger_cfg: Optional[DictConfig],
    full_cfg: DictConfig,
) -> list[Any]:
    """Build experiment loggers from ``cfg.logger``."""
    if is_empty_logger_cfg(logger_cfg):
        log.info("No logger configs found; skipping logger instantiation.")
        return []

    assert logger_cfg is not None

    if "_target_" in logger_cfg:
        log.info(f"Instantiating logger <{logger_cfg._target_}>")
        run = hydra.utils.instantiate(logger_cfg, full_cfg=full_cfg)
        runs = [run] if run is not None else []
        _ACTIVE_LOGGERS.clear()
        _ACTIVE_LOGGERS.extend(runs)
        return runs

    if logger_cfg.get("project"):
        run = init_comet_from_cfg(logger_cfg, full_cfg=full_cfg)
        runs = [run] if run is not None else []
        _ACTIVE_LOGGERS.clear()
        _ACTIVE_LOGGERS.extend(runs)
        return runs

    runs: list[Any] = []
    for key, entry in logger_cfg.items():
        if not isinstance(entry, DictConfig) or "_target_" not in entry:
            continue
        log.info(f"Instantiating logger <{key}>: <{entry._target_}>")
        run = hydra.utils.instantiate(entry, full_cfg=full_cfg)
        if run is not None:
            runs.append(run)
    _ACTIVE_LOGGERS.clear()
    _ACTIVE_LOGGERS.extend(runs)
    return runs


def log_training_metrics(
    step: int,
    metrics: Mapping[str, float],
    loggers: Loggers,
) -> None:
    """Log scalar training metrics to active experiment loggers."""
    if not metrics or loggers is None:
        return

    if find_spec("comet_ml") is None:
        return

    handles = loggers if isinstance(loggers, list) else [loggers]
    payload: MutableMapping[str, float] = {k: float(v) for k, v in metrics.items()}

    for handle in handles:
        if handle is None:
            continue
        if hasattr(handle, "log_metrics"):
            handle.log_metrics(dict(payload), step=step)


def close_active_loggers() -> None:
    """Finalize active experiment loggers if they support explicit teardown."""
    if not _ACTIVE_LOGGERS:
        return

    for handle in _ACTIVE_LOGGERS:
        if handle is None:
            continue
        if hasattr(handle, "end"):
            handle.end()
    _ACTIVE_LOGGERS.clear()


def save_resolved_config(cfg: DictConfig, output_dir: Union[str, Path]) -> Path:
    """Write the fully resolved Hydra config to ``output_dir/config.yaml``."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "config.yaml"
    resolved = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    OmegaConf.save(resolved, path)
    log.info(f"Saved resolved config to {path}")
    return path

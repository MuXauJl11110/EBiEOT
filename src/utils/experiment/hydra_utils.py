"""Hydra entry helpers: pre-run extras and task wrapper."""

import warnings
from typing import Any, Callable, Optional

from omegaconf import DictConfig

from src.utils.core.pylogger import RankedLogger
from src.utils.experiment.rich_utils import enforce_tags, print_config_tree
from src.utils.experiment.logging_utils import close_active_loggers

log = RankedLogger(__name__)


def extras(cfg: DictConfig) -> None:
    """Optional pre-task utilities (warnings, tags, Rich config tree)."""
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        enforce_tags(cfg, save_to_file=True)

    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable[..., Any]) -> Callable[[DictConfig], Any]:
    """Run ``task_func(cfg)`` and always close active experiment loggers in ``finally``."""

    def wrap(cfg: DictConfig) -> Any:
        result: Any = None
        try:
            result = task_func(cfg)
        except Exception:
            log.exception("")
            raise
        finally:
            if cfg.get("paths") and cfg.paths.get("output_dir"):
                log.info(f"Output dir: {cfg.paths.output_dir}")

            log.info("Closing experiment loggers (if active).")
            close_active_loggers()

        return result

    return wrap

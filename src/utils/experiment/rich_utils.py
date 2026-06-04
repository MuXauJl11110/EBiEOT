"""Rich helpers for Hydra config inspection."""

from pathlib import Path
from typing import Sequence

import rich
import rich.syntax
import rich.tree
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict
from rich.prompt import Prompt

from src.utils.core.pylogger import RankedLogger

log = RankedLogger(__name__)


def print_config_tree(
    cfg: DictConfig,
    print_order: Sequence[str] = (
        "method",
        "train",
        "dataset",
        "ebieot",
        "logger",
        "paths",
        "extras",
        "task_name",
        "tags",
        "seed",
    ),
    resolve: bool = False,
    save_to_file: bool = False,
) -> None:
    """Print ``cfg`` as a Rich tree (optionally save under ``paths.output_dir``)."""
    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue: list[str] = []
    for field in print_order:
        if field in cfg:
            queue.append(field)
        else:
            log.warning(
                f"Field '{field}' not found in config. Skipping '{field}' config printing..."
            )

    for field in cfg:
        if field not in queue:
            queue.append(field)

    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)
        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)
        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    if save_to_file and cfg.get("paths") and cfg.paths.get("output_dir"):
        out = Path(cfg.paths.output_dir, "config_tree.log")
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as file:
            rich.print(tree, file=file)


def enforce_tags(cfg: DictConfig, save_to_file: bool = False) -> None:
    """Prompt for tags on the CLI when none are set (skipped during multirun)."""
    if cfg.get("tags"):
        return

    if "id" in HydraConfig().cfg.hydra.job:
        raise ValueError("Specify tags before launching a multirun!")

    log.warning("No tags provided in config. Prompting user to input tags...")
    tags = Prompt.ask("Enter a list of comma separated tags", default="dev")
    parsed = [t.strip() for t in tags.split(",") if t.strip()]

    with open_dict(cfg):
        cfg.tags = parsed

    log.info(f"Tags: {cfg.tags}")

    if save_to_file and cfg.get("paths") and cfg.paths.get("output_dir"):
        out = Path(cfg.paths.output_dir, "tags.log")
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as file:
            rich.print(cfg.tags, file=file)

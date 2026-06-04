import importlib.util
from pathlib import Path
from unittest.mock import patch

import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, open_dict

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TRAIN_BASELINE_SCRIPT = _REPO_ROOT / "scripts" / "train_baseline.py"

_FAST_OVERRIDES = [
    "experiment=baseline_reg_swiss_roll_16k",
    "train.steps_to=2",
    "train.batch_size=8",
    "train.save_every=0",
    "train.log_every=0",
    "dataset.P_XY_paired=64",
    "dataset.Q_X_unpaired=64",
    "dataset.R_Y_unpaired=64",
    "dataset.paired_cache.enabled=false",
    "extras.print_config=false",
    "extras.enforce_tags=false",
]


def _load_train_baseline_module():
    spec = importlib.util.spec_from_file_location(
        "ebieot_train_baseline", _TRAIN_BASELINE_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_train_baseline_smoke(conf_dir: str, tmp_path: Path) -> None:
    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        cfg = compose(config_name="config", overrides=_FAST_OVERRIDES)

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
        cfg.paths.work_dir = str(tmp_path)

    train_mod = _load_train_baseline_module()
    with patch.object(torch.cuda, "is_available", return_value=False):
        train_mod.train(cfg)

    checkpoint = tmp_path / "checkpoint.pt"
    assert checkpoint.is_file()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["step"] == 1
    assert "state_dict" in payload
    assert payload["cfg"]["method"] == "regression"

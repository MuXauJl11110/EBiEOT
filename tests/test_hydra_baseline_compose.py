import pytest
from hydra import compose, initialize_config_dir

from src.baselines import baseline_registry

BASELINE_EXPERIMENTS = [
    "baseline_cgan_swiss_roll_16k",
    "baseline_ugan_swiss_roll_16k",
    "baseline_reg_swiss_roll_16k",
    "baseline_cnf_swiss_roll_small",
    "baseline_cnf_swiss_roll_semi",
]

_FAST_OVERRIDES = [
    "train.steps_to=2",
    "dataset.P_XY_paired=64",
    "dataset.Q_X_unpaired=64",
    "dataset.R_Y_unpaired=64",
    "dataset.paired_cache.enabled=false",
    "extras.print_config=false",
    "extras.enforce_tags=false",
]


@pytest.mark.parametrize("experiment", BASELINE_EXPERIMENTS)
def test_compose_baseline_experiment(conf_dir: str, experiment: str) -> None:
    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        cfg = compose(
            config_name="config",
            overrides=[f"experiment={experiment}", *_FAST_OVERRIDES],
        )
    assert cfg.method in baseline_registry
    assert cfg.task_name == "train_baseline"
    assert int(cfg.train.steps_to) == 2
    assert int(cfg.dataset.P_XY_paired) == 64
    assert cfg.dataset.paired_cache.enabled is False

import importlib.util
from pathlib import Path
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from src.utils.core.seed import set_seed
from src.utils.experiment.hydra_utils import extras

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TRAIN_SCRIPT = _REPO_ROOT / "scripts" / "train.py"


def _load_train_module():
    spec = importlib.util.spec_from_file_location("ebieot_train", _TRAIN_SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_compose_egeot_swiss_roll(conf_dir: str):
    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        cfg = compose(
            config_name="config",
            overrides=["experiment=egeot_swiss_roll", "train.steps_to=2"],
        )
    assert cfg.method == "neural"
    assert cfg.train.paired_batch_size == 128
    assert cfg.ebieot.cost._target_.endswith("MLPCost")


def test_compose_includes_template_groups(conf_dir: str):
    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        cfg = compose(
            config_name="config",
            overrides=["experiment=egeot_swiss_roll"],
        )
    paths = OmegaConf.to_container(cfg.paths, resolve=False)
    assert paths["root_dir"] == "${oc.env:PROJECT_ROOT,.}"
    assert paths["log_dir"] == "${paths.root_dir}/logs/"
    assert paths["output_dir"] == "${hydra:runtime.output_dir}"
    assert cfg.extras.print_config is True
    assert cfg.extras.enforce_tags is False
    assert cfg.task_name == "train"
    assert not cfg.get("logger")


def test_compose_egeot_colored_mnist(conf_dir: str):
    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        cfg = compose(
            config_name="config",
            overrides=["experiment=egeot_colored_mnist", "train.steps_to=2"],
        )
    assert cfg.method == "neural"
    assert cfg.dataset.x_dim == 3072
    assert cfg.dataset.source_digit == 2
    assert cfg.dataset.target_digit == 3
    assert cfg.ebieot.cost.hidden_layers == [1024, 512]


def test_compose_egeot_colored_mnist_cnn_vanilla(conf_dir: str):
    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        cfg = compose(
            config_name="config",
            overrides=["experiment=egeot_colored_mnist_cnn_vanilla", "train.steps_to=2"],
        )
    assert cfg.method == "neural"
    assert cfg.dataset.use_images is True
    assert cfg.ebieot.cost._target_.endswith("VanillaCost")
    assert cfg.ebieot.potential._target_.endswith("VanillaPotential")


def test_compose_gmm_swiss_roll(conf_dir: str):
    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        cfg = compose(
            config_name="config",
            overrides=["experiment=gmm_swiss_roll", "train.steps_to=2"],
        )
    assert cfg.method == "gmm"
    assert cfg.train.paired_batch_size == 128
    assert list(cfg.ebieot.cost.log_v_m_hidden_channels) == []
    assert list(cfg.ebieot.cost.b_m_hidden_channels) == []


def test_compose_gmm_alae_adult_children(conf_dir: str):
    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        cfg = compose(
            config_name="config",
            overrides=["experiment=gmm_alae_adult_children", "train.steps_to=2"],
        )
    assert cfg.method == "gmm"
    assert cfg.dataset.input_data == "ADULT"
    assert cfg.dataset.target_data == "CHILDREN"
    assert cfg.ebieot.model.y_dim == 512
    assert cfg.train.name == "GMM_ALAE_ADULT_CHILDREN"


def test_compose_egeot_classification_mnist(conf_dir: str):
    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=egeot_classification_mnist",
                "train.epochs_max=2",
            ],
        )
    assert cfg.method == "classification"
    assert cfg.ebieot.model.num_classes == 10
    assert cfg.ebieot.model.image_size == 28
    assert cfg.dataset.paired_per_class == 20
    assert cfg.dataset.unpaired_per_class == 200
    assert cfg.train.epochs_max == 2


def test_extras_smoke(cfg_hydra: DictConfig):
    extras(cfg_hydra)


def test_extras_with_print_config(cfg_hydra: DictConfig):
    from omegaconf import open_dict

    with open_dict(cfg_hydra.extras):
        cfg_hydra.extras.print_config = True
    with patch("src.utils.experiment.hydra_utils.print_config_tree"):
        extras(cfg_hydra)


def test_compose_logger_comet(conf_dir: str):
    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        cfg = compose(
            config_name="config",
            overrides=["experiment=egeot_swiss_roll", "logger=comet"],
        )
    logger = OmegaConf.to_container(cfg.logger, resolve=False)
    assert logger["project"] == "ebieot"
    assert logger["save_dir"] == "${paths.output_dir}"


def test_compose_debug(conf_dir: str):
    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        cfg = compose(
            config_name="config",
            overrides=["experiment=egeot_swiss_roll", "debug=default"],
        )
    assert cfg.train.steps_to == 10
    assert cfg.task_name == "debug"


def test_build_neural_from_compose(conf_dir: str):
    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=egeot_swiss_roll",
                "train.steps_to=2",
                "ebieot.model.num_iterations=1",
            ],
        )
    set_seed(int(cfg.train.seed))
    import torch

    train_mod = _load_train_module()
    device = torch.device("cpu")
    model = train_mod.build_neural_model(cfg, device)
    x = torch.randn(4, 2, device=device)
    y = torch.randn(4, 2, device=device)
    out = model.compute_paired_loss(x, y)
    assert "loss" in out


def test_build_classification_from_compose(conf_dir: str):
    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=egeot_classification_mnist",
                "train.epochs_max=2",
            ],
        )
    set_seed(int(cfg.train.seed))
    import torch

    train_mod = _load_train_module()
    device = torch.device("cpu")
    model = train_mod.build_classification_model(cfg, device)
    x = torch.randn(4, 1, 28, 28, device=device)
    y = torch.randint(0, 10, (4,), device=device)
    parts = model.loss_with_parts(x, y, x)
    assert "loss" in parts
    assert "jointTerm" in parts
    assert "margYTerm" in parts
    assert "logzTerm" in parts


def test_cfg_hydra_fixture(cfg_hydra: DictConfig, tmp_path):
    assert cfg_hydra.method == "neural"
    assert cfg_hydra.train.steps_to == 2
    assert cfg_hydra.paths.output_dir == str(tmp_path)
    assert cfg_hydra.paths.log_dir == str(tmp_path)

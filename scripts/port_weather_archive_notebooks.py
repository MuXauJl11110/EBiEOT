#!/usr/bin/env python3
"""One-off port of notebooks/weather/*.ipynb to final-branch APIs."""


import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ARCHIVE = REPO / "notebooks" / "weather"

IMPORTS_GAN_EXTRA = "from torch.distributions.normal import Normal\n"

IMPORTS_CELL = """import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from comet_ml import Experiment
from scipy import linalg
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
{gan_extra}
from src.utils.samplers.data import DatasetSampler, PairedLoaderSampler
from src.utils.samplers.synthetic import StandardNormalSampler, SwissRollSampler
from src.utils.training.weather_notebook import (
    load_weather_tensors,
    paired_sampler_weather,
    unpaired_sampler_weather,
)

_cwd = Path.cwd().resolve()
if (_cwd / "conf").is_dir():
    REPO_ROOT = _cwd
elif (_cwd.parent / "conf").is_dir():
    REPO_ROOT = _cwd.parent
elif (_cwd.parent.parent / "conf").is_dir():
    REPO_ROOT = _cwd.parent.parent
else:
    raise RuntimeError("Run Jupyter with cwd = EBiEOT/ or notebooks/weather/")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

tsne = TSNE(n_components=2, random_state=50)
"""

REPLACEMENTS = [
    ("import wandb\n", ""),
    ("from comet_ml import Experiment\n", ""),  # dedupe if already in template
    ("import moscot.plotting as mtp\n", ""),
    ("from moscot import datasets\n", ""),
    ("from moscot.problems.cross_modality import TranslationProblem\n", ""),
    ("#https://moscot.readthedocs.io/en/latest/notebooks/tutorials/600_tutorial_translation.html\n", ""),
    ("import anndata as ad\n", ""),
    ("import scanpy as sc\n", ""),
    ("sys.path.append('/home/quickjkee/projects/Light-GCOT')\n\n", ""),
    ("sys.path.append('/home/quickjkee/projects/Light-GCOT')\n", ""),
    ("from src.samplers.from_dataset import DatasetSampler\n", ""),
    ("from src.samplers.from_loader import PairedLoaderSampler\n", ""),
    ("from src.samplers.primary import StandardNormalSampler, SwissRollSampler\n", ""),
    ("from src.costs.lse import MLPLSECost\n", "from src.ebieot.costs.lse import MLPLSECost\n"),
    ("from src.models.gmm_based import GMMEOT\n", "from src.ebieot.ebieot_gmm import EbieotGmm\n"),
    (
        "from src.models.models import MyCDiscriminator, MyCGenerator\n",
        "from src.networks.gan.descriminator import CGANDiscriminator\n"
        "from src.networks.gan.generator import CGANGenerator\n",
    ),
    (
        "from src.models.models import ConditionalRealNVP\n",
        "from src.networks.nf.real_nvp import ConditionalRealNVP\n",
    ),
    (
        "from src.models.models import ConditionalRealNVP, MLPnet\n",
        "from src.networks.mlp import MLPnet\n"
        "from src.networks.nf.real_nvp import ConditionalRealNVP\n",
    ),
    ("from src.utils.discrete_ot import OTPlanSampler\n", "from src.utils.samplers.discrete_ot import OTPlanSampler\n"),
    ("#wandb.init(name=EXP_NAME, config=config)\n", ""),
    (
        "netG = MyCGenerator(\n        x_dim=X_DIM,\n        t_dim=2,\n        n_t=1,\n",
        "netG = CGANGenerator(\n        x_dim=X_DIM,\n        y_dim=2,\n        n_y=1,\n",
    ),
    (
        "netD = MyCDiscriminator(x_dim=141, \n                        t_dim=2, \n                        n_t=1, \n",
        "netD = CGANDiscriminator(x_dim=141,\n                        y_dim=2,\n                        n_y=1,\n",
    ),
    (
        "netG = MyGenerator(\n        x_dim=X_DIM,\n",
        "from src.networks.gan.generator import MLPGenerator\n"
        "from src.networks.gan.descriminator import MLPDiscriminator\n\n"
        "netG = MLPGenerator(\n        x_dim=X_DIM,\n",
    ),
    (
        "netD = MyDiscriminator(x_dim=Y_DIM, \n",
        "netD = MLPDiscriminator(x_dim=Y_DIM,\n",
    ),
    (
        "T = ConditionalRealNVP(\n        features=Y_DIM,\n        context_features=X_DIM,\n        hidden_context_features=512,\n        hidden_features=128,\n",
        "T = ConditionalRealNVP(\n        features=Y_DIM,\n        context_features=X_DIM,\n        hidden_context_features=512,\n        hidden_features=[128],\n",
    ),
    (
        "from src.models.models import MyCDiscriminator, MyCGenerator\nfrom src.samplers.from_dataset import DatasetSampler\nfrom src.samplers.primary import StandardNormalSampler, SwissRollSampler\n",
        "from src.networks.gan.descriminator import CGANDiscriminator\n"
        "from src.networks.gan.generator import CGANGenerator\n"
        "from src.utils.samplers.data import DatasetSampler\n"
        "from src.utils.samplers.synthetic import StandardNormalSampler, SwissRollSampler\n",
    ),
]

EBM_CONFIG_CELL = """import importlib.util

_train_spec = importlib.util.spec_from_file_location(
    "ebieot_train", REPO_ROOT / "scripts" / "train.py"
)
assert _train_spec is not None and _train_spec.loader is not None
_train_mod = importlib.util.module_from_spec(_train_spec)
_train_spec.loader.exec_module(_train_mod)
build_gmm_model = _train_mod.build_gmm_model

from src.utils.training import CometExperiment, compose_weather_cfg, make_adam

EXPERIMENT = "gmm_weather"
OVERRIDES: list[str] = []
EXPERIMENT_ALIASES = {"gmm-weather": "gmm_weather"}

cfg, EXPERIMENT_KEY, seed = compose_weather_cfg(
    str(REPO_ROOT), str(EXPERIMENT), OVERRIDES, aliases=EXPERIMENT_ALIASES
)

device = torch.device(
    f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
)
torch.set_default_device(device)
dtype = torch.float64
torch.set_default_dtype(dtype)
"""


def apply_replacements(text: str) -> str:
    for old, new in REPLACEMENTS:
        text = text.replace(old, new)
    return text


def set_cell_source(cell: dict, source: str) -> None:
    lines = source.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    cell["source"] = lines


def port_notebook(path: Path, *, gan: bool = False) -> None:
    nb = json.loads(path.read_text())
    gan_extra = IMPORTS_GAN_EXTRA if gan else ""
    set_cell_source(nb["cells"][0], IMPORTS_CELL.format(gan_extra=gan_extra))

    for cell in nb["cells"][1:]:
        src = "".join(cell.get("source", []))
        new_src = apply_replacements(src)
        if new_src != src:
            set_cell_source(cell, new_src)

    # Comet logging in training cells
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if "#wandb.init" in src or (
            "experiment = Experiment" not in src
            and "MAX_STEPS" in src
            and "for step in tqdm" in src
            and "stats = []" in src
        ):
            if "experiment = Experiment" not in src and "EXP_NAME" in src:
                src = src.replace(
                    "stats = []\n",
                    'experiment = Experiment(project_name="inverse_ot")\n'
                    "experiment.set_name(EXP_NAME)\n"
                    "stats = []\n",
                    1,
                )
                set_cell_source(cell, apply_replacements(src))

    if path.name == "weather-ebm.ipynb":
        port_weather_ebm(nb)
    elif path.name == "weather.ipynb":
        port_weather_ebieot_gmm(nb)

    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"ported {path.name}")


def port_weather_ebm(nb: dict) -> None:
    # Replace configs.gmm_based cell with Hydra setup
    for i, cell in enumerate(nb["cells"]):
        src = "".join(cell.get("source", []))
        if "from configs.gmm_based.cost import MLPLSECostConfig" in src:
            set_cell_source(nb["cells"][i], EBM_CONFIG_CELL)
        if "def paired_sampler(X_pair, Y_pair, b_size):" in src:
            set_cell_source(
                nb["cells"][i],
                "paired_sampler = paired_sampler_weather\nunpaired_sampler = unpaired_sampler_weather\n",
            )
        if "cost = MLPLSECost(**cost_config.model_dump())" in src:
            src = src.replace(
                "cost = MLPLSECost(**cost_config.model_dump()).to(dtype)\n\n    model = GMMEOT(\n        y_dim=Y_DIM,\n        n_potentials=N_POTENTIALS,\n        cost=cost.to(dtype),\n    ).to(dtype)",
                "model = build_gmm_model(cfg, device).to(dtype)",
            )
            src = src.replace("train_config.unpaired_batch_size", "int(cfg.train.unpaired_batch_size)")
            src = src.replace("train_config.unpaired_batch_size", "int(cfg.train.unpaired_batch_size)")
            set_cell_source(nb["cells"][i], src)


def port_weather_ebieot_gmm(nb: dict) -> None:
    """Ensure weather.ipynb uses EBiEOT-GMM (build_gmm_model) training cells."""
    del nb  # handled by rename_weather_ebieot.py after port


def main() -> None:
    port_notebook(ARCHIVE / "weather-ebm.ipynb")
    port_notebook(ARCHIVE / "weather.ipynb")
    port_notebook(ARCHIVE / "weather-gan.ipynb", gan=True)
    port_notebook(ARCHIVE / "weather-reg.ipynb")
    port_notebook(ARCHIVE / "weather-cnf.ipynb")
    port_notebook(ARCHIVE / "weather-ugan.ipynb")


if __name__ == "__main__":
    main()

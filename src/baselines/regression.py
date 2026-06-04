import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import optim
from torch.optim import Optimizer

from src.baselines.common import batch_size, cfg_float, cfg_int
from src.networks.mlp import MLPnet


class RegressionTrainer:
    def __init__(self, cfg: DictConfig, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        model_cfg = cfg.model
        input_size = cfg_int(model_cfg, "input_size", cfg_int(cfg.dataset, "x_dim", 2))
        hidden_size = cfg_int(model_cfg, "hidden_size", 256)
        num_hidden_layers = cfg_int(model_cfg, "num_hidden_layers", 4)

        self.batch_size = batch_size(cfg)
        self.model = MLPnet(
            input_size=input_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
        ).to(device)
        self.optimizer: Optimizer | None = None

    def build_optimizers(self, cfg: DictConfig) -> tuple[Optimizer]:
        train_cfg = cfg.train
        lr = cfg_float(train_cfg, "lr", 3e-4)
        weight_decay = cfg_float(train_cfg, "weight_decay", 0.01)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        return (self.optimizer,)

    def train_step(self, step: int, usd, utd, pd) -> dict[str, float]:
        del step, usd, utd
        self.optimizer.zero_grad()
        x_paired, y_paired = pd.sample(self.batch_size)
        loss = F.mse_loss(y_paired, self.model(x_paired))
        loss.backward()
        self.optimizer.step()
        return {"loss": float(loss.item())}

    def state_dict(self) -> dict:
        return {"model": self.model.state_dict()}

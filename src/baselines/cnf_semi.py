import torch
from omegaconf import DictConfig
from torch import optim
from torch.optim import Optimizer

from src.baselines.cnf import build_conditional_real_nvp
from src.baselines.common import batch_size, cfg_float


class CnfSemiTrainer:
    def __init__(self, cfg: DictConfig, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.batch_size = batch_size(cfg)
        self.model = build_conditional_real_nvp(cfg, device)
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
        del step
        self.optimizer.zero_grad()
        batch_size = self.batch_size
        x_paired, y_paired = pd.sample(batch_size)
        log_prob = self.model.log_prob(inputs=y_paired, context=x_paired)

        x_unpaired = usd.sample(batch_size)
        y_unpaired = utd.sample(batch_size)
        fwd = self.model.log_prob(
            inputs=y_unpaired.repeat(batch_size, 1),
            context=x_unpaired.repeat(batch_size, 1),
        )
        unp_loss = torch.log(torch.mean(torch.exp(fwd), dim=-1)).mean()

        loss = -log_prob.mean() - unp_loss
        loss.backward()
        self.optimizer.step()
        return {
            "loss": float(loss.item()),
            "paired_nll": float((-log_prob.mean()).item()),
            "unp_loss": float(unp_loss.item()),
        }

    def state_dict(self) -> dict:
        return {"model": self.model.state_dict()}

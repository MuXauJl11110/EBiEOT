import torch
from omegaconf import DictConfig
from torch import optim
from torch.optim import Optimizer

from src.baselines.common import batch_size, cfg_float, cfg_int, cfg_list, x_dim
from src.networks.nf.real_nvp import ConditionalRealNVP


def build_conditional_real_nvp(
    cfg: DictConfig, device: torch.device
) -> ConditionalRealNVP:
    model_cfg = cfg.model
    features = cfg_int(model_cfg, "features", x_dim(cfg))
    context_features = cfg_int(model_cfg, "context_features", features)
    hidden_features_cfg = cfg_list(model_cfg, "hidden_features", [128])
    hidden_features = int(hidden_features_cfg[0])
    hidden_context_features = cfg_int(model_cfg, "hidden_context_features", 512)
    num_layers = cfg_int(model_cfg, "num_layers", 5)
    num_blocks_per_layer = cfg_int(model_cfg, "num_blocks_per_layer", 4)
    use_volume_preserving = bool(model_cfg.get("use_volume_preserving", False))
    return ConditionalRealNVP(
        features=features,
        hidden_features=hidden_features,
        context_features=context_features,
        hidden_context_features=hidden_context_features,
        num_layers=num_layers,
        num_blocks_per_layer=num_blocks_per_layer,
        use_volume_preserving=use_volume_preserving,
    ).to(device)


class CnfTrainer:
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
        del step, usd, utd
        self.optimizer.zero_grad()
        x_paired, y_paired = pd.sample(self.batch_size)
        log_prob = self.model.log_prob(inputs=y_paired, context=x_paired)
        loss = -log_prob.mean()
        loss.backward()
        self.optimizer.step()
        return {"loss": float(loss.item())}

    def state_dict(self) -> dict:
        return {"model": self.model.state_dict()}

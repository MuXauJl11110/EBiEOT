import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import optim
from torch.optim import Optimizer

from src.baselines.common import (
    batch_size,
    cfg_float,
    cfg_int,
    cfg_list,
    default_active,
    x_dim,
    y_dim,
)
from src.networks.gan.descriminator import CGANDiscriminator
from src.networks.gan.generator import CGANGenerator


class CganTrainer:
    def __init__(self, cfg: DictConfig, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        train_cfg = cfg.train
        active = default_active()
        z_dim = cfg_int(train_cfg, "z_dim", 1)
        num_timesteps = cfg_int(train_cfg, "num_timesteps", 1)
        layers_g = cfg_list(train_cfg, "layers_g", [256, 256, 256])
        layers_d = cfg_list(train_cfg, "layers_d", [256, 256, 256])

        self.z_dim = z_dim
        self.num_timesteps = num_timesteps
        self.batch_size = batch_size(cfg)
        self.r1_gamma = cfg_float(train_cfg, "r1_gamma", 0.01)
        lazy_reg = train_cfg.get("lazy_reg", 1)
        self.lazy_reg = None if lazy_reg is None else int(lazy_reg)

        self.net_g = CGANGenerator(
            x_dim=x_dim(cfg),
            y_dim=y_dim(cfg),
            n_y=num_timesteps,
            out_dim=y_dim(cfg),
            z_dim=z_dim,
            layers=layers_g,
            active=active,
        ).to(device)
        self.net_d = CGANDiscriminator(
            x_dim=x_dim(cfg),
            y_dim=y_dim(cfg),
            n_y=num_timesteps,
            layers=layers_d,
            active=active,
        ).to(device)
        self.optimizer_d: Optimizer | None = None
        self.optimizer_g: Optimizer | None = None

    def build_optimizers(self, cfg: DictConfig) -> tuple[Optimizer, Optimizer]:
        train_cfg = cfg.train
        beta1 = cfg_float(train_cfg, "beta1", 0.5)
        beta2 = cfg_float(train_cfg, "beta2", 0.9)
        lr_d = cfg_float(train_cfg, "lr_d", 3e-4)
        lr_g = cfg_float(train_cfg, "lr_g", 1e-3)
        self.optimizer_d = optim.Adam(
            self.net_d.parameters(), lr=lr_d, betas=(beta1, beta2)
        )
        self.optimizer_g = optim.Adam(
            self.net_g.parameters(), lr=lr_g, betas=(beta1, beta2)
        )
        return self.optimizer_d, self.optimizer_g

    def train_step(self, step: int, usd, utd, pd) -> dict[str, float]:
        del utd
        device = self.device
        batch_size = self.batch_size
        nz = self.z_dim

        for param in self.net_d.parameters():
            param.requires_grad = True
        self.net_d.zero_grad()

        x_paired, y_paired = pd.sample(batch_size)
        t = torch.randint(0, self.num_timesteps, (x_paired.size(0),), device=device)
        y_paired = y_paired.requires_grad_(True)

        d_real = self.net_d(y_paired, t, x_paired.detach())
        err_d_real = F.softplus(-d_real).mean()
        err_d_real.backward(retain_graph=True)

        if self.lazy_reg is None or step % self.lazy_reg == 0:
            grad_real = torch.autograd.grad(
                outputs=d_real.sum(),
                inputs=y_paired,
                create_graph=True,
            )[0]
            grad_penalty = (
                grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
            ).mean()
            (self.r1_gamma / 2 * grad_penalty).backward()

        latent_z = torch.randn(batch_size, nz, device=device)
        x_fake = self.net_g(x_paired.detach(), t, latent_z)
        err_d_fake = F.softplus(self.net_d(x_fake, t, x_paired.detach())).mean()
        err_d_fake.backward()
        err_d = err_d_real + err_d_fake
        self.optimizer_d.step()

        for param in self.net_d.parameters():
            param.requires_grad = False
        self.net_g.zero_grad()

        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device)
        unp_sample = usd.sample(batch_size)
        latent_z = torch.randn(batch_size, nz, device=device)
        x_predict = self.net_g(unp_sample.detach(), t, latent_z)
        output = self.net_d(x_predict, t, unp_sample.detach())
        err_g = F.softplus(-output).mean()
        err_g.backward()
        self.optimizer_g.step()

        return {"d_loss": float(err_d.item()), "g_loss": float(err_g.item())}

    def state_dict(self) -> dict:
        return {"net_g": self.net_g.state_dict(), "net_d": self.net_d.state_dict()}

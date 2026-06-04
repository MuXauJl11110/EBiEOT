import torch
from omegaconf import DictConfig
from src.utils.samplers.base import Sampler
from src.utils.samplers.data import DatasetSampler, PairedTensorBatchSampler
from src.utils.samplers.discrete_ot import OTPlanSampler


def swiss_roll_transform(
    t: torch.Tensor,
    generator: torch.Generator,
    noise: float = 0.0,
) -> torch.Tensor:
    """
    Maps parameter t → 2D swiss roll with optional isotropic noise.
    """
    x = t * torch.cos(t)
    z = t * torch.sin(t)

    X = torch.stack((x, z), dim=1)

    if noise > 0:
        X = X + noise * torch.randn(size=X.shape, generator=generator)

    return X


class SwissRollSampler(Sampler):
    def __init__(
        self,
        dim: int = 2,
        noise: float = 0.8,
        scale: float = 7.5,
        t_min: float = 1.5 * torch.pi,
        t_max: float = 4.5 * torch.pi,
        generator: torch.Generator | None = None,
        device: str = "cuda",
    ):
        super().__init__(generator=generator, device=device)
        assert dim == 2

        self.noise = noise
        self.scale = scale
        self.t_min = t_min
        self.t_max = t_max

    def sample(self, batch_size: int = 10):
        diff = self.t_max - self.t_min

        t = self.t_min + diff * torch.rand(
            (batch_size,), generator=self.generator, device=self.device
        )

        batch = (
            swiss_roll_transform(t=t, generator=self.generator, noise=self.noise)
            / self.scale
        )

        return batch


class StandardNormalSampler(Sampler):
    def __init__(
        self,
        dim: int = 1,
        generator: torch.Generator | None = None,
        device: str = "cuda",
    ):
        super(StandardNormalSampler, self).__init__(generator=generator, device=device)
        self.dim = dim

    def sample(self, batch_size: int = 10):
        return torch.randn(batch_size, self.dim, device=self.device)


def build_swiss_roll_samplers(
    cfg: DictConfig, device: torch.device
) -> tuple[DatasetSampler, DatasetSampler, PairedTensorBatchSampler]:
    dataset_cfg = cfg.dataset
    x_dim = int(dataset_cfg.x_dim)
    y_dim = int(dataset_cfg.y_dim)
    minibatch_cfg = dataset_cfg.minibatch
    ot_plan_sampler = OTPlanSampler(
        method=str(minibatch_cfg.method),
        reg=float(minibatch_cfg.reg),
        reg_m=float(minibatch_cfg.get("reg_m", 1.0)),
        cost_function=str(minibatch_cfg.cost_function),
        normalize_cost=bool(minibatch_cfg.get("normalize_cost", False)),
    )

    x_sampler = StandardNormalSampler(dim=x_dim, device=str(device))
    y_sampler = SwissRollSampler(dim=y_dim, device=str(device))
    num_paired_samples = int(dataset_cfg.P_XY_paired)
    x_for_pairing = x_sampler.sample(num_paired_samples).to(device)
    y_for_pairing = y_sampler.sample(num_paired_samples).to(device)

    x_paired, y_paired = ot_plan_sampler.sample_plan(
        x_for_pairing.cpu(), y_for_pairing.cpu()
    )
    x_paired = x_paired.to(device)
    y_paired = y_paired.to(device)
    paired_batch_sampler = PairedTensorBatchSampler(x_paired, y_paired)
    num_unpaired_x = int(dataset_cfg.Q_X_unpaired)
    num_unpaired_y = int(dataset_cfg.R_Y_unpaired)
    if num_unpaired_x > 0:
        unpaired_source_sampler = DatasetSampler(
            x_sampler.sample(num_unpaired_x).to(device), device=str(device)
        )
    else:
        unpaired_source_sampler = DatasetSampler(x_paired.cpu(), device=str(device))
    if num_unpaired_y > 0:
        unpaired_target_sampler = DatasetSampler(
            y_sampler.sample(num_unpaired_y).to(device), device=str(device)
        )
    else:
        unpaired_target_sampler = DatasetSampler(y_paired.cpu(), device=str(device))
    return unpaired_source_sampler, unpaired_target_sampler, paired_batch_sampler

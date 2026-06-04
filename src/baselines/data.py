import torch
from omegaconf import DictConfig, OmegaConf

from src.utils.datasets.paired import generate_paired_data, get_paired_sampler
from src.utils.samplers.data import DatasetSampler, PairedTensorBatchSampler
from src.utils.samplers.discrete_ot import OTPlanSampler
from src.utils.samplers.synthetic import StandardNormalSampler, SwissRollSampler


def build_baseline_samplers(
    cfg: DictConfig, device: torch.device
) -> tuple[DatasetSampler, DatasetSampler, PairedTensorBatchSampler | object]:
    dataset_cfg = cfg.dataset
    train_cfg = cfg.train
    x_dim = int(dataset_cfg.x_dim)
    y_dim = int(dataset_cfg.y_dim)
    p_xy_paired = int(dataset_cfg.P_XY_paired)
    q_x_unpaired = int(dataset_cfg.Q_X_unpaired)
    r_y_unpaired = int(dataset_cfg.R_Y_unpaired)
    batch_size = int(train_cfg.batch_size)

    minibatch_cfg = dataset_cfg.minibatch
    ot_plan_sampler = OTPlanSampler(
        method=str(minibatch_cfg.method),
        reg=float(minibatch_cfg.reg),
        reg_m=float(OmegaConf.select(minibatch_cfg, "reg_m", default=1.0)),
        cost_function=str(minibatch_cfg.cost_function),
        normalize_cost=bool(
            OmegaConf.select(minibatch_cfg, "normalize_cost", default=False)
        ),
    )

    x_sampler = StandardNormalSampler(dim=x_dim, device=str(device))
    y_sampler = SwissRollSampler(dim=y_dim, device=str(device))

    paired_cache_cfg = OmegaConf.select(dataset_cfg, "paired_cache", default=None)
    use_cache = bool(
        OmegaConf.select(paired_cache_cfg, "enabled", default=False)
        if paired_cache_cfg
        else False
    )

    if use_cache:
        cache_dir = str(paired_cache_cfg.dir)
        file_postfix = str(paired_cache_cfg.file_postfix)
        x_paired_train, y_paired_train, _, _ = generate_paired_data(
            x_sampler,
            y_sampler,
            ot_plan_sampler,
            p_xy_paired,
            cache_dir,
            file_postfix,
            device=str(device),
        )
        paired_sampler = get_paired_sampler(
            x_paired_train,
            y_paired_train,
            batch_size,
            p_xy_paired,
            device=str(device),
        )
        x_paired = x_paired_train
        y_paired = y_paired_train
    else:
        x_for_pairing = x_sampler.sample(p_xy_paired).to(device)
        y_for_pairing = y_sampler.sample(p_xy_paired).to(device)
        x_paired, y_paired = ot_plan_sampler.sample_plan(
            x_for_pairing.cpu(), y_for_pairing.cpu()
        )
        x_paired = x_paired.to(device)
        y_paired = y_paired.to(device)
        paired_sampler = PairedTensorBatchSampler(x_paired, y_paired)

    if q_x_unpaired > 0:
        unpaired_source = DatasetSampler(
            x_sampler.sample(q_x_unpaired).to(device), device=str(device)
        )
    else:
        unpaired_source = DatasetSampler(x_paired.cpu(), device=str(device))

    if r_y_unpaired > 0:
        unpaired_target = DatasetSampler(
            y_sampler.sample(r_y_unpaired).to(device), device=str(device)
        )
    else:
        unpaired_target = DatasetSampler(y_paired.cpu(), device=str(device))

    return unpaired_source, unpaired_target, paired_sampler

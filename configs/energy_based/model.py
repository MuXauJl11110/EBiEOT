from configs.energy_based.sampling import LangevinConfig, PseudoLangevinConfig
from pydantic import BaseModel


class EBMConfig(BaseModel):
    sampling: LangevinConfig | PseudoLangevinConfig = LangevinConfig()
    alpha: float = 0.0
    reference_data_noise_sigma: float = 3e-2  # 3e-2 for Colored Mnist
    epsilon: float = 1.0
    # SPECTRAL_NORM_ITERS = ?

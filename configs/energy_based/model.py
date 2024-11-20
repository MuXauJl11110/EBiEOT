from pydantic import BaseModel


from configs.energy_based.sampling import LangevinConfig, PseudoLangevinConfig


class EBMConfig(BaseModel):
    sampling: LangevinConfig | PseudoLangevinConfig = LangevinConfig()
    alpha: float = 0.0
    reference_data_noise_sigma: float = 0.0
    epsilon: float = 1.0
    # SPECTRAL_NORM_ITERS = ?

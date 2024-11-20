from pydantic import BaseModel


class TrainConfig(BaseModel):
    name: str = "GMM_Swiss_Roll"
    seed: int = 42

    paired_batch_size: int = 128
    unpaired_batch_size: int = 128
    gradient_max_norm: float = float("inf")

    steps_from: int = 0
    steps_to: int = 100000

    ema_update: bool = False
    plot_every: int = 1000

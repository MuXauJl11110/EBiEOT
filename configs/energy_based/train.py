from pydantic import BaseModel


class TrainConfig(BaseModel):
    name: str = "Energy-Based_Swiss_Roll"
    seed: int = 42

    batch_size: int = 1024
    gradient_max_norm: float = float("inf")

    steps_from: int = 0
    steps_to: int = 200

    ema_update: bool = False
    plot_every: int = 100

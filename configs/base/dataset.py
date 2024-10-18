from typing import Literal

from pydantic import BaseModel


class BaseDatasetConfig(BaseModel):
    x_dim: int = 2
    y_dim: int = 2
    P_XY_paired: int = 128
    Q_X_unpaired: int = 1024
    R_Y_unpaired: int = 1024


class BaseMiniBatchConfig(BaseModel):
    method: Literal["exact", "sinkhorn", "unbalanced", "partial"] = "sinkhorn"
    reg: float = 0.05
    cost_function: Literal["l2", "anti-l2", "rotation", "rotation-v2"] = "rotation-v2"

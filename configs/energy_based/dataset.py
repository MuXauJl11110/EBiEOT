from configs.base.dataset import BaseDatasetConfig, BaseMiniBatchConfig


class DatasetConfig(BaseDatasetConfig):
    P_XY_paired: int = 128
    Q_X_unpaired: int = 1024
    R_Y_unpaired: int = 1024


class MiniBatchConfig(BaseMiniBatchConfig):
    pass

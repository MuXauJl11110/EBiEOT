from pydantic import model_validator

from configs.base.cost import BaseCostConfig


class CostConfig(BaseCostConfig):
    m_potentials: int = 4
    log_v_m_hidden_channels: list[int] = [128, 128]
    b_m_hidden_channels: list[int] = [128, 128]

    @model_validator(mode="after")
    def append_hidden_channels(self):
        self.log_v_m_hidden_channels.append(self.m_potentials)
        self.b_m_hidden_channels.append(self.m_potentials * self.y_dim)

        return self

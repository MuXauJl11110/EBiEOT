"""Stdlib logging wrapper for CLI training (single-GPU; no Lightning rank helpers)."""

import logging
from typing import Mapping, Optional


class RankedLogger(logging.LoggerAdapter):
    """Thin command-line logger.

    ``rank_zero_only`` is accepted for API compatibility with the Hydra template but
  is a no-op in single-process CLI runs.
    """

    def __init__(
        self,
        name: str = __name__,
        rank_zero_only: bool = False,
        extra: Optional[Mapping[str, object]] = None,
    ) -> None:
        logger = logging.getLogger(name)
        super().__init__(logger=logger, extra=extra)
        self.rank_zero_only = rank_zero_only

    def log(self, level: int, msg: str, *args, **kwargs) -> None:
        if self.isEnabledFor(level):
            msg, kwargs = self.process(msg, kwargs)
            self.logger.log(level, msg, *args, **kwargs)

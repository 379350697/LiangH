from __future__ import annotations

import os
from collections.abc import Mapping

from .config import MarketMakerConfig


LIVE_ORDERS_ENV = "LIANGH_MARKET_MAKER_LIVE_ORDERS"


class LiveExecutorSafetyError(RuntimeError):
    pass


def assert_live_orders_enabled(
    config: MarketMakerConfig,
    env: Mapping[str, str] | None = None,
) -> bool:
    runtime_env = os.environ if env is None else env
    if config.mode != "live":
        raise LiveExecutorSafetyError("live orders require mode=live")
    if not config.execution.allow_live_orders:
        raise LiveExecutorSafetyError("live orders require execution.allow_live_orders=true")
    if runtime_env.get(LIVE_ORDERS_ENV) != "1":
        raise LiveExecutorSafetyError(f"live orders require {LIVE_ORDERS_ENV}=1")
    return True


class LiveExecutorStub:
    def __init__(self, config: MarketMakerConfig) -> None:
        self.config = config

    def place_order(self, *args: object, **kwargs: object) -> None:
        assert_live_orders_enabled(self.config)
        raise NotImplementedError("live order submission is intentionally disabled in v1")


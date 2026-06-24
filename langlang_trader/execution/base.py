from __future__ import annotations

from typing import Protocol

from langlang_trader.models import AccountSnapshot, OrderIntent, OrderResult, Position


class Executor(Protocol):
    def get_account(self) -> AccountSnapshot:
        ...

    def get_positions(self) -> list[Position]:
        ...

    def place_order(self, intent: OrderIntent) -> OrderResult:
        ...

    def cancel_order(self, order_id: str) -> OrderResult:
        ...

    def sync_fills(self) -> list[OrderResult]:
        ...

    def close_position(self, symbol: str, reason: str) -> OrderResult:
        ...

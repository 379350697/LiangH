from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Protocol

from .models import BookTick, InventoryState, LimitOrderState, QuoteIntent, TopBookTick, TradeTick


class ExecutionGateway(Protocol):
    gateway_name: str
    inventory: InventoryState

    def open_orders(self) -> list[LimitOrderState]:
        ...

    def place_quotes(self, quotes: list[QuoteIntent], book: BookTick, now_ns: int) -> list[LimitOrderState]:
        ...

    def cancel_all(self, reason: str, now_ns: int | None = None) -> int:
        ...

    def expire_orders(self, now_ns: int) -> int:
        ...

    def on_trade(self, tick: TradeTick) -> list[object]:
        ...


class RecoveryGateway(Protocol):
    def list_open_orders(self, symbol: str, context: str) -> list[object]:
        ...

    def cancel_all_open_orders(self, symbol: str, context: str) -> dict[str, object]:
        ...


class UserOrderStream(Protocol):
    async def stream(self) -> object:
        ...


@dataclass
class MarketSignalState:
    max_spread_bps: float
    use_book_ticker: bool = True
    use_trade_flow: bool = True
    last_top_book: TopBookTick | None = None
    last_trade: TradeTick | None = None
    cancel_urgency: bool = False
    spread_shock: bool = False

    def update_top_book(self, tick: TopBookTick) -> None:
        if not self.use_book_ticker:
            return
        self.last_top_book = tick
        spread_bps = tick.spread_bps
        self.spread_shock = spread_bps is None or spread_bps > self.max_spread_bps
        self.cancel_urgency = self.spread_shock

    def update_trade(self, tick: TradeTick) -> None:
        if not self.use_trade_flow:
            return
        self.last_trade = tick

    def clear_urgent_flags(self) -> None:
        self.cancel_urgency = False
        self.spread_shock = False


@dataclass
class RateLimitBudget:
    max_order_ops_per_minute: int = 1000
    max_order_ops_per_10s: int = 240
    external_remaining_minute: int | None = None
    external_remaining_10s: int | None = None
    _local_order_ops: deque[tuple[int, int]] = field(default_factory=deque)

    def can_submit(self, cost: int, now_ns: int | None = None) -> bool:
        now = time.monotonic_ns() if now_ns is None else now_ns
        self._prune(now)
        minute_used = self._used_since(now - 60_000_000_000)
        ten_second_used = self._used_since(now - 10_000_000_000)
        if minute_used + cost > self.max_order_ops_per_minute:
            return False
        if ten_second_used + cost > self.max_order_ops_per_10s:
            return False
        if self.external_remaining_minute is not None and cost > self.external_remaining_minute:
            return False
        if self.external_remaining_10s is not None and cost > self.external_remaining_10s:
            return False
        return True

    def record_order_op(self, cost: int = 1, now_ns: int | None = None) -> None:
        now = time.monotonic_ns() if now_ns is None else now_ns
        self._prune(now)
        self._local_order_ops.append((now, cost))
        if self.external_remaining_minute is not None:
            self.external_remaining_minute = max(0, self.external_remaining_minute - cost)
        if self.external_remaining_10s is not None:
            self.external_remaining_10s = max(0, self.external_remaining_10s - cost)

    def snapshot(self) -> dict[str, int | None]:
        return {
            "max_order_ops_per_minute": self.max_order_ops_per_minute,
            "max_order_ops_per_10s": self.max_order_ops_per_10s,
            "external_remaining_minute": self.external_remaining_minute,
            "external_remaining_10s": self.external_remaining_10s,
        }

    def sync_from_binance_rate_limits(self, rate_limits: list[dict[str, object]]) -> None:
        for item in rate_limits:
            if item.get("rateLimitType") != "ORDERS":
                continue
            limit = int(item.get("limit", 0))
            count = int(item.get("count", 0))
            remaining = max(0, limit - count)
            interval = str(item.get("interval", "")).upper()
            interval_num = int(item.get("intervalNum", 1))
            if interval == "MINUTE" and interval_num == 1:
                self.external_remaining_minute = remaining
                self.max_order_ops_per_minute = min(self.max_order_ops_per_minute, limit)
            elif interval == "SECOND" and interval_num == 10:
                self.external_remaining_10s = remaining
                self.max_order_ops_per_10s = min(self.max_order_ops_per_10s, limit)

    def _used_since(self, cutoff_ns: int) -> int:
        return sum(cost for ts, cost in self._local_order_ops if ts >= cutoff_ns)

    def _prune(self, now_ns: int) -> None:
        cutoff = now_ns - 60_000_000_000
        while self._local_order_ops and self._local_order_ops[0][0] < cutoff:
            self._local_order_ops.popleft()

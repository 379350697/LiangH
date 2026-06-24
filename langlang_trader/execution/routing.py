from __future__ import annotations

from dataclasses import dataclass

from langlang_trader.models import OrderIntent
from langlang_trader.universe import UniverseSnapshot, UniverseSymbol


@dataclass(frozen=True)
class RoutedOrderIntent:
    intent: OrderIntent
    exchange: str
    exchange_symbol: str
    route_reason: str


class ExecutionRouter:
    """Route canonical order intents to exchange-specific paper executors."""

    def __init__(self, snapshot: UniverseSnapshot, *, shared_symbol_policy: str = "binance_first"):
        if shared_symbol_policy != "binance_first":
            raise ValueError(f"unsupported shared symbol policy: {shared_symbol_policy}")
        self.snapshot = snapshot
        self.shared_symbol_policy = shared_symbol_policy
        self._okx_symbols = self._symbols_for_exchange("okx")
        self._binance_symbols = self._symbols_for_exchange("binance")

    def route(self, intent: OrderIntent) -> RoutedOrderIntent | None:
        symbol = intent.symbol
        okx_row = self._okx_symbols.get(symbol)
        binance_row = self._binance_symbols.get(symbol)
        if okx_row is not None and binance_row is not None:
            return RoutedOrderIntent(
                intent=intent,
                exchange="binance",
                exchange_symbol=self._exchange_symbol(binance_row),
                route_reason="shared_binance_preferred",
            )
        if binance_row is not None:
            return RoutedOrderIntent(
                intent=intent,
                exchange="binance",
                exchange_symbol=self._exchange_symbol(binance_row),
                route_reason="binance_only",
            )
        if okx_row is not None:
            return RoutedOrderIntent(
                intent=intent,
                exchange="okx",
                exchange_symbol=self._exchange_symbol(okx_row),
                route_reason="okx_only",
            )
        return None

    def rejection_reason(self, intent: OrderIntent) -> str:
        if self.route(intent) is not None:
            return ""
        return "symbol_not_executable_on_configured_exchanges"

    def _symbols_for_exchange(self, exchange: str) -> dict[str, UniverseSymbol]:
        rows: dict[str, UniverseSymbol] = {}
        for row in self.snapshot.rows:
            if row.source_exchange != exchange or row.is_reference:
                continue
            if self._is_executable(row):
                rows[row.symbol] = row
        return rows

    @staticmethod
    def _is_executable(row: UniverseSymbol) -> bool:
        if row.tradable:
            return True
        if row.source_exchange == "binance":
            return row.filter_reason in {"okx_executable_overlap", "binance_observed_only_not_okx_executable"}
        return bool(row.execution_symbol)

    @staticmethod
    def _exchange_symbol(row: UniverseSymbol) -> str:
        return row.exchange_symbol or row.execution_symbol or row.symbol

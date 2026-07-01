from __future__ import annotations

import json
import socket
import time
from collections.abc import AsyncIterator
from typing import Any

from .config import LedgerContext
from .models import OrderTruthEvent


def parse_order_trade_update(
    payload: dict[str, Any],
    receive_time_ns: int,
    context: LedgerContext,
) -> OrderTruthEvent | None:
    data = payload.get("data", payload)
    if data.get("e") != "ORDER_TRADE_UPDATE":
        return None
    order = data.get("o")
    if not isinstance(order, dict):
        return None
    return OrderTruthEvent(
        symbol=str(order.get("s", context.symbol)).upper(),
        order_id=str(order.get("i", "")),
        client_order_id=str(order.get("c", "")),
        event_type=str(data.get("e", "ORDER_TRADE_UPDATE")),
        order_status=str(order.get("X", "")),
        execution_type=str(order.get("x", "")),
        side=str(order.get("S", "")).upper(),
        price=_float(order.get("p")),
        qty=_float(order.get("q")),
        filled_qty=_float(order.get("z")),
        last_fill_qty=_float(order.get("l")),
        last_fill_price=_float(order.get("L")),
        event_time_ms=int(data.get("E", 0)),
        transaction_time_ms=int(order.get("T", 0)),
        receive_time_ns=receive_time_ns,
        truth_source="user_data",
        strategy_version=context.strategy_version,
        strategy_tree_variant_id=context.strategy_tree_variant_id,
        strategy_tree_parent_id=context.strategy_tree_parent_id,
        strategy_tree_path=list(context.strategy_tree_path),
        venue=context.venue,
        payload=data,
    )


class BinanceUserOrderStream:
    def __init__(self, listen_key: str, context: LedgerContext, base_url: str = "wss://fstream.binance.com/ws") -> None:
        self.listen_key = listen_key
        self.context = context
        self.base_url = base_url.rstrip("/")

    @property
    def url(self) -> str:
        return f"{self.base_url}/{self.listen_key}"

    async def stream(self) -> AsyncIterator[OrderTruthEvent]:
        try:
            import websockets
        except ModuleNotFoundError as exc:
            raise RuntimeError("Install websockets>=15.0 to use Binance user data stream") from exc

        async with websockets.connect(self.url, ping_interval=20, family=socket.AF_INET) as ws:
            async for message in ws:
                payload = json.loads(message)
                event = parse_order_trade_update(payload, receive_time_ns=time.monotonic_ns(), context=self.context)
                if event is not None:
                    yield event


def _float(value: object) -> float:
    if value in {None, ""}:
        return 0.0
    return float(value)

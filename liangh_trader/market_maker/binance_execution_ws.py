from __future__ import annotations

import hashlib
import hmac
import itertools
import time
from decimal import Decimal
from urllib.parse import urlencode

from .config import MarketMakerConfig
from .exchange_interfaces import RateLimitBudget
from .ledger import MarketMakerLedger
from .models import BookTick, InventoryState, LimitOrderState, OrderTruthEvent, QuoteIntent, TradeTick


class BinanceWsApiRequestBuilder:
    def __init__(self, api_key: str, api_secret: str, recv_window_ms: int = 5000) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.recv_window_ms = recv_window_ms
        self._request_seq = itertools.count(1)

    def build_order_place_request(
        self,
        *,
        symbol: str,
        side: str,
        price: float | Decimal | str,
        quantity: float | Decimal | str,
        client_order_id: str,
        timestamp_ms: int | None = None,
        request_id: str | None = None,
        post_only: bool = True,
    ) -> dict[str, object]:
        params = {
            "apiKey": self.api_key,
            "symbol": symbol.upper(),
            "side": _binance_side(side),
            "type": "LIMIT",
            "timeInForce": "GTX" if post_only else "GTC",
            "quantity": _decimal_text(quantity),
            "price": _decimal_text(price),
            "newClientOrderId": client_order_id,
            "recvWindow": self.recv_window_ms,
            "timestamp": int(time.time() * 1000) if timestamp_ms is None else int(timestamp_ms),
        }
        params["signature"] = self.sign_params(params)
        return {
            "id": request_id or self._next_request_id("place"),
            "method": "order.place",
            "params": params,
        }

    def build_order_modify_request(
        self,
        *,
        symbol: str,
        order_id: str | int,
        side: str,
        price: float | Decimal | str,
        quantity: float | Decimal | str,
        timestamp_ms: int | None = None,
        request_id: str | None = None,
    ) -> dict[str, object]:
        params = {
            "apiKey": self.api_key,
            "symbol": symbol.upper(),
            "orderId": str(order_id),
            "side": _binance_side(side),
            "quantity": _decimal_text(quantity),
            "price": _decimal_text(price),
            "recvWindow": self.recv_window_ms,
            "timestamp": int(time.time() * 1000) if timestamp_ms is None else int(timestamp_ms),
        }
        params["signature"] = self.sign_params(params)
        return {
            "id": request_id or self._next_request_id("modify"),
            "method": "order.modify",
            "params": params,
        }

    def build_order_cancel_request(
        self,
        *,
        symbol: str,
        order_id: str | int,
        timestamp_ms: int | None = None,
        request_id: str | None = None,
    ) -> dict[str, object]:
        params = {
            "apiKey": self.api_key,
            "symbol": symbol.upper(),
            "orderId": str(order_id),
            "recvWindow": self.recv_window_ms,
            "timestamp": int(time.time() * 1000) if timestamp_ms is None else int(timestamp_ms),
        }
        params["signature"] = self.sign_params(params)
        return {
            "id": request_id or self._next_request_id("cancel"),
            "method": "order.cancel",
            "params": params,
        }

    def sign_params(self, params: dict[str, object]) -> str:
        query = urlencode(sorted((key, str(value)) for key, value in params.items()))
        return hmac.new(self.api_secret.encode("utf-8"), query.encode("utf-8"), hashlib.sha256).hexdigest()

    def _next_request_id(self, prefix: str) -> str:
        return f"{prefix}-{next(self._request_seq)}"


class BinanceWsExecutionGateway:
    gateway_name = "binance_ws_api"

    def __init__(
        self,
        config: MarketMakerConfig,
        ledger: MarketMakerLedger,
        builder: BinanceWsApiRequestBuilder,
        rate_limit_budget: RateLimitBudget,
        dry_run: bool = True,
    ) -> None:
        self.config = config
        self.ledger = ledger
        self.builder = builder
        self.rate_limit_budget = rate_limit_budget
        self.dry_run = dry_run
        self.inventory = InventoryState(symbol=config.symbol)
        self._orders: dict[str, LimitOrderState] = {}

    def open_orders(self) -> list[LimitOrderState]:
        return [
            order
            for order in self._orders.values()
            if order.status in {"open", "partially_filled"} and order.remaining_qty > 0
        ]

    def place_quotes(self, quotes: list[QuoteIntent], book: BookTick, now_ns: int) -> list[LimitOrderState]:
        accepted: list[LimitOrderState] = []
        for quote in quotes:
            client_order_id = quote.quote_id or f"mm-{quote.side}-{now_ns}"
            request = self.builder.build_order_place_request(
                symbol=quote.symbol,
                side=quote.side,
                price=quote.price,
                quantity=quote.qty,
                client_order_id=client_order_id,
                timestamp_ms=int(time.time() * 1000),
            )
            self.ledger.record_execution_request(
                gateway=self.gateway_name,
                method=str(request["method"]),
                request_id=str(request["id"]),
                status="dry_run" if self.dry_run else "built",
                latency_ms=0.0,
                payload={"request": request, "book_update_id": book.update_id},
                rate_limit_snapshot=self.rate_limit_budget.snapshot(),
            )
            order = LimitOrderState(
                order_id=client_order_id,
                quote_id=quote.quote_id,
                symbol=quote.symbol,
                side=quote.side,
                price=quote.price,
                qty=quote.qty,
                remaining_qty=quote.qty,
                status="open" if self.dry_run else "pending_ack",
                post_only=quote.post_only,
                created_at_ns=now_ns,
                updated_at_ns=now_ns,
                expires_at_ns=now_ns + quote.ttl_ms * 1_000_000,
                strategy_version=quote.strategy_version,
                strategy_tree_variant_id=quote.strategy_tree_variant_id,
                strategy_tree_parent_id=quote.strategy_tree_parent_id,
                strategy_tree_path=list(quote.strategy_tree_path),
                venue=quote.venue,
            )
            self._orders[order.order_id] = order
            accepted.append(order)
        return accepted

    def cancel_all(self, reason: str, now_ns: int | None = None) -> int:
        timestamp_ns = time.monotonic_ns() if now_ns is None else now_ns
        canceled = 0
        for order in self.open_orders():
            order.status = "canceled"
            order.updated_at_ns = timestamp_ns
            canceled += 1
        return canceled

    def expire_orders(self, now_ns: int) -> int:
        expired = 0
        for order in self.open_orders():
            if now_ns >= order.expires_at_ns:
                order.status = "expired"
                order.updated_at_ns = now_ns
                expired += 1
        return expired

    def on_trade(self, tick: TradeTick) -> list[object]:
        return []

    def apply_order_truth_event(self, event: OrderTruthEvent) -> LimitOrderState | None:
        order = self._orders.get(event.order_id) or self._orders.get(event.client_order_id)
        if order is None:
            return None
        status = _local_status_from_user_data(event.order_status)
        if status is not None:
            order.status = status
        order.remaining_qty = max(0.0, order.qty - event.filled_qty)
        order.updated_at_ns = event.receive_time_ns
        return order


def _binance_side(side: str) -> str:
    normalized = side.upper()
    if normalized not in {"BUY", "SELL"}:
        raise ValueError(f"unsupported Binance side: {side}")
    return normalized


def _decimal_text(value: float | Decimal | str) -> str:
    decimal_value = Decimal(str(value))
    text = format(decimal_value.normalize(), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _local_status_from_user_data(status: str) -> str | None:
    return {
        "NEW": "open",
        "PARTIALLY_FILLED": "partially_filled",
        "FILLED": "filled",
        "CANCELED": "canceled",
        "EXPIRED": "expired",
        "EXPIRED_IN_MATCH": "expired",
    }.get(status)

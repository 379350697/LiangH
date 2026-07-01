from __future__ import annotations

import itertools
import time

from .config import MarketMakerConfig
from .ledger import MarketMakerLedger
from .models import BookTick, FillEvent, InventoryState, LimitOrderState, OrderTruthEvent, QuoteIntent, TradeTick


class MarketMakerPaperExecutor:
    def __init__(self, config: MarketMakerConfig, ledger: MarketMakerLedger) -> None:
        self.config = config
        self.ledger = ledger
        self.inventory = InventoryState(
            symbol=config.symbol,
            quote_usdt=config.paper.initial_quote_usdt,
        )
        self._orders: dict[str, LimitOrderState] = {}
        self._order_seq = itertools.count(1)
        self._fill_seq = itertools.count(1)

    def open_orders(self) -> list[LimitOrderState]:
        return [
            order
            for order in self._orders.values()
            if order.status in {"open", "partially_filled"} and order.remaining_qty > 0
        ]

    def place_quotes(self, quotes: list[QuoteIntent], book: BookTick) -> list[LimitOrderState]:
        accepted: list[LimitOrderState] = []
        for quote in quotes:
            self.ledger.record_quote(quote, status="submitted")
            if quote.symbol != self.config.symbol:
                self._record_rejected(quote, "symbol_mismatch")
                continue
            if quote.side not in {"buy", "sell"}:
                self._record_rejected(quote, "invalid_side")
                continue
            if quote.post_only and self._would_cross_spread(quote, book):
                self._record_rejected(quote, "post_only_would_cross")
                continue
            self._cancel_side_for_replace(quote.side, quote.created_at_ns)
            order = LimitOrderState(
                order_id=f"mm-{next(self._order_seq)}",
                quote_id=quote.quote_id,
                symbol=quote.symbol,
                side=quote.side,
                price=quote.price,
                qty=quote.qty,
                remaining_qty=quote.qty,
                status="open",
                post_only=quote.post_only,
                created_at_ns=quote.created_at_ns,
                updated_at_ns=quote.created_at_ns,
                expires_at_ns=quote.created_at_ns + quote.ttl_ms * 1_000_000,
                strategy_version=quote.strategy_version,
                strategy_tree_variant_id=quote.strategy_tree_variant_id,
                strategy_tree_parent_id=quote.strategy_tree_parent_id,
                strategy_tree_path=list(quote.strategy_tree_path),
                venue=quote.venue,
            )
            self._orders[order.order_id] = order
            self.ledger.record_order_state(order, reason="accepted")
            accepted.append(order)
        return accepted

    def cancel_order(self, order_id: str, reason: str = "cancel") -> bool:
        order = self._orders.get(order_id)
        if order is None or order.status not in {"open", "partially_filled"}:
            return False
        order.status = "canceled"
        order.updated_at_ns = time.monotonic_ns()
        self.ledger.record_order_state(order, reason=reason)
        return True

    def cancel_all(self, reason: str = "cancel_all") -> int:
        canceled = 0
        for order in list(self.open_orders()):
            if self.cancel_order(order.order_id, reason=reason):
                canceled += 1
        return canceled

    def expire_orders(self, now_ns: int) -> int:
        expired = 0
        for order in list(self.open_orders()):
            if now_ns >= order.expires_at_ns:
                order.status = "expired"
                order.updated_at_ns = now_ns
                self.ledger.record_order_state(order, reason="ttl_expired")
                expired += 1
        return expired

    def on_trade(self, tick: TradeTick) -> list[FillEvent]:
        if tick.symbol != self.config.symbol:
            return []
        fills: list[FillEvent] = []
        remaining_trade_qty = max(tick.qty, 0.0) * max(self.config.paper.queue_fill_ratio, 0.0)
        for order in list(self.open_orders()):
            if remaining_trade_qty <= 0:
                break
            if not self._trade_fills_order(order, tick):
                continue
            qty = min(order.remaining_qty, remaining_trade_qty)
            if qty <= 0:
                continue
            remaining_trade_qty -= qty
            order.remaining_qty -= qty
            order.updated_at_ns = tick.receive_time_ns
            order.status = "filled" if order.remaining_qty <= 1e-12 else "partially_filled"
            fill_price = order.price
            fee = fill_price * qty * self.config.paper.maker_fee_bps / 10_000.0
            self._apply_inventory_fill(order.side, fill_price, qty, fee)
            fill = FillEvent(
                fill_id=f"fill-{next(self._fill_seq)}",
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                price=fill_price,
                qty=qty,
                fee_usdt=fee,
                liquidity="maker",
                trade_id=tick.trade_id,
                event_time_ms=tick.event_time_ms,
                receive_time_ns=tick.receive_time_ns,
                inventory_base_qty=self.inventory.base_qty,
                inventory_quote_usdt=self.inventory.quote_usdt,
                strategy_version=order.strategy_version,
                strategy_tree_variant_id=order.strategy_tree_variant_id,
                strategy_tree_parent_id=order.strategy_tree_parent_id,
                strategy_tree_path=list(order.strategy_tree_path),
                venue=order.venue,
            )
            self.ledger.record_fill(fill)
            self.ledger.record_order_state(order, reason="fill")
            self.ledger.record_inventory(self.inventory, reason="fill")
            fills.append(fill)
        return fills

    def flatten_inventory(self, book: BookTick, now_ns: int, reason: str) -> list[FillEvent]:
        base_qty = self.inventory.base_qty
        if abs(base_qty) <= 1e-12:
            return []
        if base_qty > 0:
            if book.best_bid is None or book.best_bid <= 0:
                return []
            side = "sell"
            price = book.best_bid
            qty = base_qty
        else:
            if book.best_ask is None or book.best_ask <= 0:
                return []
            side = "buy"
            price = book.best_ask
            qty = abs(base_qty)
        fee = price * qty * self.config.paper.taker_fee_bps / 10_000.0
        if side == "sell":
            if self.inventory.avg_price > 0:
                self.inventory.realized_pnl_usdt += (price - self.inventory.avg_price) * qty
            self.inventory.quote_usdt += price * qty - fee
        else:
            if self.inventory.avg_price > 0:
                self.inventory.realized_pnl_usdt += (self.inventory.avg_price - price) * qty
            self.inventory.quote_usdt -= price * qty + fee
        self.inventory.base_qty = 0.0
        self.inventory.avg_price = 0.0
        self.inventory.fees_usdt += fee
        fill = FillEvent(
            fill_id=f"fill-{next(self._fill_seq)}",
            order_id=f"flatten-{next(self._order_seq)}",
            symbol=self.config.symbol,
            side=side,
            price=price,
            qty=qty,
            fee_usdt=fee,
            liquidity="taker_stop",
            trade_id=reason,
            event_time_ms=book.event_time_ms,
            receive_time_ns=now_ns,
            inventory_base_qty=self.inventory.base_qty,
            inventory_quote_usdt=self.inventory.quote_usdt,
            strategy_version=self.config.strategy.strategy_version,
            strategy_tree_variant_id=self.config.strategy_tree.strategy_tree_variant_id,
            strategy_tree_parent_id=self.config.strategy_tree.strategy_tree_parent_id,
            strategy_tree_path=list(self.config.strategy_tree.strategy_tree_path),
            venue=self.config.venue,
        )
        self.ledger.record_fill(fill)
        self.ledger.record_inventory(self.inventory, reason=reason)
        return [fill]

    def apply_order_truth_event(self, event: OrderTruthEvent) -> LimitOrderState | None:
        order = self._orders.get(event.order_id)
        if order is None:
            order = next((candidate for candidate in self._orders.values() if candidate.quote_id == event.client_order_id), None)
        if order is None:
            return None
        status = _local_status_from_user_data(event.order_status)
        if status is not None:
            order.status = status
        order.remaining_qty = max(0.0, order.qty - event.filled_qty)
        order.updated_at_ns = event.receive_time_ns
        self.ledger.record_order_state(order, reason="user_data_truth")
        return order

    def _cancel_side_for_replace(self, side: str, now_ns: int) -> None:
        for order in list(self.open_orders()):
            if order.side == side:
                order.status = "canceled"
                order.updated_at_ns = now_ns
                self.ledger.record_order_state(order, reason="replace")

    def _record_rejected(self, quote: QuoteIntent, reason: str) -> None:
        now_ns = quote.created_at_ns
        order = LimitOrderState(
            order_id=f"reject-{next(self._order_seq)}",
            quote_id=quote.quote_id,
            symbol=quote.symbol,
            side=quote.side,
            price=quote.price,
            qty=quote.qty,
            remaining_qty=quote.qty,
            status="rejected",
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
        self.ledger.record_order_state(order, reason=reason)

    @staticmethod
    def _would_cross_spread(quote: QuoteIntent, book: BookTick) -> bool:
        if quote.side == "buy" and book.best_ask is not None:
            return quote.price >= book.best_ask
        if quote.side == "sell" and book.best_bid is not None:
            return quote.price <= book.best_bid
        return False

    @staticmethod
    def _trade_fills_order(order: LimitOrderState, tick: TradeTick) -> bool:
        if order.side == "buy":
            return tick.price <= order.price
        if order.side == "sell":
            return tick.price >= order.price
        return False

    def _apply_inventory_fill(self, side: str, price: float, qty: float, fee: float) -> None:
        inventory = self.inventory
        if side == "buy":
            previous_base = inventory.base_qty
            new_base = previous_base + qty
            if previous_base >= 0 and new_base > 0:
                inventory.avg_price = ((inventory.avg_price * previous_base) + (price * qty)) / new_base
            elif new_base > 0:
                inventory.avg_price = price
            inventory.base_qty = new_base
            inventory.quote_usdt -= price * qty + fee
        else:
            if inventory.base_qty > 0 and inventory.avg_price > 0:
                matched_qty = min(qty, inventory.base_qty)
                inventory.realized_pnl_usdt += (price - inventory.avg_price) * matched_qty
            inventory.base_qty -= qty
            inventory.quote_usdt += price * qty - fee
            if abs(inventory.base_qty) <= 1e-12:
                inventory.avg_price = 0.0
        inventory.fees_usdt += fee


def _local_status_from_user_data(status: str) -> str | None:
    return {
        "NEW": "open",
        "PARTIALLY_FILLED": "partially_filled",
        "FILLED": "filled",
        "CANCELED": "canceled",
        "EXPIRED": "expired",
        "EXPIRED_IN_MATCH": "expired",
    }.get(status)

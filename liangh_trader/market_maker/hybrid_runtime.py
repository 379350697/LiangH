from __future__ import annotations

import itertools
import time

from .config import MarketMakerConfig
from .exchange_interfaces import ExecutionGateway, MarketSignalState, RateLimitBudget, RecoveryGateway
from .ledger import MarketMakerLedger
from .models import BookTick, InventoryState, LatencySample, LimitOrderState, OrderTruthEvent, QuoteIntent, TopBookTick, TradeTick
from .paper_executor import MarketMakerPaperExecutor
from .strategy import OfiInventorySkewMakerStrategy, ReferencePassiveMakerStrategy


class PaperExecutionGateway:
    gateway_name = "paper"

    def __init__(self, config: MarketMakerConfig, ledger: MarketMakerLedger) -> None:
        self.executor = MarketMakerPaperExecutor(config=config, ledger=ledger)

    @property
    def inventory(self) -> InventoryState:
        return self.executor.inventory

    def open_orders(self) -> list[LimitOrderState]:
        return self.executor.open_orders()

    def place_quotes(self, quotes: list[QuoteIntent], book: BookTick, now_ns: int) -> list[LimitOrderState]:
        return self.executor.place_quotes(quotes, book=book)

    def cancel_all(self, reason: str, now_ns: int | None = None) -> int:
        return self.executor.cancel_all(reason=reason)

    def expire_orders(self, now_ns: int) -> int:
        return self.executor.expire_orders(now_ns)

    def on_trade(self, tick: TradeTick) -> list[object]:
        return self.executor.on_trade(tick)

    def apply_order_truth_event(self, event: OrderTruthEvent) -> LimitOrderState | None:
        return self.executor.apply_order_truth_event(event)

    def flatten_inventory(self, book: BookTick, now_ns: int, reason: str) -> list[object]:
        return self.executor.flatten_inventory(book, now_ns=now_ns, reason=reason)


class HybridMarketMakerRuntime:
    def __init__(
        self,
        config: MarketMakerConfig,
        ledger: MarketMakerLedger,
        execution_gateway: ExecutionGateway | None = None,
        recovery_gateway: RecoveryGateway | None = None,
        signal_state: MarketSignalState | None = None,
        rate_limit_budget: RateLimitBudget | None = None,
        strategy: ReferencePassiveMakerStrategy | None = None,
    ) -> None:
        self.config = config
        self.ledger = ledger
        self.execution_gateway = execution_gateway or PaperExecutionGateway(config=config, ledger=ledger)
        self.recovery_gateway = recovery_gateway
        self.signal_state = signal_state or MarketSignalState(
            max_spread_bps=config.risk.max_spread_bps,
            use_book_ticker=config.signals.use_book_ticker,
            use_trade_flow=config.signals.use_trade_flow,
        )
        self.rate_limit_budget = rate_limit_budget or RateLimitBudget(
            max_order_ops_per_minute=config.limits.max_order_ops_per_minute,
            max_order_ops_per_10s=config.limits.max_order_ops_per_10s,
        )
        self.strategy = strategy or _strategy_for_config(config)
        self.last_quote_ns = 0
        self._request_seq = itertools.count(1)

    def open_orders(self) -> list[LimitOrderState]:
        return self.execution_gateway.open_orders()

    def on_market_event(
        self,
        event: BookTick | TopBookTick | TradeTick,
        now_ns: int,
        loop_lag_ms: float = 0.0,
    ) -> None:
        if isinstance(event, TopBookTick):
            self._record_book_latency(event, now_ns)
            self.signal_state.update_top_book(event)
            if self.signal_state.spread_shock:
                self.ledger.record_risk_event(
                    reason="spread_shock",
                    payload={"source": event.source, "spread_bps": event.spread_bps},
                )
                self.execution_gateway.cancel_all(reason="spread_shock", now_ns=now_ns)
            return
        if isinstance(event, TradeTick):
            self._record_trade_latency(event, now_ns)
            self.signal_state.update_trade(event)
            self.execution_gateway.on_trade(event)
            return
        self._on_l2_book(event, now_ns=now_ns, loop_lag_ms=loop_lag_ms)

    def on_order_truth_event(self, event: OrderTruthEvent) -> None:
        self.ledger.record_order_truth_event(event)
        apply_truth = getattr(self.execution_gateway, "apply_order_truth_event", None)
        if apply_truth is not None:
            apply_truth(event)

    def _on_l2_book(self, book: BookTick, now_ns: int, loop_lag_ms: float) -> None:
        self._record_book_latency(book, now_ns)
        self.execution_gateway.expire_orders(now_ns)
        if not self._quote_authority_ok(book=book, now_ns=now_ns, loop_lag_ms=loop_lag_ms):
            return
        if self.last_quote_ns and now_ns - self.last_quote_ns < self.config.strategy.quote_interval_ms * 1_000_000:
            return
        quotes = self.strategy.generate_quotes(book, self.execution_gateway.inventory, now_ns=now_ns)
        if not quotes:
            return
        cost = len(quotes)
        if not self.rate_limit_budget.can_submit(cost=cost, now_ns=now_ns):
            self.ledger.record_risk_event(
                reason="rate_limit_backoff",
                payload={"cost": cost, "budget": self.rate_limit_budget.snapshot()},
            )
            return
        accepted = self.execution_gateway.place_quotes(quotes, book=book, now_ns=now_ns)
        self.rate_limit_budget.record_order_op(cost=cost, now_ns=now_ns)
        self.ledger.record_execution_request(
            gateway=self.execution_gateway.gateway_name,
            method="quote_batch",
            request_id=f"{self.execution_gateway.gateway_name}-{next(self._request_seq)}",
            status="accepted" if accepted else "empty",
            latency_ms=0.0,
            payload={"quote_count": len(quotes), "accepted_count": len(accepted), "source": book.source},
            rate_limit_snapshot=self.rate_limit_budget.snapshot(),
        )
        self.last_quote_ns = now_ns

    def _quote_authority_ok(self, book: BookTick, now_ns: int, loop_lag_ms: float) -> bool:
        reasons: list[tuple[str, dict[str, object]]] = []
        if book.source != "l2_depth":
            reasons.append(("top_book_not_tradeable", {"source": book.source}))
        if book.book_status != "hot":
            reasons.append(("book_not_hot", {"book_status": book.book_status}))
        if book.stale or book.sequence_gap:
            reasons.append(("depth_sequence_gap" if book.sequence_gap else "stale_feed", {"book_stale": book.stale}))
        feed_age_ms = (now_ns - book.receive_time_ns) / 1_000_000.0
        if feed_age_ms > self.config.risk.stale_feed_ms:
            reasons.append(("stale_feed", {"feed_age_ms": feed_age_ms}))
        if loop_lag_ms > self.config.risk.max_loop_lag_ms:
            reasons.append(("loop_lag_exceeded", {"loop_lag_ms": loop_lag_ms}))
        spread_bps = book.spread_bps
        if spread_bps is None or spread_bps > self.config.risk.max_spread_bps:
            reasons.append(("abnormal_spread", {"spread_bps": spread_bps}))
        if self.signal_state.spread_shock:
            reasons.append(("spread_shock", {"signal": "book_ticker"}))
        inventory = self.execution_gateway.inventory
        if abs(inventory.base_qty) > self.config.risk.max_inventory_base:
            self._force_flatten_inventory(
                book=book,
                now_ns=now_ns,
                reason="inventory_cap_exceeded",
                payload={"base_qty": inventory.base_qty},
            )
            return False
        mid_price = book.mid_price
        if mid_price is not None:
            if _inventory_stop_hit(
                base_qty=inventory.base_qty,
                avg_price=inventory.avg_price,
                mid_price=mid_price,
                stop_bps=self.config.strategy.inventory_stop_bps,
            ):
                self.ledger.record_risk_event(
                    reason="inventory_stop_loss",
                    payload={
                        "base_qty": inventory.base_qty,
                        "avg_price": inventory.avg_price,
                        "mid_price": mid_price,
                        "stop_bps": self.config.strategy.inventory_stop_bps,
                    },
                )
                self.execution_gateway.cancel_all(reason="inventory_stop_loss", now_ns=now_ns)
                flatten = getattr(self.execution_gateway, "flatten_inventory", None)
                if flatten is not None:
                    flatten(book, now_ns=now_ns, reason="inventory_stop_loss")
                return False
            if abs(inventory.base_qty * mid_price) > self.config.risk.max_notional_usdt:
                self._force_flatten_inventory(
                    book=book,
                    now_ns=now_ns,
                    reason="notional_cap_exceeded",
                    payload={"base_qty": inventory.base_qty, "mid_price": mid_price},
                )
                return False
        if not reasons:
            return True
        for reason, payload in reasons:
            self.ledger.record_risk_event(reason=reason, payload=payload)
        self.execution_gateway.cancel_all(reason="risk_halt", now_ns=now_ns)
        return False

    def _force_flatten_inventory(self, book: BookTick, now_ns: int, reason: str, payload: dict[str, object]) -> None:
        self.ledger.record_risk_event(reason=reason, payload=payload)
        self.execution_gateway.cancel_all(reason=reason, now_ns=now_ns)
        flatten = getattr(self.execution_gateway, "flatten_inventory", None)
        if flatten is not None:
            flatten(book, now_ns=now_ns, reason=reason)

    def _record_book_latency(self, event: BookTick | TopBookTick, now_ns: int) -> None:
        event_age_ms = 0.0
        if event.event_time_ms > 0:
            event_age_ms = max(0.0, time.time() * 1000.0 - event.event_time_ms)
        self.ledger.record_latency(
            LatencySample(
                name="market_data_book",
                sample_time_ns=now_ns,
                latency_ms=event_age_ms,
                symbol=event.symbol,
                venue=event.venue,
                payload={
                    "source": event.source,
                    "book_status": event.book_status,
                    "event_age_ms": event_age_ms,
                    "update_id": event.update_id,
                    "stale": event.stale,
                    "sequence_gap": event.sequence_gap,
                    "resync_count": event.resync_count,
                    "sequence_gap_count": event.sequence_gap_count,
                },
            )
        )

    def _record_trade_latency(self, tick: TradeTick, now_ns: int) -> None:
        event_age_ms = 0.0
        if tick.event_time_ms > 0:
            event_age_ms = max(0.0, time.time() * 1000.0 - tick.event_time_ms)
        self.ledger.record_latency(
            LatencySample(
                name="market_data_trade",
                sample_time_ns=now_ns,
                latency_ms=event_age_ms,
                symbol=tick.symbol,
                venue=tick.venue,
                payload={
                    "source": "trade",
                    "event_age_ms": event_age_ms,
                    "price": tick.price,
                    "qty": tick.qty,
                },
            )
        )


def _strategy_for_config(config: MarketMakerConfig):
    if config.strategy.strategy_version == "scalp_passive_maker_ofi_v1":
        return OfiInventorySkewMakerStrategy(config)
    return ReferencePassiveMakerStrategy(config)


def _inventory_stop_hit(*, base_qty: float, avg_price: float, mid_price: float, stop_bps: float) -> bool:
    if abs(base_qty) <= 1e-12 or avg_price <= 0 or mid_price <= 0 or stop_bps <= 0:
        return False
    if base_qty > 0:
        return (mid_price / avg_price - 1.0) * 10_000.0 <= -abs(stop_bps)
    return (avg_price / mid_price - 1.0) * 10_000.0 <= -abs(stop_bps)

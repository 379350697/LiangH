from __future__ import annotations

import asyncio
import time

from .config import MarketMakerConfig
from .ledger import MarketMakerLedger
from .models import BookTick, LatencySample, TopBookTick, TradeTick
from .paper_executor import MarketMakerPaperExecutor
from .strategy import OfiInventorySkewMakerStrategy, ReferencePassiveMakerStrategy


class MarketMakerRunner:
    def __init__(
        self,
        config: MarketMakerConfig,
        ledger: MarketMakerLedger,
        executor: MarketMakerPaperExecutor | None = None,
        strategy: ReferencePassiveMakerStrategy | None = None,
    ) -> None:
        self.config = config
        self.ledger = ledger
        self.executor = executor or MarketMakerPaperExecutor(config=config, ledger=ledger)
        self.strategy = strategy or _strategy_for_config(config)
        self.quote_interval_ms = config.strategy.quote_interval_ms
        self.order_ttl_ms = config.strategy.order_ttl_ms
        self.stale_feed_ms = config.risk.stale_feed_ms
        self.max_loop_lag_ms = config.risk.max_loop_lag_ms
        self.last_quote_ns = 0
        self.last_latency_sample_ns = 0
        self.last_latency_status = ""
        self.last_latency_sample_ns_by_source: dict[str, int] = {}
        self.last_latency_status_by_source: dict[str, str] = {}
        self.last_event_receive_ns: int | None = None
        self.last_trade_latency_sample_ns = 0
        self.last_trade_receive_ns: int | None = None

    def evaluate_risk(
        self,
        book: BookTick,
        now_ns: int,
        last_event_receive_ns: int | None,
        loop_lag_ms: float,
    ) -> bool:
        reasons: list[tuple[str, dict[str, object]]] = []
        if book.source != "l2_depth":
            reasons.append(("top_book_not_tradeable", {"source": book.source}))
        if book.book_status != "hot":
            reasons.append(
                (
                    "book_not_hot",
                    {
                        "book_status": book.book_status,
                        "resync_count": book.resync_count,
                        "sequence_gap_count": book.sequence_gap_count,
                    },
                )
            )
        if book.stale or book.sequence_gap:
            reasons.append(("depth_sequence_gap" if book.sequence_gap else "stale_feed", {"book_stale": book.stale}))
        if last_event_receive_ns is None:
            reasons.append(("stale_feed", {"last_event_receive_ns": None}))
        else:
            feed_age_ms = (now_ns - last_event_receive_ns) / 1_000_000.0
            if feed_age_ms > self.config.risk.stale_feed_ms:
                reasons.append(("stale_feed", {"feed_age_ms": feed_age_ms}))
        if loop_lag_ms > self.config.risk.max_loop_lag_ms:
            reasons.append(("loop_lag_exceeded", {"loop_lag_ms": loop_lag_ms}))
        mid_price = book.mid_price
        spread_bps = book.spread_bps
        if spread_bps is None or spread_bps > self.config.risk.max_spread_bps:
            reasons.append(("abnormal_spread", {"spread_bps": spread_bps}))
        if abs(self.executor.inventory.base_qty) > self.config.risk.max_inventory_base:
            reasons.append(("inventory_cap_exceeded", {"base_qty": self.executor.inventory.base_qty}))
        if mid_price is not None:
            if _inventory_stop_hit(
                base_qty=self.executor.inventory.base_qty,
                avg_price=self.executor.inventory.avg_price,
                mid_price=mid_price,
                stop_bps=self.config.strategy.inventory_stop_bps,
            ):
                self.ledger.record_risk_event(
                    reason="inventory_stop_loss",
                    payload={
                        "base_qty": self.executor.inventory.base_qty,
                        "avg_price": self.executor.inventory.avg_price,
                        "mid_price": mid_price,
                        "stop_bps": self.config.strategy.inventory_stop_bps,
                    },
                )
                self.executor.cancel_all(reason="inventory_stop_loss")
                self.executor.flatten_inventory(book, now_ns=now_ns, reason="inventory_stop_loss")
                return False
            inventory_notional = abs(self.executor.inventory.base_qty * mid_price)
            if inventory_notional > self.config.risk.max_notional_usdt:
                reasons.append(("notional_cap_exceeded", {"notional_usdt": inventory_notional}))

        if reasons:
            for reason, payload in reasons:
                self.ledger.record_risk_event(reason=reason, payload=payload)
            self.executor.cancel_all(reason="risk_halt")
            return False
        return True

    def record_market_data_latency(
        self,
        event: BookTick | TopBookTick,
        now_ns: int,
        message_gap_ms: float | None = None,
    ) -> None:
        if message_gap_ms is None and self.last_event_receive_ns is not None:
            message_gap_ms = max(0.0, (event.receive_time_ns - self.last_event_receive_ns) / 1_000_000.0)
        self.last_event_receive_ns = event.receive_time_ns

        event_age_ms = 0.0
        if event.event_time_ms > 0:
            event_age_ms = max(0.0, time.time() * 1000.0 - event.event_time_ms)
        status_key = f"{event.source}:{event.book_status}:{event.stale}:{event.sequence_gap}"
        last_sample_ns = self.last_latency_sample_ns_by_source.get(event.source, 0)
        last_status = self.last_latency_status_by_source.get(event.source, "")
        should_record = (
            last_sample_ns == 0
            or now_ns - last_sample_ns >= 1_000_000_000
            or (event.source == "l2_depth" and status_key != last_status)
            or event.stale
            or event.sequence_gap
        )
        if not should_record:
            return
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
                    "message_gap_ms": message_gap_ms,
                    "update_id": event.update_id,
                    "stale": event.stale,
                    "sequence_gap": event.sequence_gap,
                    "resync_count": event.resync_count,
                    "sequence_gap_count": event.sequence_gap_count,
                    "stale_since_ns": event.stale_since_ns,
                },
            )
        )
        self.last_latency_sample_ns = now_ns
        self.last_latency_status = status_key
        self.last_latency_sample_ns_by_source[event.source] = now_ns
        self.last_latency_status_by_source[event.source] = status_key

    def on_book(self, book: BookTick | TopBookTick, now_ns: int, loop_lag_ms: float = 0.0) -> None:
        self.record_market_data_latency(book, now_ns=now_ns)
        self.executor.expire_orders(now_ns)
        if not self.evaluate_risk(book, now_ns=now_ns, last_event_receive_ns=book.receive_time_ns, loop_lag_ms=loop_lag_ms):
            return
        if self.last_quote_ns and now_ns - self.last_quote_ns < self.config.strategy.quote_interval_ms * 1_000_000:
            return
        quotes = self.strategy.generate_quotes(book, self.executor.inventory, now_ns=now_ns)
        self.executor.place_quotes(quotes, book=book)
        self.last_quote_ns = now_ns

    def on_trade(self, tick: TradeTick) -> None:
        self.record_trade_latency(tick, now_ns=tick.receive_time_ns)
        self.executor.on_trade(tick)

    def record_trade_latency(self, tick: TradeTick, now_ns: int) -> None:
        message_gap_ms = None
        if self.last_trade_receive_ns is not None:
            message_gap_ms = max(0.0, (tick.receive_time_ns - self.last_trade_receive_ns) / 1_000_000.0)
        self.last_trade_receive_ns = tick.receive_time_ns
        if self.last_trade_latency_sample_ns and now_ns - self.last_trade_latency_sample_ns < 1_000_000_000:
            return
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
                    "message_gap_ms": message_gap_ms,
                    "price": tick.price,
                    "qty": tick.qty,
                },
            )
        )
        self.last_trade_latency_sample_ns = now_ns

    async def run_synthetic_replay(self, duration_seconds: float) -> None:
        end_ns = time.monotonic_ns() + int(duration_seconds * 1_000_000_000)
        update_id = 1
        while time.monotonic_ns() < end_ns:
            now_ns = time.monotonic_ns()
            book = BookTick(
                symbol=self.config.symbol,
                event_time_ms=int(time.time() * 1000),
                receive_time_ns=now_ns,
                best_bid=99.0,
                best_bid_qty=5.0,
                best_ask=101.0,
                best_ask_qty=5.0,
                update_id=update_id,
            )
            self.on_book(book, now_ns=now_ns)
            update_id += 1
            await asyncio.sleep(self.config.strategy.quote_interval_ms / 1000.0)


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

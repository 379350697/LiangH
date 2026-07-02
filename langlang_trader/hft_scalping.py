from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass, field, replace
import json
import time
from pathlib import Path
from typing import Any

from langlang_trader.factor_research import (
    FactorResearchSample,
    FactorResearchStore,
    batch7_factor_registry,
    hft_candidate_features,
    observation_from_book,
)
from langlang_trader.ledger import Ledger
from langlang_trader.models import AccountSnapshot, OrderIntent, Position, Side, utc_now_iso
from liangh_trader.market_maker.binance_ws import BinanceUsdmWebSocketMarketData
from liangh_trader.market_maker.models import BookTick, TopBookTick, TradeTick


HFT_QUEUE_IMBALANCE_VERSION = "hft_queue_imbalance_one_tick_v1"
HFT_SWEEP_REPLENISHMENT_VERSION = "hft_sweep_replenishment_reversion_v1"
HFT_LEAD_LAG_VERSION = "hft_lead_lag_fair_value_v1"
HFT_SCALP_STRATEGY_VERSIONS = {
    HFT_QUEUE_IMBALANCE_VERSION,
    HFT_SWEEP_REPLENISHMENT_VERSION,
    HFT_LEAD_LAG_VERSION,
}
HFT_DEFAULT_FEE_BPS = 4.0
HFT_DEFAULT_MIN_NET_TAKE_PROFIT_BPS = 2.0


@dataclass(frozen=True)
class HftScalpVariant:
    variant_id: str
    symbol: str
    exchange_symbol: str
    strategy_kind: str
    position_size_usdt: float = 100.0
    leverage: int = 3
    max_spread_bps: float = 8.0
    stop_bps: float = 2.5
    take_profit_bps: float = 10.0
    time_stop_ms: int = 3_000
    max_slippage_bps: float = 4.0
    round_trip_fee_bps: float = 8.0
    min_net_take_profit_bps: float = 2.0
    take_profit_cost_floor_bps: float = 10.0
    min_queue_imbalance: float = 0.60
    min_sweep_notional_usdt: float = 20_000.0
    sweep_window_ms: int = 750
    replenishment_ratio: float = 0.45
    lead_exchange_symbol: str = "BTCUSDT"
    min_lead_move_bps: float = 6.0
    min_lag_divergence_bps: float = 3.0
    strategy_tree_variant_id: str = ""
    strategy_tree_parent_id: str = ""
    strategy_tree_path: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["strategy_tree_path"] = list(self.strategy_tree_path)
        return row


@dataclass(frozen=True)
class HftScalpSignal:
    symbol: str
    side: Side
    strength: float
    reason_codes: list[str]
    features: dict[str, Any]
    invalidation_price: float
    take_profit_hint: float
    strategy_version: str
    decision_trace: dict[str, Any]
    filter_codes: list[str] = field(default_factory=lambda: ["no_failure_filter"])
    take_profit_plan: dict[str, Any] = field(default_factory=dict)
    hold_plan: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)


@dataclass(frozen=True)
class HftScalpBotConfig:
    bot_id: str
    strategy_version: str
    variant: HftScalpVariant


@dataclass(frozen=True)
class HftScalpFleetConfig:
    run_id: str
    ledger_path: str
    symbols: list[str]
    exchange_symbols: list[str]
    bots: list[HftScalpBotConfig]
    allow_live_orders: bool = False
    initial_equity_usdt: float = 10_000.0
    fee_bps: float = HFT_DEFAULT_FEE_BPS
    slippage_bps: float = 2.0
    max_loop_lag_ms: float = 200.0
    factor_research_enabled: bool = True
    factor_research_db_path: str | None = None


class RulesQueueImbalanceOneTickStrategy:
    version = HFT_QUEUE_IMBALANCE_VERSION

    def __init__(self, variant: HftScalpVariant) -> None:
        self.variant = variant

    def on_book(self, book: BookTick | TopBookTick) -> HftScalpSignal | None:
        if not _book_matches(self.variant, book) or not _book_tradeable(book, self.variant):
            return None
        imbalance = _queue_imbalance(book)
        if imbalance >= self.variant.min_queue_imbalance:
            side = Side.LONG
        elif imbalance <= -self.variant.min_queue_imbalance:
            side = Side.SHORT
        else:
            return None
        entry = _entry_price(book, side)
        return _build_signal(
            variant=self.variant,
            version=self.version,
            side=side,
            entry=entry,
            reason_codes=["queue_imbalance_one_tick", f"variant:{self.variant.variant_id}"],
            features={
                "queue_imbalance": imbalance,
                "spread_bps": book.spread_bps,
                "book_source": book.source,
            },
        )


class RulesSweepReplenishmentReversionStrategy:
    version = HFT_SWEEP_REPLENISHMENT_VERSION

    def __init__(self, variant: HftScalpVariant) -> None:
        self.variant = variant
        self._last_sweep: dict[str, Any] | None = None

    def on_trade(self, tick: TradeTick) -> None:
        if tick.symbol != self.variant.exchange_symbol:
            return
        notional = max(0.0, tick.price * tick.qty)
        if notional < self.variant.min_sweep_notional_usdt:
            return
        self._last_sweep = {
            "direction": "sell" if tick.is_buyer_maker else "buy",
            "notional_usdt": notional,
            "price": tick.price,
            "receive_time_ns": tick.receive_time_ns,
            "trade_id": tick.trade_id,
        }

    def on_book(self, book: BookTick | TopBookTick) -> HftScalpSignal | None:
        if not _book_matches(self.variant, book) or not _book_tradeable(book, self.variant):
            return None
        sweep = self._last_sweep
        if sweep is None:
            return None
        age_ms = (book.receive_time_ns - int(sweep["receive_time_ns"])) / 1_000_000.0
        if age_ms < 0 or age_ms > self.variant.sweep_window_ms:
            return None
        bid_qty = float(book.best_bid_qty or 0.0)
        ask_qty = float(book.best_ask_qty or 0.0)
        if sweep["direction"] == "buy":
            if ask_qty > bid_qty * self.variant.replenishment_ratio:
                return None
            side = Side.SHORT
        else:
            if bid_qty > ask_qty * self.variant.replenishment_ratio:
                return None
            side = Side.LONG
        entry = _entry_price(book, side)
        return _build_signal(
            variant=self.variant,
            version=self.version,
            side=side,
            entry=entry,
            reason_codes=["sweep_replenishment_failed", f"variant:{self.variant.variant_id}"],
            features={
                "sweep_direction": sweep["direction"],
                "sweep_notional_usdt": sweep["notional_usdt"],
                "sweep_age_ms": age_ms,
                "bid_qty": bid_qty,
                "ask_qty": ask_qty,
                "replenishment_ratio": self.variant.replenishment_ratio,
            },
        )


class RulesLeadLagFairValueStrategy:
    version = HFT_LEAD_LAG_VERSION

    def __init__(self, variant: HftScalpVariant) -> None:
        self.variant = variant
        self._previous_mid_by_symbol: dict[str, float] = {}
        self._last_lead_move_bps = 0.0
        self._last_lead_mid = 0.0

    def on_book(self, book: BookTick | TopBookTick) -> HftScalpSignal | None:
        if not _book_tradeable(book, self.variant):
            return None
        mid = book.mid_price
        if mid is None or mid <= 0:
            return None
        if book.symbol == self.variant.lead_exchange_symbol:
            previous = self._previous_mid_by_symbol.get(book.symbol)
            if previous and previous > 0:
                self._last_lead_move_bps = (mid / previous - 1.0) * 10_000.0
            self._last_lead_mid = mid
            self._previous_mid_by_symbol[book.symbol] = mid
            return None
        if not _book_matches(self.variant, book):
            return None
        previous_lag = self._previous_mid_by_symbol.get(book.symbol, mid)
        self._previous_mid_by_symbol[book.symbol] = mid
        lead_move = self._last_lead_move_bps
        if abs(lead_move) < self.variant.min_lead_move_bps:
            return None
        lag_move = (mid / previous_lag - 1.0) * 10_000.0 if previous_lag > 0 else 0.0
        divergence = lead_move - lag_move
        if abs(divergence) < self.variant.min_lag_divergence_bps:
            return None
        side = Side.LONG if divergence > 0 else Side.SHORT
        entry = _entry_price(book, side)
        return _build_signal(
            variant=self.variant,
            version=self.version,
            side=side,
            entry=entry,
            reason_codes=["lead_lag_fair_value", f"variant:{self.variant.variant_id}"],
            features={
                "lead_exchange_symbol": self.variant.lead_exchange_symbol,
                "lead_mid": self._last_lead_mid,
                "lag_mid": mid,
                "lead_move_bps": lead_move,
                "lag_move_bps": lag_move,
                "divergence_bps": divergence,
            },
        )


@dataclass
class _OpenHftPosition:
    bot_id: str
    trade_id: str
    variant: HftScalpVariant
    strategy_version: str
    side: Side
    entry_price: float
    qty: float
    stop_loss: float
    take_profit: float
    opened_at_ns: int
    entry_fee: float


class HftScalpPaperRunner:
    def __init__(
        self,
        *,
        run_id: str,
        ledger: Ledger,
        bots: list[tuple[str, HftScalpVariant, str]],
        initial_equity_usdt: float = 10_000.0,
        fee_bps: float = 4.0,
        exchange: str = "binance",
        factor_research_store: FactorResearchStore | None = None,
    ) -> None:
        self.run_id = run_id
        self.ledger = ledger
        self.initial_equity_usdt = initial_equity_usdt
        self.fee_bps = fee_bps
        self.exchange = exchange
        self.factor_research_store = factor_research_store
        if self.factor_research_store is not None:
            self.factor_research_store.register_factors(batch7_factor_registry())
        normalized_bots = [
            (bot_id, _variant_with_cost_floor(variant, fee_bps=fee_bps), strategy_version)
            for bot_id, variant, strategy_version in bots
        ]
        self._strategies = {
            bot_id: _strategy_for_bot(strategy_version, variant)
            for bot_id, variant, strategy_version in normalized_bots
        }
        self._bot_variants = {bot_id: variant for bot_id, variant, _ in normalized_bots}
        self._bot_versions = {bot_id: strategy_version for bot_id, _, strategy_version in normalized_bots}
        self._open_positions: dict[str, _OpenHftPosition] = {}
        self._realized_pnl_by_bot: dict[str, float] = {}
        self._restore_open_positions()

    @classmethod
    def from_config(cls, config: HftScalpFleetConfig) -> "HftScalpPaperRunner":
        return cls(
            run_id=config.run_id,
            ledger=Ledger(config.ledger_path),
            bots=[(bot.bot_id, bot.variant, bot.strategy_version) for bot in config.bots],
            initial_equity_usdt=config.initial_equity_usdt,
            fee_bps=config.fee_bps,
            exchange="binance",
            factor_research_store=(
                FactorResearchStore(config.factor_research_db_path)
                if config.factor_research_enabled and config.factor_research_db_path
                else None
            ),
        )

    def on_trade(self, tick: TradeTick) -> None:
        for strategy in self._strategies.values():
            if hasattr(strategy, "on_trade"):
                strategy.on_trade(tick)

    def on_book(self, book: BookTick | TopBookTick, *, loop_lag_ms: float = 0.0) -> None:
        self._record_factor_observation(book)
        closed_bot_ids: set[str] = set()
        for bot_id, position in list(self._open_positions.items()):
            if book.symbol == position.variant.exchange_symbol:
                if self._maybe_close_position(bot_id, book, loop_lag_ms=loop_lag_ms):
                    closed_bot_ids.add(bot_id)
        for bot_id, strategy in self._strategies.items():
            if bot_id in closed_bot_ids or bot_id in self._open_positions or not hasattr(strategy, "on_book"):
                continue
            signal = strategy.on_book(book)
            self._record_factor_sample(bot_id, strategy, book, signal)
            if signal is None:
                continue
            self._open_position(bot_id, signal, book.receive_time_ns)

    def _record_factor_observation(self, book: BookTick | TopBookTick) -> None:
        if self.factor_research_store is None:
            return
        self.factor_research_store.record_observation(observation_from_book(book))

    def _record_factor_sample(
        self,
        bot_id: str,
        strategy: Any,
        book: BookTick | TopBookTick,
        signal: HftScalpSignal | None,
    ) -> None:
        if self.factor_research_store is None:
            return
        variant = self._bot_variants[bot_id]
        if not _book_is_research_candidate(variant, book):
            return
        decision_time_ns = max(time.monotonic_ns(), int(book.receive_time_ns))
        features = hft_candidate_features(variant, book, signal=signal, strategy=strategy)
        event_seq = int(book.update_id or book.receive_time_ns)
        sample = FactorResearchSample(
            sample_id=f"{self.run_id}:{bot_id}:{book.symbol}:{book.receive_time_ns}:{event_seq}",
            run_id=self.run_id,
            bot_id=bot_id,
            strategy_tree_id=variant.strategy_tree_variant_id or variant.variant_id,
            symbol=book.symbol,
            venue=getattr(book, "venue", "binance_usdm"),
            event_seq=event_seq,
            exchange_event_time_ms=book.event_time_ms,
            receive_time_ns=book.receive_time_ns,
            decision_time_ns=decision_time_ns,
            sample_type="hft_book_decision",
            fired=signal is not None,
            side=signal.side.value if signal is not None else "",
            mid_price=float(book.mid_price or 0.0),
            features=features,
            feature_times_ns={key: int(book.receive_time_ns) for key in features},
            data_quality_flags=tuple(
                flag
                for flag, active in {
                    "stale_book": book.stale,
                    "sequence_gap": book.sequence_gap,
                    "not_hot_book": book.book_status != "hot",
                }.items()
                if active
            ),
        )
        self.factor_research_store.record_sample(sample)

    def _open_position(self, bot_id: str, signal: HftScalpSignal, now_ns: int) -> None:
        variant = self._bot_variants[bot_id]
        strategy_version = self._bot_versions[bot_id]
        entry = float(signal.features["entry_price"])
        qty = variant.position_size_usdt / max(entry, 1e-12)
        fee = entry * qty * self.fee_bps / 10_000.0
        bot_ledger = self._bot_ledger(bot_id, variant)
        signal_id = bot_ledger.record_signal(signal, strategy_version)
        intent = OrderIntent(
            symbol=variant.symbol,
            side=signal.side,
            order_type="market",
            qty=qty,
            leverage=variant.leverage,
            reduce_only=False,
            entry_reason=signal.reason_codes[0],
            stop_loss=signal.invalidation_price,
            max_slippage_bps=variant.max_slippage_bps,
            strategy_version=strategy_version,
            decision_trace=signal.decision_trace,
        )
        intent_id = bot_ledger.record_order_intent(intent, signal_id=signal_id)
        order_id = bot_ledger.record_order(intent, status="filled", intent_id=intent_id, raw_payload={"paper_only": True})
        fill_id = bot_ledger.record_fill(
            order_id=order_id,
            exchange_order_id=f"hft-entry-{order_id}",
            symbol=variant.symbol,
            side=signal.side,
            qty=qty,
            price=entry,
            fee=fee,
            liquidity="taker",
            strategy_version=strategy_version,
            decision_trace=signal.decision_trace,
        )
        trade_id = bot_ledger.record_trade_fill(intent=intent, order_id=order_id, fill_id=fill_id, price=entry, fee=fee)
        if trade_id is None:
            raise RuntimeError(f"HFT entry did not create a trade lifecycle for {bot_id}:{variant.symbol}")
        bot_ledger.upsert_position(
            Position(
                symbol=variant.symbol,
                side=signal.side,
                qty=qty,
                avg_price=entry,
                leverage=variant.leverage,
                exchange=self.exchange,
                strategy_version=strategy_version,
            )
        )
        bot_ledger.record_equity_snapshot(
            AccountSnapshot(
                equity_usdt=self.initial_equity_usdt - fee,
                cash_usdt=self.initial_equity_usdt - fee,
                margin_used_usdt=variant.position_size_usdt / max(variant.leverage, 1),
                realized_pnl_usdt=self._realized_pnl_by_bot.get(bot_id, 0.0) - fee,
            ),
            strategy_version=strategy_version,
            decision_trace={"event_count": 1, "entry_reason": signal.reason_codes[0]},
        )
        self._open_positions[bot_id] = _OpenHftPosition(
            bot_id=bot_id,
            trade_id=trade_id,
            variant=variant,
            strategy_version=strategy_version,
            side=signal.side,
            entry_price=entry,
            qty=qty,
            stop_loss=signal.invalidation_price,
            take_profit=signal.take_profit_hint,
            opened_at_ns=now_ns,
            entry_fee=fee,
        )

    def _maybe_close_position(self, bot_id: str, book: BookTick | TopBookTick, *, loop_lag_ms: float) -> bool:
        position = self._open_positions[bot_id]
        mark_price = _exit_price(book, position.side)
        if mark_price is None or mark_price <= 0:
            return False
        self._bot_ledger(bot_id, position.variant).record_trade_mark(
            symbol=position.variant.symbol,
            mark_price=mark_price,
            trade_id=position.trade_id,
        )
        if position.side is Side.LONG:
            if mark_price >= position.take_profit:
                self._close_position(bot_id, mark_price, "take_profit_exit")
                return True
            elif mark_price <= position.stop_loss:
                self._close_position(bot_id, mark_price, "stop_loss_exit")
                return True
            elif book.receive_time_ns - position.opened_at_ns >= position.variant.time_stop_ms * 1_000_000:
                self._close_position(bot_id, mark_price, "time_or_guard_exit")
                return True
        else:
            if mark_price <= position.take_profit:
                self._close_position(bot_id, mark_price, "take_profit_exit")
                return True
            elif mark_price >= position.stop_loss:
                self._close_position(bot_id, mark_price, "stop_loss_exit")
                return True
            elif (
                loop_lag_ms > 0
                or book.receive_time_ns - position.opened_at_ns >= position.variant.time_stop_ms * 1_000_000
            ):
                self._close_position(bot_id, mark_price, "time_or_guard_exit")
                return True
        return False

    def _close_position(self, bot_id: str, price: float, exit_reason: str) -> None:
        position = self._open_positions.pop(bot_id)
        bot_ledger = self._bot_ledger(bot_id, position.variant)
        exit_side = Side.SHORT if position.side is Side.LONG else Side.LONG
        exit_fee = price * position.qty * self.fee_bps / 10_000.0
        gross_pnl = (price - position.entry_price) * position.qty * position.side.sign
        realized = gross_pnl - position.entry_fee - exit_fee
        self._realized_pnl_by_bot[bot_id] = self._realized_pnl_by_bot.get(bot_id, 0.0) + realized
        trace = {
            "exit_reason": exit_reason,
            "entry_trade_id": position.trade_id,
            "entry_price": position.entry_price,
            "exit_price": price,
            "exit_semantics": "full_tp_sl",
        }
        intent = OrderIntent(
            symbol=position.variant.symbol,
            side=exit_side,
            order_type="market",
            qty=position.qty,
            leverage=position.variant.leverage,
            reduce_only=True,
            entry_reason="reduce_only_close",
            stop_loss=None,
            max_slippage_bps=position.variant.max_slippage_bps,
            strategy_version=position.strategy_version,
            exit_reason=exit_reason,
            decision_trace=trace,
        )
        intent_id = bot_ledger.record_order_intent(intent)
        order_id = bot_ledger.record_order(
            intent,
            status="filled",
            intent_id=intent_id,
            raw_payload={"paper_only": True, "reduce_only": True},
        )
        fill_id = bot_ledger.record_fill(
            order_id=order_id,
            exchange_order_id=f"hft-exit-{order_id}",
            symbol=position.variant.symbol,
            side=exit_side,
            qty=position.qty,
            price=price,
            fee=exit_fee,
            liquidity="taker",
            strategy_version=position.strategy_version,
            exit_reason=exit_reason,
            decision_trace=trace,
        )
        bot_ledger.record_trade_fill(intent=intent, order_id=order_id, fill_id=fill_id, price=price, fee=exit_fee)
        bot_ledger.delete_position(position.variant.symbol, exchange=self.exchange)
        bot_ledger.record_equity_snapshot(
            AccountSnapshot(
                equity_usdt=self.initial_equity_usdt + self._realized_pnl_by_bot[bot_id],
                cash_usdt=self.initial_equity_usdt + self._realized_pnl_by_bot[bot_id],
                margin_used_usdt=0.0,
                realized_pnl_usdt=self._realized_pnl_by_bot[bot_id],
            ),
            strategy_version=position.strategy_version,
            decision_trace=trace,
        )

    def _restore_open_positions(self) -> None:
        for bot_id, variant in self._bot_variants.items():
            bot_ledger = self._bot_ledger(bot_id, variant)
            bot_ledger.reconcile_open_trades_with_position(variant.symbol, exchange=self.exchange)
            position = bot_ledger.get_position(variant.symbol, exchange=self.exchange)
            if position is None:
                continue
            exit_state = bot_ledger.open_trade_exit_state(variant.symbol, exchange=self.exchange)
            if exit_state is None:
                continue
            entry = float(exit_state["entry_price"])
            side = position.side
            self._open_positions[bot_id] = _OpenHftPosition(
                bot_id=bot_id,
                trade_id=str(exit_state["trade_id"]),
                variant=variant,
                strategy_version=self._bot_versions[bot_id],
                side=side,
                entry_price=entry,
                qty=position.qty,
                stop_loss=float(exit_state["current_stop_loss"]),
                take_profit=_take_profit_price(entry, side, variant.take_profit_bps),
                opened_at_ns=time.monotonic_ns(),
                entry_fee=0.0,
            )

    def _bot_ledger(self, bot_id: str, variant: HftScalpVariant) -> Ledger:
        return self.ledger.scoped(
            run_id=self.run_id,
            bot_id=bot_id,
            variant_id=variant.variant_id,
            exchange=self.exchange,
        )


def load_hft_scalp_fleet_config(path: str | Path) -> HftScalpFleetConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    bot_rows = raw.get("bots", [])
    if not bot_rows and "bot_matrix" in raw:
        bot_rows = _expand_bot_matrix(raw)
    fee_bps = float(raw.get("paper", {}).get("fee_bps", HFT_DEFAULT_FEE_BPS))
    ledger_path = str(raw["ledger_path"])
    research = raw.get("factor_research", raw.get("research", {}))
    research_db_path = research.get("db_path") or str(Path(ledger_path).with_name("batch7_factor_research.sqlite3"))
    bots = [
        HftScalpBotConfig(
            bot_id=str(row["bot_id"]),
            strategy_version=str(row["strategy_version"]),
            variant=_variant_with_cost_floor(_variant_from_dict(row["variant"]), fee_bps=fee_bps),
        )
        for row in bot_rows
    ]
    return HftScalpFleetConfig(
        run_id=str(raw["run_id"]),
        ledger_path=ledger_path,
        symbols=[str(row) for row in raw.get("symbols", [])],
        exchange_symbols=[str(row) for row in raw.get("exchange_symbols", [])],
        bots=bots,
        allow_live_orders=bool(raw.get("execution", {}).get("allow_live_orders", False)),
        initial_equity_usdt=float(raw.get("paper", {}).get("initial_equity_usdt", 10_000.0)),
        fee_bps=fee_bps,
        slippage_bps=float(raw.get("paper", {}).get("slippage_bps", 2.0)),
        max_loop_lag_ms=float(raw.get("risk", {}).get("max_loop_lag_ms", 200.0)),
        factor_research_enabled=bool(research.get("enabled", True)),
        factor_research_db_path=research_db_path,
    )


async def run_hft_scalp_fleet(config: HftScalpFleetConfig, duration_seconds: float) -> None:
    if config.allow_live_orders:
        raise PermissionError("batch7 HFT scalping signal fleet is paper-only")
    runner = HftScalpPaperRunner.from_config(config)
    deadline_ns = time.monotonic_ns() + int(duration_seconds * 1_000_000_000)
    await asyncio.gather(
        *[
            _run_symbol_stream(exchange_symbol, runner, deadline_ns=deadline_ns)
            for exchange_symbol in config.exchange_symbols
        ]
    )


async def _run_symbol_stream(exchange_symbol: str, runner: HftScalpPaperRunner, *, deadline_ns: int) -> None:
    market_data = BinanceUsdmWebSocketMarketData(exchange_symbol)
    async for event in market_data.stream():
        if isinstance(event, TradeTick):
            runner.on_trade(event)
        elif isinstance(event, (BookTick, TopBookTick)):
            runner.on_book(event)
        if time.monotonic_ns() >= deadline_ns:
            break


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the batch7 event-driven HFT scalping paper fleet")
    parser.add_argument("--config", required=True)
    parser.add_argument("--duration-seconds", type=float, default=60.0)
    args = parser.parse_args(argv)
    config = load_hft_scalp_fleet_config(args.config)
    asyncio.run(run_hft_scalp_fleet(config, duration_seconds=args.duration_seconds))
    return 0


def _variant_from_dict(raw: dict[str, Any]) -> HftScalpVariant:
    row = dict(raw)
    if "strategy_tree_path" in row:
        row["strategy_tree_path"] = tuple(row["strategy_tree_path"])
    return HftScalpVariant(**row)


def _variant_with_cost_floor(
    variant: HftScalpVariant,
    *,
    fee_bps: float,
) -> HftScalpVariant:
    round_trip_fee_bps = 2.0 * fee_bps
    min_net_take_profit_bps = variant.min_net_take_profit_bps
    take_profit_cost_floor_bps = round_trip_fee_bps + min_net_take_profit_bps
    if variant.take_profit_bps + 1e-12 < take_profit_cost_floor_bps:
        raise ValueError(
            f"{variant.variant_id} take_profit_bps={variant.take_profit_bps} is below "
            f"cost floor {take_profit_cost_floor_bps:.1f} "
            f"(round_trip_fee_bps={round_trip_fee_bps:.1f}, "
            f"min_net_take_profit_bps={min_net_take_profit_bps:.1f})"
        )
    return replace(
        variant,
        round_trip_fee_bps=round_trip_fee_bps,
        min_net_take_profit_bps=min_net_take_profit_bps,
        take_profit_cost_floor_bps=take_profit_cost_floor_bps,
    )


def _expand_bot_matrix(raw: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    symbols = raw.get("symbol_matrix", [])
    templates = raw.get("bot_matrix", {}).get("strategies", [])
    for symbol_row in symbols:
        slug = str(symbol_row["slug"])
        symbol = str(symbol_row["symbol"])
        exchange_symbol = str(symbol_row["exchange_symbol"])
        lead_exchange_symbol = str(symbol_row.get("lead_exchange_symbol", "BTCUSDT"))
        for template in templates:
            strategy_key = str(template["strategy_key"])
            variant_id = f"{template['variant_prefix']}_{slug}_v1"
            bot_id = f"{template['bot_prefix']}_{slug}_paper"
            strategy_tree_parent_id = str(template["strategy_tree_parent_id"])
            strategy_tree_path = [
                "scalping",
                "batch7_hft_scalp",
                strategy_key,
                variant_id,
            ]
            variant = {
                "variant_id": variant_id,
                "symbol": symbol,
                "exchange_symbol": exchange_symbol,
                "strategy_kind": strategy_key.removeprefix("hft_"),
                "strategy_tree_parent_id": strategy_tree_parent_id,
                "strategy_tree_variant_id": variant_id,
                "strategy_tree_path": strategy_tree_path,
                **template.get("parameters", {}),
            }
            if strategy_key == "hft_lead_lag_fair_value":
                variant["lead_exchange_symbol"] = lead_exchange_symbol
            rows.append(
                {
                    "bot_id": bot_id,
                    "strategy_version": str(template["strategy_version"]),
                    "variant": variant,
                }
            )
    return rows


def _strategy_for_bot(strategy_version: str, variant: HftScalpVariant):
    if strategy_version == HFT_QUEUE_IMBALANCE_VERSION:
        return RulesQueueImbalanceOneTickStrategy(variant)
    if strategy_version == HFT_SWEEP_REPLENISHMENT_VERSION:
        return RulesSweepReplenishmentReversionStrategy(variant)
    if strategy_version == HFT_LEAD_LAG_VERSION:
        return RulesLeadLagFairValueStrategy(variant)
    raise ValueError(f"unsupported HFT scalping strategy version: {strategy_version}")


def _build_signal(
    *,
    variant: HftScalpVariant,
    version: str,
    side: Side,
    entry: float,
    reason_codes: list[str],
    features: dict[str, Any],
) -> HftScalpSignal:
    stop = _stop_price(entry, side, variant.stop_bps)
    take_profit = _take_profit_price(entry, side, variant.take_profit_bps)
    trace = _tree_trace(variant)
    payload = {
        **features,
        **trace,
        "entry_price": entry,
        "stop_bps": variant.stop_bps,
        "take_profit_bps": variant.take_profit_bps,
        "round_trip_fee_bps": variant.round_trip_fee_bps,
        "min_net_take_profit_bps": variant.min_net_take_profit_bps,
        "take_profit_cost_floor_bps": variant.take_profit_cost_floor_bps,
        "time_stop_ms": variant.time_stop_ms,
        "strategy_kind": variant.strategy_kind,
        "variant": variant.to_dict(),
    }
    return HftScalpSignal(
        symbol=variant.symbol,
        side=side,
        strength=min(1.0, max(0.35, abs(float(features.get("queue_imbalance", 0.5))))),
        reason_codes=reason_codes,
        features=payload,
        invalidation_price=stop,
        take_profit_hint=take_profit,
        strategy_version=version,
        decision_trace={
            **trace,
            "exit_semantics": "full_tp_sl",
            "round_trip_fee_bps": variant.round_trip_fee_bps,
            "min_net_take_profit_bps": variant.min_net_take_profit_bps,
            "take_profit_cost_floor_bps": variant.take_profit_cost_floor_bps,
            "reason_codes": reason_codes,
            "features": payload,
        },
        take_profit_plan={
            "mode": "full_position",
            "take_profit_bps": variant.take_profit_bps,
            "round_trip_fee_bps": variant.round_trip_fee_bps,
            "min_net_take_profit_bps": variant.min_net_take_profit_bps,
            "take_profit_cost_floor_bps": variant.take_profit_cost_floor_bps,
        },
        hold_plan={"time_stop_ms": variant.time_stop_ms},
    )


def _tree_trace(variant: HftScalpVariant) -> dict[str, Any]:
    return {
        "strategy_tree_variant_id": variant.strategy_tree_variant_id or variant.variant_id,
        "strategy_tree_parent_id": variant.strategy_tree_parent_id or variant.strategy_kind,
        "strategy_tree_path": list(variant.strategy_tree_path),
    }


def _book_matches(variant: HftScalpVariant, book: BookTick | TopBookTick) -> bool:
    return book.symbol == variant.exchange_symbol


def _book_is_research_candidate(variant: HftScalpVariant, book: BookTick | TopBookTick) -> bool:
    if book.symbol == variant.exchange_symbol:
        return True
    return variant.strategy_kind == "lead_lag_fair_value" and book.symbol == variant.lead_exchange_symbol


def _book_tradeable(book: BookTick | TopBookTick, variant: HftScalpVariant) -> bool:
    if book.source != "l2_depth" or book.book_status != "hot":
        return False
    if book.stale or book.sequence_gap:
        return False
    if book.best_bid is None or book.best_ask is None or book.best_bid <= 0 or book.best_ask <= 0:
        return False
    if book.best_bid >= book.best_ask:
        return False
    spread_bps = book.spread_bps
    return spread_bps is not None and spread_bps <= variant.max_spread_bps


def _queue_imbalance(book: BookTick | TopBookTick) -> float:
    bid_qty = float(book.best_bid_qty or 0.0)
    ask_qty = float(book.best_ask_qty or 0.0)
    total = bid_qty + ask_qty
    if total <= 0:
        return 0.0
    return (bid_qty - ask_qty) / total


def _entry_price(book: BookTick | TopBookTick, side: Side) -> float:
    if side is Side.LONG:
        return float(book.best_ask or book.mid_price or 0.0)
    return float(book.best_bid or book.mid_price or 0.0)


def _exit_price(book: BookTick | TopBookTick, side: Side) -> float | None:
    if side is Side.LONG:
        return book.best_bid
    return book.best_ask


def _stop_price(entry: float, side: Side, stop_bps: float) -> float:
    if side is Side.LONG:
        return entry * (1.0 - stop_bps / 10_000.0)
    return entry * (1.0 + stop_bps / 10_000.0)


def _take_profit_price(entry: float, side: Side, take_profit_bps: float) -> float:
    if side is Side.LONG:
        return entry * (1.0 + take_profit_bps / 10_000.0)
    return entry * (1.0 - take_profit_bps / 10_000.0)


if __name__ == "__main__":
    raise SystemExit(main())

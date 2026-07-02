from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from statistics import fmean
from typing import Any

from langlang_trader.models import Candle, OrderBook, Side, Signal, utc_now_iso


class EntryMode(str, Enum):
    BREAKOUT = "breakout"
    FRACTAL_CONFIRM = "fractal_confirm"


class OrderFlowMode(str, Enum):
    STRONG = "strong"
    WEAK = "weak"


@dataclass(frozen=True)
class FiveBarScalpConfig:
    scalp_bar: str = "5s"
    entry_mode: EntryMode = EntryMode.BREAKOUT
    trend_bars: tuple[str, ...] = ("1m", "3m", "5m")
    min_aligned_trend_bars: int = 1
    require_order_flow: bool = True
    order_flow_mode: str = OrderFlowMode.STRONG.value
    min_order_flow_imbalance: float = 0.20
    min_depth_replenish: float = 0.05
    min_queue_ratio: float = 0.55
    max_spread_bps: float = 6.0
    total_cost_bps: float = 4.0
    stop_buffer_bps: float = 2.0
    min_stop_bps: float = 8.0
    max_stop_bps: float = 35.0
    stop_loss_policy: str = "fractal_extreme_buffer_with_min_max_risk"
    min_range_cost_multiple: float = 3.0
    take_profit_r: float = 1.5
    min_take_profit_cost_multiple: float = 3.0
    time_stop_bars: int = 5
    position_size_multiplier: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["entry_mode"] = self.entry_mode.value
        data["trend_bars"] = list(self.trend_bars)
        return data


@dataclass(frozen=True)
class OrderFlowWindow:
    buy_volume: float
    sell_volume: float
    bid_depth_change: float
    ask_depth_change: float
    latest_spread_bps: float

    @property
    def imbalance(self) -> float:
        total = self.buy_volume + self.sell_volume
        if total <= 0:
            return 0.0
        return (self.buy_volume - self.sell_volume) / total


@dataclass(frozen=True)
class FiveBarFractal:
    side: Side
    center_ts: int
    extreme_price: float
    breakout_price: float
    reason_code: str


@dataclass(frozen=True)
class ScalpSignal:
    symbol: str
    side: Side
    entry_trigger: float
    stop_loss: float
    take_profit: float
    time_stop_bars: int
    entry_mode: str
    reason_codes: list[str]
    filter_codes: list[str]
    features: dict[str, Any]
    created_at: str = field(default_factory=utc_now_iso)


@dataclass(frozen=True)
class ScalpDecision:
    action: str
    explanation: str
    filter_codes: list[str]
    signal: ScalpSignal | None = None


@dataclass(frozen=True)
class ScalpTrade:
    symbol: str
    side: Side
    entry_price: float
    exit_price: float
    fee_bps: float
    spread_bps: float
    slippage_bps: float
    bucket: str = "unknown"
    qty: float = 1.0

    @property
    def gross_bps(self) -> float:
        if self.entry_price <= 0:
            return 0.0
        return self.side.sign * ((self.exit_price / self.entry_price) - 1.0) * 10_000

    @property
    def total_cost_bps(self) -> float:
        return self.fee_bps + self.spread_bps + self.slippage_bps

    @property
    def net_bps(self) -> float:
        return self.gross_bps - self.total_cost_bps


@dataclass(frozen=True)
class ScalpingVariant:
    variant_id: str
    symbol: str
    scalp_bar: str = "5s"
    allowed_side: str = "both"
    entry_mode: str = EntryMode.BREAKOUT.value
    trend_bars: tuple[str, ...] = ("1m", "3m", "5m")
    min_aligned_trend_bars: int = 1
    require_order_flow: bool = True
    order_flow_mode: str = OrderFlowMode.STRONG.value
    min_order_flow_imbalance: float = 0.20
    min_depth_replenish: float = 0.05
    min_queue_ratio: float = 0.55
    max_spread_bps: float = 6.0
    total_cost_bps: float = 4.0
    stop_buffer_bps: float = 2.0
    min_stop_bps: float = 8.0
    max_stop_bps: float = 35.0
    take_profit_r: float = 1.5
    min_take_profit_cost_multiple: float = 3.0
    time_stop_bars: int = 5
    position_size_multiplier: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["trend_bars"] = list(self.trend_bars)
        return row

    def to_config(self) -> FiveBarScalpConfig:
        return FiveBarScalpConfig(
            scalp_bar=self.scalp_bar,
            entry_mode=EntryMode(self.entry_mode),
            trend_bars=tuple(self.trend_bars),
            min_aligned_trend_bars=self.min_aligned_trend_bars,
            require_order_flow=self.require_order_flow,
            order_flow_mode=self.order_flow_mode,
            min_order_flow_imbalance=self.min_order_flow_imbalance,
            min_depth_replenish=self.min_depth_replenish,
            min_queue_ratio=self.min_queue_ratio,
            max_spread_bps=self.max_spread_bps,
            total_cost_bps=self.total_cost_bps,
            stop_buffer_bps=self.stop_buffer_bps,
            min_stop_bps=self.min_stop_bps,
            max_stop_bps=self.max_stop_bps,
            take_profit_r=self.take_profit_r,
            min_take_profit_cost_multiple=self.min_take_profit_cost_multiple,
            time_stop_bars=self.time_stop_bars,
            position_size_multiplier=self.position_size_multiplier,
        )


class FiveBarScalpStrategy:
    version = "five_bar_fractal_scalp_v1"

    def __init__(self, config: FiveBarScalpConfig | None = None):
        self.config = config or FiveBarScalpConfig()

    def evaluate(
        self,
        *,
        symbol: str,
        scalp_candles: list[Candle],
        trend_candles_by_bar: dict[str, list[Candle]],
        order_flow: OrderFlowWindow | None,
        order_book: OrderBook | None = None,
    ) -> ScalpDecision:
        rows = sorted(scalp_candles, key=lambda item: item.ts)
        if len(rows) < 5:
            return _skip("skip:need_5_scalp_bars", ["insufficient_scalp_bars"])

        window = rows[-5:]
        fractal = detect_five_bar_fractal(window)
        if fractal is None:
            return _skip("skip:no_5_bar_fractal", ["no_5_bar_fractal"])

        trend = _trend_alignment(fractal.side, trend_candles_by_bar, self.config)
        if not trend["aligned"]:
            return _skip("skip:trend_not_aligned", ["trend_not_aligned"])

        spread_bps = _effective_spread_bps(order_flow, order_book)
        if spread_bps > self.config.max_spread_bps:
            return _skip("skip:spread_too_wide", ["spread_too_wide"])

        range_bps = _range_bps(window)
        min_range_bps = self.config.total_cost_bps * self.config.min_range_cost_multiple
        if range_bps < min_range_bps:
            return _skip("skip:range_too_small_after_cost", ["range_too_small_after_cost"])

        order_flow_features = _order_flow_features(fractal.side, order_flow, order_book, self.config)
        if self.config.require_order_flow and not order_flow_features["confirmed"]:
            return _skip("skip:order_flow_not_confirmed", ["order_flow_not_confirmed"])

        entry_price = _entry_trigger(fractal, window, self.config.entry_mode)
        stop = _stop_loss(fractal, entry_price, self.config)
        if stop["rejected"]:
            return _skip("skip:stop_distance_too_wide", ["stop_distance_too_wide"])
        stop_loss = float(stop["price"])
        risk = abs(entry_price - stop_loss)
        min_profit = entry_price * (self.config.total_cost_bps * self.config.min_take_profit_cost_multiple / 10_000)
        target_distance = max(risk * self.config.take_profit_r, min_profit)
        take_profit = entry_price + fractal.side.sign * target_distance
        reason_codes = [fractal.reason_code, "higher_timeframe_trend_aligned"]
        if self.config.entry_mode is EntryMode.FRACTAL_CONFIRM:
            reason_codes.append("entry_mode_fractal_confirm")
        else:
            reason_codes.append("entry_mode_breakout")
        if order_flow_features["confirmed"]:
            if order_flow_features["tier"] == OrderFlowMode.WEAK.value:
                reason_codes.append("order_flow_weak_confirmed")
            else:
                reason_codes.append(
                    "order_flow_reclaim_confirmed" if fractal.side is Side.LONG else "order_flow_breakdown_confirmed"
                )
        else:
            reason_codes.append("order_flow_ablation_mode")

        features = {
            "strategy_version": self.version,
            "fractal_extreme_price": fractal.extreme_price,
            "range_bps": range_bps,
            "effective_spread_bps": spread_bps,
            "trend_alignment": trend,
            "order_flow": order_flow_features,
            "stop_loss": stop,
            "position_size_multiplier": self.config.position_size_multiplier,
            "config": self.config.to_dict(),
        }
        signal = ScalpSignal(
            symbol=symbol,
            side=fractal.side,
            entry_trigger=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            time_stop_bars=self.config.time_stop_bars,
            entry_mode=self.config.entry_mode.value,
            reason_codes=reason_codes,
            filter_codes=["no_failure_filter"],
            features=features,
        )
        return ScalpDecision(
            action="enter",
            explanation=f"enter:{fractal.side.value}_5_bar_fractal_scalp",
            filter_codes=["no_failure_filter"],
            signal=signal,
        )


class RulesFiveBarScalpStrategy:
    version = FiveBarScalpStrategy.version

    def __init__(self, variant: ScalpingVariant):
        self.variant = variant
        self.engine = FiveBarScalpStrategy(variant.to_config())

    def generate_from_market_data(
        self,
        *,
        symbol: str,
        candles_by_bar: dict[str, list[Candle]],
        order_book: OrderBook | None,
    ) -> Signal | None:
        if symbol != self.variant.symbol:
            return None
        scalp_candles = candles_by_bar.get(self.variant.scalp_bar, [])
        order_flow = order_flow_from_recent_candles(scalp_candles[-8:], order_book)
        decision = self.engine.evaluate(
            symbol=symbol,
            scalp_candles=scalp_candles,
            trend_candles_by_bar={bar: candles_by_bar.get(bar, []) for bar in self.variant.trend_bars},
            order_flow=order_flow,
            order_book=order_book,
        )
        if decision.signal is None:
            return None
        scalp_signal = decision.signal
        return Signal(
            symbol=scalp_signal.symbol,
            side=scalp_signal.side,
            strength=_signal_strength(scalp_signal.features),
            reason_codes=scalp_signal.reason_codes,
            features={
                **scalp_signal.features,
                "scalp_bar": self.variant.scalp_bar,
                "entry_trigger": scalp_signal.entry_trigger,
                "time_stop_bars": scalp_signal.time_stop_bars,
                "filter_codes": scalp_signal.filter_codes,
                "position_size_multiplier": self.variant.position_size_multiplier,
                "variant": self.variant.to_dict(),
            },
            invalidation_price=scalp_signal.stop_loss,
            take_profit_hint=scalp_signal.take_profit,
            created_at=scalp_signal.created_at,
        )


def detect_five_bar_fractal(candles: list[Candle]) -> FiveBarFractal | None:
    rows = sorted(candles, key=lambda item: item.ts)[-5:]
    if len(rows) < 5:
        return None
    lows = [row.low for row in rows]
    highs = [row.high for row in rows]
    center = rows[2]
    if lows[0] > lows[1] > lows[2] and lows[2] < lows[3] < lows[4]:
        return FiveBarFractal(
            side=Side.LONG,
            center_ts=center.ts,
            extreme_price=center.low,
            breakout_price=max(rows[3].high, rows[4].high),
            reason_code="bullish_5_bar_fractal",
        )
    if highs[0] < highs[1] < highs[2] and highs[2] > highs[3] > highs[4]:
        return FiveBarFractal(
            side=Side.SHORT,
            center_ts=center.ts,
            extreme_price=center.high,
            breakout_price=min(rows[3].low, rows[4].low),
            reason_code="bearish_5_bar_fractal",
        )
    return None


def _trend_alignment(side: Side, candles_by_bar: dict[str, list[Candle]], config: FiveBarScalpConfig) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}
    aligned_count = 0
    available_count = 0
    for bar in config.trend_bars:
        rows = sorted(candles_by_bar.get(bar, []), key=lambda item: item.ts)
        if len(rows) < 2:
            checks[bar] = {"available": False, "aligned": False, "slope": 0.0}
            continue
        available_count += 1
        closes = [row.close for row in rows]
        slope = closes[-1] - fmean(closes[:-1])
        aligned = slope > 0 if side is Side.LONG else slope < 0
        aligned_count += 1 if aligned else 0
        checks[bar] = {"available": True, "aligned": aligned, "slope": slope}

    required = min(config.min_aligned_trend_bars, available_count)
    return {
        "aligned": available_count > 0 and aligned_count >= max(1, required),
        "aligned_count": aligned_count,
        "available_count": available_count,
        "checks": checks,
    }


def _order_flow_features(
    side: Side,
    order_flow: OrderFlowWindow | None,
    order_book: OrderBook | None,
    config: FiveBarScalpConfig,
) -> dict[str, Any]:
    mode = config.order_flow_mode
    imbalance = order_flow.imbalance if order_flow is not None else 0.0
    bid_depth_change = order_flow.bid_depth_change if order_flow is not None else 0.0
    ask_depth_change = order_flow.ask_depth_change if order_flow is not None else 0.0
    queue_ratio = _queue_ratio(order_book)
    if mode == OrderFlowMode.WEAK.value:
        if side is Side.LONG:
            confirmed = imbalance >= config.min_order_flow_imbalance
        else:
            confirmed = imbalance <= -config.min_order_flow_imbalance
    elif side is Side.LONG:
        confirmed = (
            imbalance >= config.min_order_flow_imbalance
            and bid_depth_change >= config.min_depth_replenish
            and ask_depth_change <= -config.min_depth_replenish
            and queue_ratio >= config.min_queue_ratio
        )
    else:
        confirmed = (
            imbalance <= -config.min_order_flow_imbalance
            and bid_depth_change <= -config.min_depth_replenish
            and ask_depth_change >= config.min_depth_replenish
            and queue_ratio <= 1 - config.min_queue_ratio
        )
    return {
        "confirmed": confirmed,
        "tier": mode,
        "imbalance": imbalance,
        "bid_depth_change": bid_depth_change,
        "ask_depth_change": ask_depth_change,
        "queue_ratio": queue_ratio,
    }


def _queue_ratio(order_book: OrderBook | None) -> float:
    if order_book is None:
        return 0.5
    bid_qty = sum(level.qty for level in order_book.bids[:3])
    ask_qty = sum(level.qty for level in order_book.asks[:3])
    total = bid_qty + ask_qty
    if total <= 0:
        return 0.5
    return bid_qty / total


def _effective_spread_bps(order_flow: OrderFlowWindow | None, order_book: OrderBook | None) -> float:
    spreads = []
    if order_flow is not None:
        spreads.append(order_flow.latest_spread_bps)
    if order_book is not None and order_book.bids and order_book.asks:
        bid = order_book.bids[0].price
        ask = order_book.asks[0].price
        mid = (bid + ask) / 2
        if mid > 0:
            spreads.append(max(0.0, (ask - bid) / mid * 10_000))
    return max(spreads) if spreads else 0.0


def _range_bps(candles: list[Candle]) -> float:
    high = max(row.high for row in candles)
    low = min(row.low for row in candles)
    close = candles[-1].close
    if close <= 0:
        return 0.0
    return (high - low) / close * 10_000


def _entry_trigger(fractal: FiveBarFractal, candles: list[Candle], entry_mode: EntryMode) -> float:
    if entry_mode is EntryMode.FRACTAL_CONFIRM:
        return candles[-1].close
    return fractal.breakout_price


def order_flow_from_recent_candles(candles: list[Candle], order_book: OrderBook | None) -> OrderFlowWindow:
    buy_volume = 0.0
    sell_volume = 0.0
    for row in candles:
        body = row.close - row.open
        body_weight = min(1.0, abs(body) / max(row.high - row.low, 1e-12))
        directional_volume = row.volume * (0.5 + body_weight * 0.5)
        passive_volume = max(0.0, row.volume - directional_volume)
        if body >= 0:
            buy_volume += directional_volume
            sell_volume += passive_volume
        else:
            sell_volume += directional_volume
            buy_volume += passive_volume
    queue_ratio = _queue_ratio(order_book)
    depth_edge = queue_ratio - 0.5
    return OrderFlowWindow(
        buy_volume=buy_volume,
        sell_volume=sell_volume,
        bid_depth_change=depth_edge,
        ask_depth_change=-depth_edge,
        latest_spread_bps=_effective_spread_bps(None, order_book),
    )


def _stop_loss(fractal: FiveBarFractal, entry_price: float, config: FiveBarScalpConfig) -> dict[str, Any]:
    if entry_price <= 0:
        return {"rejected": True, "reason": "invalid_entry_price"}
    buffer = config.stop_buffer_bps / 10_000
    if fractal.side is Side.LONG:
        raw_price = fractal.extreme_price * (1 - buffer)
        raw_risk_bps = (entry_price - raw_price) / entry_price * 10_000
        if raw_price >= entry_price or raw_risk_bps > config.max_stop_bps:
            return {
                "rejected": True,
                "reason": "stop_distance_too_wide_or_invalid",
                "raw_price": raw_price,
                "raw_risk_bps": raw_risk_bps,
                "max_stop_bps": config.max_stop_bps,
            }
        price = min(raw_price, entry_price * (1 - config.min_stop_bps / 10_000))
    else:
        raw_price = fractal.extreme_price * (1 + buffer)
        raw_risk_bps = (raw_price - entry_price) / entry_price * 10_000
        if raw_price <= entry_price or raw_risk_bps > config.max_stop_bps:
            return {
                "rejected": True,
                "reason": "stop_distance_too_wide_or_invalid",
                "raw_price": raw_price,
                "raw_risk_bps": raw_risk_bps,
                "max_stop_bps": config.max_stop_bps,
            }
        price = max(raw_price, entry_price * (1 + config.min_stop_bps / 10_000))
    risk_bps = abs(entry_price - price) / entry_price * 10_000
    return {
        "rejected": False,
        "policy": config.stop_loss_policy,
        "price": price,
        "raw_price": raw_price,
        "raw_risk_bps": raw_risk_bps,
        "risk_bps": risk_bps,
        "min_stop_bps": config.min_stop_bps,
        "max_stop_bps": config.max_stop_bps,
        "buffer_bps": config.stop_buffer_bps,
    }


def _signal_strength(features: dict[str, Any]) -> float:
    trend = features.get("trend_alignment", {})
    order_flow = features.get("order_flow", {})
    aligned_count = float(trend.get("aligned_count", 0) or 0)
    imbalance = abs(float(order_flow.get("imbalance", 0.0) or 0.0))
    return min(0.95, 0.55 + aligned_count * 0.06 + imbalance * 0.25)


def _skip(explanation: str, filters: list[str]) -> ScalpDecision:
    return ScalpDecision(action="skip", explanation=explanation, filter_codes=filters)


def summarize_scalp_trades(trades: list[ScalpTrade]) -> dict[str, Any]:
    net_values = [trade.net_bps for trade in trades]
    return {
        **_summarize_net_values(net_values),
        "by_symbol": _summarize_grouped(trades, key=lambda trade: trade.symbol),
        "by_bucket": _summarize_grouped(trades, key=lambda trade: trade.bucket),
    }


def _summarize_grouped(trades: list[ScalpTrade], *, key: Any) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[ScalpTrade]] = {}
    for trade in trades:
        grouped.setdefault(str(key(trade)), []).append(trade)
    return {name: _summarize_net_values([trade.net_bps for trade in rows]) for name, rows in grouped.items()}


def _summarize_net_values(values: list[float]) -> dict[str, Any]:
    wins = [value for value in values if value > 0]
    losses = [value for value in values if value < 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = float("inf") if gross_loss == 0 and gross_profit > 0 else gross_profit / gross_loss if gross_loss else 0.0
    return {
        "trade_count": len(values),
        "net_expectancy_bps": fmean(values) if values else 0.0,
        "gross_profit_bps": gross_profit,
        "gross_loss_bps": gross_loss,
        "profit_factor": profit_factor,
        "win_rate": len(wins) / len(values) if values else 0.0,
        "max_drawdown_bps": _max_drawdown(values),
    }


def _max_drawdown(values: list[float]) -> float:
    cumulative = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for value in values:
        cumulative += value
        peak = max(peak, cumulative)
        max_drawdown = max(max_drawdown, peak - cumulative)
    return max_drawdown

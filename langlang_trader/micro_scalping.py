from __future__ import annotations

from dataclasses import asdict, dataclass, field
from statistics import fmean
from typing import Any

from langlang_trader.models import Candle, OrderBook, Side, utc_now_iso


@dataclass(frozen=True)
class MicroScalpVariant:
    variant_id: str
    symbol: str
    strategy_kind: str
    bar: str = "5s"
    lookback_bars: int = 20
    max_spread_bps: float = 6.0
    total_cost_bps: float = 4.0
    min_edge_cost_multiple: float = 2.0
    stop_bps: float = 12.0
    max_stop_bps: float = 30.0
    take_profit_r: float = 1.2
    runner_take_profit_r: float = 2.4
    time_stop_bars: int = 10
    position_size_multiplier: float = 0.5
    min_ofi: float = 0.20
    min_microprice_edge_bps: float = 1.0
    vwap_deviation_bps: float = 8.0
    breakout_lookback_bars: int = 12
    min_volume_ratio: float = 1.4
    min_basis_bps: float = 8.0
    pair_stop_bps: float = 12.0
    strategy_tree_variant_id: str = ""
    strategy_tree_parent_id: str = ""
    strategy_tree_path: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["strategy_tree_path"] = list(self.strategy_tree_path)
        return row


@dataclass(frozen=True)
class MicroScalpSignal:
    symbol: str
    side: Side
    strength: float
    reason_codes: list[str]
    features: dict[str, Any]
    invalidation_price: float
    take_profit_hint: float | None
    strategy_version: str
    decision_trace: dict[str, Any]
    take_profit_plan: dict[str, Any] = field(default_factory=dict)
    hold_plan: dict[str, Any] = field(default_factory=dict)
    filter_codes: list[str] = field(default_factory=lambda: ["no_failure_filter"])
    created_at: str = field(default_factory=utc_now_iso)


@dataclass(frozen=True)
class ShadowPairSignal:
    variant_id: str
    symbol: str
    strategy_version: str
    strategy_tree_variant_id: str
    strategy_tree_parent_id: str
    strategy_tree_path: list[str]
    perp_side: str
    hedge_side: str
    entry_price: float
    basis_bps: float
    funding_rate: float
    stop_basis_bps: float
    take_profit_basis_bps: float
    time_stop_seconds: int
    features: dict[str, Any]
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RulesOfiMicropriceScalpStrategy:
    version = "scalp_ofi_microprice_directional_v1"

    def __init__(self, variant: MicroScalpVariant):
        self.variant = variant

    def generate_from_market_data(
        self,
        *,
        symbol: str,
        candles_by_bar: dict[str, list[Candle]],
        order_book: OrderBook | None,
        market_metrics: dict[str, Any] | None = None,
    ) -> MicroScalpSignal | None:
        if symbol != self.variant.symbol:
            return None
        rows = _rows(self.variant, candles_by_bar)
        if len(rows) < 3 or order_book is None:
            return None
        book = microstructure_from_book(order_book)
        if not _book_tradeable(book, self.variant):
            return None
        momentum_bps = _momentum_bps(rows)
        projected_edge_bps = max(abs(book["microprice_edge_bps"]), abs(momentum_bps))
        min_edge_bps = self.variant.total_cost_bps * self.variant.min_edge_cost_multiple
        if projected_edge_bps + 1e-9 < min_edge_bps:
            return None
        ofi = float(book["queue_imbalance"])
        if ofi >= self.variant.min_ofi and book["microprice_edge_bps"] >= self.variant.min_microprice_edge_bps:
            side = Side.LONG
        elif ofi <= -self.variant.min_ofi and book["microprice_edge_bps"] <= -self.variant.min_microprice_edge_bps:
            side = Side.SHORT
        else:
            return None
        entry = rows[-1].close
        return _build_signal(
            version=self.version,
            variant=self.variant,
            symbol=symbol,
            side=side,
            entry=entry,
            reason_codes=["ofi_microprice_directional", f"variant:{self.variant.variant_id}"],
            features={
                "order_book": book,
                "momentum_bps": momentum_bps,
                "projected_edge_bps": projected_edge_bps,
                "market_metrics": market_metrics or {},
            },
        )


class RulesVwapMeanReversionScalpStrategy:
    version = "scalp_vwap_mean_reversion_v1"

    def __init__(self, variant: MicroScalpVariant):
        self.variant = variant

    def generate_from_market_data(
        self,
        *,
        symbol: str,
        candles_by_bar: dict[str, list[Candle]],
        order_book: OrderBook | None,
        market_metrics: dict[str, Any] | None = None,
    ) -> MicroScalpSignal | None:
        if symbol != self.variant.symbol:
            return None
        rows = _rows(self.variant, candles_by_bar)
        if len(rows) < 3:
            return None
        book = microstructure_from_book(order_book) if order_book is not None else {"spread_bps": 0.0}
        if float(book["spread_bps"]) > self.variant.max_spread_bps:
            return None
        latest = rows[-1].close
        vwap = _vwap(rows[-self.variant.lookback_bars :])
        if latest <= 0 or vwap <= 0:
            return None
        deviation_bps = (latest / vwap - 1.0) * 10_000.0
        min_edge = max(self.variant.vwap_deviation_bps, self.variant.total_cost_bps * self.variant.min_edge_cost_multiple)
        if abs(deviation_bps) < min_edge:
            return None
        side = Side.SHORT if deviation_bps > 0 else Side.LONG
        return _build_signal(
            version=self.version,
            variant=self.variant,
            symbol=symbol,
            side=side,
            entry=latest,
            reason_codes=["vwap_mean_reversion", f"variant:{self.variant.variant_id}"],
            features={
                "vwap": vwap,
                "vwap_deviation_bps": deviation_bps,
                "order_book": book,
                "market_metrics": market_metrics or {},
            },
        )


class RulesVolatilityBreakoutScalpStrategy:
    version = "scalp_volatility_breakout_v1"

    def __init__(self, variant: MicroScalpVariant):
        self.variant = variant

    def generate_from_market_data(
        self,
        *,
        symbol: str,
        candles_by_bar: dict[str, list[Candle]],
        order_book: OrderBook | None,
        market_metrics: dict[str, Any] | None = None,
    ) -> MicroScalpSignal | None:
        if symbol != self.variant.symbol:
            return None
        rows = _rows(self.variant, candles_by_bar)
        lookback = max(3, self.variant.breakout_lookback_bars)
        if len(rows) < lookback + 1:
            return None
        book = microstructure_from_book(order_book) if order_book is not None else {"spread_bps": 0.0}
        if float(book["spread_bps"]) > self.variant.max_spread_bps:
            return None
        latest = rows[-1]
        previous = rows[-lookback - 1 : -1]
        upper = max(row.high for row in previous)
        lower = min(row.low for row in previous)
        avg_volume = fmean([row.volume for row in previous]) if previous else 0.0
        volume_ratio = latest.volume / avg_volume if avg_volume > 0 else 0.0
        if volume_ratio < self.variant.min_volume_ratio:
            return None
        if latest.close > upper:
            side = Side.LONG
            breakout_distance_bps = (latest.close / upper - 1.0) * 10_000.0
        elif latest.close < lower:
            side = Side.SHORT
            breakout_distance_bps = (lower / latest.close - 1.0) * 10_000.0
        else:
            return None
        if breakout_distance_bps < self.variant.total_cost_bps * self.variant.min_edge_cost_multiple:
            return None
        return _build_signal(
            version=self.version,
            variant=self.variant,
            symbol=symbol,
            side=side,
            entry=latest.close,
            reason_codes=["volatility_breakout", f"variant:{self.variant.variant_id}"],
            features={
                "breakout_lookback_bars": lookback,
                "breakout_upper": upper,
                "breakout_lower": lower,
                "breakout_distance_bps": breakout_distance_bps,
                "volume_ratio": volume_ratio,
                "order_book": book,
                "market_metrics": market_metrics or {},
            },
        )


class RulesFundingBasisShadowStrategy:
    version = "scalp_funding_basis_delta_neutral_v1"

    def __init__(self, variant: MicroScalpVariant):
        self.variant = variant

    def generate_from_market_data(
        self,
        *,
        symbol: str,
        candles_by_bar: dict[str, list[Candle]],
        order_book: OrderBook | None,
        market_metrics: dict[str, Any] | None = None,
    ) -> None:
        return None

    def generate_shadow_pair_from_market_data(
        self,
        *,
        symbol: str,
        market_metrics: dict[str, Any],
    ) -> ShadowPairSignal | None:
        if symbol != self.variant.symbol:
            return None
        basis_bps = _float(market_metrics.get("basis_bps"))
        if basis_bps == 0.0:
            mark = _float(market_metrics.get("mark_price"))
            index = _float(market_metrics.get("index_price"))
            basis_bps = (mark / index - 1.0) * 10_000.0 if mark > 0 and index > 0 else 0.0
        funding_rate = _float(market_metrics.get("funding_rate_last"))
        if abs(basis_bps) < self.variant.min_basis_bps:
            return None
        if basis_bps > 0 and funding_rate >= 0:
            perp_side = "short"
            hedge_side = "long_spot_or_inverse"
            stop_basis = basis_bps + self.variant.pair_stop_bps
            take_profit = max(0.0, basis_bps - self.variant.pair_stop_bps)
        elif basis_bps < 0 and funding_rate <= 0:
            perp_side = "long"
            hedge_side = "short_spot_or_inverse"
            stop_basis = basis_bps - self.variant.pair_stop_bps
            take_profit = min(0.0, basis_bps + self.variant.pair_stop_bps)
        else:
            return None
        trace = _tree_trace(self.version, self.variant)
        entry_price = _float(market_metrics.get("mark_price")) or _float(market_metrics.get("index_price"))
        features = {
            **trace,
            "strategy_kind": self.variant.strategy_kind,
            "basis_bps": basis_bps,
            "funding_rate": funding_rate,
            "risk": {
                "pair_stop_bps": self.variant.pair_stop_bps,
                "time_stop_seconds": self.variant.time_stop_bars * 60,
                "paper_only": True,
            },
            "variant": self.variant.to_dict(),
        }
        return ShadowPairSignal(
            variant_id=self.variant.variant_id,
            symbol=symbol,
            strategy_version=self.version,
            strategy_tree_variant_id=str(trace["strategy_tree_variant_id"]),
            strategy_tree_parent_id=str(trace["strategy_tree_parent_id"]),
            strategy_tree_path=list(trace["strategy_tree_path"]),
            perp_side=perp_side,
            hedge_side=hedge_side,
            entry_price=entry_price,
            basis_bps=basis_bps,
            funding_rate=funding_rate,
            stop_basis_bps=stop_basis,
            take_profit_basis_bps=take_profit,
            time_stop_seconds=self.variant.time_stop_bars * 60,
            features=features,
        )


def microstructure_from_book(order_book: OrderBook) -> dict[str, float]:
    if not order_book.bids or not order_book.asks:
        return {
            "best_bid": 0.0,
            "best_ask": 0.0,
            "mid": 0.0,
            "spread_bps": float("inf"),
            "queue_imbalance": 0.0,
            "microprice": 0.0,
            "microprice_edge_bps": 0.0,
        }
    bid = order_book.bids[0]
    ask = order_book.asks[0]
    if bid.price <= 0 or ask.price <= 0 or bid.price >= ask.price:
        spread_bps = float("inf")
        mid = 0.0
    else:
        mid = (bid.price + ask.price) / 2.0
        spread_bps = (ask.price - bid.price) / mid * 10_000.0
    total_qty = max(bid.qty + ask.qty, 1e-12)
    queue_imbalance = (bid.qty - ask.qty) / total_qty
    microprice = (ask.price * bid.qty + bid.price * ask.qty) / total_qty
    microprice_edge_bps = (microprice / mid - 1.0) * 10_000.0 if mid > 0 else 0.0
    return {
        "best_bid": bid.price,
        "best_ask": ask.price,
        "mid": mid,
        "spread_bps": spread_bps,
        "queue_imbalance": queue_imbalance,
        "microprice": microprice,
        "microprice_edge_bps": microprice_edge_bps,
    }


def _build_signal(
    *,
    version: str,
    variant: MicroScalpVariant,
    symbol: str,
    side: Side,
    entry: float,
    reason_codes: list[str],
    features: dict[str, Any],
) -> MicroScalpSignal:
    stop_bps = min(max(variant.stop_bps, variant.total_cost_bps * variant.min_edge_cost_multiple), variant.max_stop_bps)
    if side is Side.LONG:
        stop = entry * (1.0 - stop_bps / 10_000.0)
        take_profit = entry + (entry - stop) * variant.take_profit_r
    else:
        stop = entry * (1.0 + stop_bps / 10_000.0)
        take_profit = entry - (stop - entry) * variant.take_profit_r
    trace = _tree_trace(version, variant)
    signal_features = {
        **features,
        **trace,
        "strategy_version": version,
        "strategy_kind": variant.strategy_kind,
        "entry_price": entry,
        "time_stop_bars": variant.time_stop_bars,
        "take_profit_r": variant.take_profit_r,
        "runner_take_profit_r": variant.runner_take_profit_r,
        "position_size_multiplier": variant.position_size_multiplier,
        "risk": {
            "strict_hard_stop_bps": stop_bps,
            "max_stop_bps": variant.max_stop_bps,
            "take_profit_r": variant.take_profit_r,
            "runner_take_profit_r": variant.runner_take_profit_r,
            "time_stop_bars": variant.time_stop_bars,
            "max_spread_bps": variant.max_spread_bps,
            "total_cost_bps": variant.total_cost_bps,
        },
        "variant": variant.to_dict(),
    }
    decision_trace = {
        **trace,
        "strategy_version": version,
        "strategy_kind": variant.strategy_kind,
        "time_stop_bars": variant.time_stop_bars,
        "take_profit_r": variant.take_profit_r,
        "runner_take_profit_r": variant.runner_take_profit_r,
        "position_size_multiplier": variant.position_size_multiplier,
        "hard_stop_bps": stop_bps,
        "reason_codes": list(reason_codes),
    }
    take_profit_plan = {
        "take_profit_r": variant.take_profit_r,
        "partial_r": variant.take_profit_r,
        "partial_exit_fraction": 0.5,
        "runner_r": variant.runner_take_profit_r,
        "time_stop_bars": variant.time_stop_bars,
    }
    strength = min(0.95, 0.55 + min(0.20, abs(stop_bps) / 100.0) + 0.10)
    return MicroScalpSignal(
        symbol=symbol,
        side=side,
        strength=strength,
        reason_codes=reason_codes,
        features=signal_features,
        invalidation_price=stop,
        take_profit_hint=take_profit,
        take_profit_plan=take_profit_plan,
        hold_plan={"time_stop_bars": variant.time_stop_bars},
        strategy_version=version,
        decision_trace=decision_trace,
    )


def _tree_trace(version: str, variant: MicroScalpVariant) -> dict[str, Any]:
    variant_id = variant.strategy_tree_variant_id or variant.variant_id
    parent_id = variant.strategy_tree_parent_id or version
    path = list(variant.strategy_tree_path) if variant.strategy_tree_path else ["scalping", version, variant_id]
    return {
        "strategy_tree_variant_id": variant_id,
        "strategy_tree_parent_id": parent_id,
        "strategy_tree_path": path,
    }


def _rows(variant: MicroScalpVariant, candles_by_bar: dict[str, list[Candle]]) -> list[Candle]:
    rows = candles_by_bar.get(variant.bar, [])
    return sorted(rows, key=lambda row: row.ts)[-variant.lookback_bars :]


def _book_tradeable(book: dict[str, float], variant: MicroScalpVariant) -> bool:
    return book["mid"] > 0 and book["spread_bps"] <= variant.max_spread_bps


def _momentum_bps(rows: list[Candle]) -> float:
    if len(rows) < 2 or rows[0].close <= 0:
        return 0.0
    return (rows[-1].close / rows[0].close - 1.0) * 10_000.0


def _vwap(rows: list[Candle]) -> float:
    notional = sum(row.close * row.volume for row in rows)
    volume = sum(row.volume for row in rows)
    return notional / volume if volume > 0 else 0.0


def _float(value: Any) -> float:
    try:
        if value in {None, ""}:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Side(str, Enum):
    LONG = "long"
    SHORT = "short"

    @property
    def sign(self) -> int:
        return 1 if self is Side.LONG else -1

    @property
    def okx_order_side(self) -> str:
        return "buy" if self is Side.LONG else "sell"

    @classmethod
    def from_value(cls, value: str | "Side") -> "Side":
        if isinstance(value, cls):
            return value
        normalized = value.lower()
        if normalized in {"long", "buy"}:
            return cls.LONG
        if normalized in {"short", "sell"}:
            return cls.SHORT
        raise ValueError(f"unknown side: {value}")


class MarketRegime(str, Enum):
    PRE_MAIN_UPTREND = "pre_main_uptrend"
    MAIN_UPTREND = "main_uptrend"
    STRONG_PULLBACK = "strong_pullback"
    BREAKOUT_RETEST = "breakout_retest"
    FIRST_DIVERGENCE = "first_divergence"
    SECOND_DIVERGENCE = "second_divergence"
    TOP_DIVERGENCE = "top_divergence"
    POST_LARGE_DIVERGENCE = "post_large_divergence"
    WEAK_WATERFALL = "weak_waterfall"
    CHOPPY_INVALID = "choppy_invalid"


class EntrySetup(str, Enum):
    STARTER_BUY = "starter_buy"
    SMALL_DIVERGENCE_ENTRY = "small_divergence_entry"
    FIRST_BREAKOUT = "first_breakout"
    FIRST_PULLBACK = "first_pullback"
    SECOND_ENTRY = "second_entry"
    SECOND_PRESSURE_RETEST = "second_pressure_retest"
    PLATFORM_RETEST = "platform_retest"
    POST_DIVERGENCE_REBOUND = "post_divergence_rebound"
    TOP_SHORT = "top_short"
    FINAL_TOP_SHORT = "final_top_short"
    BOX_REBOUND_LONG = "box_rebound_long"
    SHORT_REBOUND_FAILURE = "short_rebound_failure"
    WATERFALL_CONTINUATION = "waterfall_continuation"


class FailureFilter(str, Enum):
    CHASE_OVERHEAT = "chase_overheat"
    STRUCTURE_BREAK = "structure_break"
    VOLUME_REVERSAL = "volume_reversal"
    LARGE_DIVERGENCE = "large_divergence"
    SUPER_LARGE_DIVERGENCE = "super_large_divergence"
    LOW_LIQUIDITY = "low_liquidity"
    LIQUIDITY_RANK_FILTERED = "liquidity_rank_filtered"
    BIG_LOSS_SIMILARITY = "big_loss_similarity"
    INSUFFICIENT_UPSIDE_SPACE = "insufficient_upside_space"
    FIRST_10X_TOO_HIGH = "first_10x_too_high"
    HIGH_POSITION_NO_STRUCTURE = "high_position_no_structure"
    NO_BOTTOM_LIFT = "no_bottom_lift"
    STOP_LOSS_CLUSTER = "stop_loss_cluster"
    EMOTIONAL_REVENGE_PROXY = "emotional_revenge_proxy"
    NO_HISTORICAL_SUPPORT = "no_historical_support"
    VARIANT_SIDE_NOT_ALLOWED = "variant_side_not_allowed"
    THIRD_SMALL_DIVERGENCE = "third_small_divergence"
    BTC_DIVERGENCE_ALT_BREAKOUT = "btc_divergence_alt_breakout"
    COUNTER_TREND_SHORT_DISABLED = "counter_trend_short_disabled"
    FALSE_BREAKOUT_AFTER_CONTRACTION = "false_breakout_after_contraction"
    FIVE_WAVE_LATE_RISK = "five_wave_late_risk"
    AUTUMN_WINTER_REDUCED_FREQUENCY = "autumn_winter_reduced_frequency"
    CATCH_UP_NO_RUNNER = "catch_up_no_runner"
    BOX_REBOUND_LOW_QUALITY = "box_rebound_low_quality"
    NO_FAILURE_FILTER = "no_failure_filter"


class ExitPlan(str, Enum):
    STRUCTURE_STOP = "structure_stop"
    MAE_STOP = "mae_stop"
    TIME_STOP = "time_stop"
    RUNNER_HOLD = "runner_hold"
    PARTIAL_TAKE_PROFIT = "partial_take_profit"
    TREND_BREAK_EXIT = "trend_break_exit"


class StrategyAction(str, Enum):
    ENTER = "enter"
    SKIP = "skip"
    HOLD = "hold"
    REDUCE = "reduce"
    CLOSE = "close"


@dataclass(frozen=True)
class Candle:
    symbol: str
    bar: str
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    vol_ccy: float | None = None
    vol_quote: float | None = None
    source: str = ""


@dataclass(frozen=True)
class Ticker:
    symbol: str
    ts: int
    last: float
    bid: float | None = None
    ask: float | None = None
    volume_24h: float | None = None


@dataclass(frozen=True)
class OrderBookLevel:
    price: float
    qty: float


@dataclass(frozen=True)
class OrderBook:
    symbol: str
    ts: int
    bids: list[OrderBookLevel]
    asks: list[OrderBookLevel]


@dataclass(frozen=True)
class Signal:
    symbol: str
    side: Side
    strength: float
    reason_codes: list[str]
    features: dict[str, Any]
    invalidation_price: float
    take_profit_hint: float | None
    created_at: str = field(default_factory=utc_now_iso)


@dataclass(frozen=True)
class LangLangSignal:
    symbol: str
    side: Side
    strength: float
    reason_codes: list[str]
    filter_codes: list[str]
    features: dict[str, Any]
    invalidation_price: float
    stop_loss: float
    take_profit_hint: float | None
    take_profit_plan: dict[str, Any]
    hold_plan: dict[str, Any]
    strategy_version: str
    regime: MarketRegime
    setup: EntrySetup
    decision_trace: dict[str, Any] = field(default_factory=dict)
    historical_match_score: float | None = None
    matched_trade_examples: list[dict[str, Any]] = field(default_factory=list)
    risk_notes: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now_iso)


@dataclass(frozen=True)
class StrategyDecision:
    action: StrategyAction
    explanation: str
    matched_historical_patterns: list[dict[str, Any]]
    risk_notes: list[str]
    filter_codes: list[FailureFilter] = field(default_factory=list)
    signal: LangLangSignal | None = None


@dataclass(frozen=True)
class DistillDataset:
    trades: list[dict[str, Any]]
    candles_by_symbol_bar: dict[str, dict[str, list[Candle]]]
    trade_labels: dict[str, list[str]]
    feature_rows: list[dict[str, Any]]
    coverage: dict[str, Any]


@dataclass(frozen=True)
class OrderIntent:
    symbol: str
    side: Side
    order_type: str
    qty: float
    leverage: int
    reduce_only: bool
    entry_reason: str
    stop_loss: float | None
    max_slippage_bps: float
    strategy_version: str | None = None
    regime: str | None = None
    setup: str | None = None
    exit_reason: str | None = None
    decision_trace: dict[str, Any] = field(default_factory=dict)
    historical_match_score: float | None = None


@dataclass(frozen=True)
class OrderResult:
    exchange_order_id: str
    status: str
    filled_qty: float
    avg_price: float | None
    fee: float
    raw_payload: dict[str, Any]


@dataclass(frozen=True)
class AccountSnapshot:
    equity_usdt: float
    cash_usdt: float
    margin_used_usdt: float
    realized_pnl_usdt: float = 0.0


@dataclass(frozen=True)
class Position:
    symbol: str
    side: Side
    qty: float
    avg_price: float
    leverage: int
    unrealized_pnl: float = 0.0
    exchange: str = "okx"
    strategy_version: str | None = None
    regime: str | None = None
    setup: str | None = None


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return {k: to_jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    return value

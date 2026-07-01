from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BookTick:
    symbol: str
    event_time_ms: int
    receive_time_ns: int
    best_bid: float | None
    best_bid_qty: float | None
    best_ask: float | None
    best_ask_qty: float | None
    update_id: int | None = None
    sequence_gap: bool = False
    stale: bool = False
    venue: str = "binance_usdm"
    source: str = "l2_depth"
    book_status: str = "hot"
    resync_count: int = 0
    sequence_gap_count: int = 0
    stale_since_ns: int | None = None

    @property
    def mid_price(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid + self.best_ask) / 2.0

    @property
    def spread_bps(self) -> float | None:
        mid = self.mid_price
        if mid is None or mid <= 0:
            return None
        return (self.best_ask - self.best_bid) / mid * 10_000.0


@dataclass(frozen=True)
class TradeTick:
    symbol: str
    event_time_ms: int
    receive_time_ns: int
    price: float
    qty: float
    is_buyer_maker: bool
    trade_id: str | int
    venue: str = "binance_usdm"


@dataclass(frozen=True)
class TopBookTick(BookTick):
    source: str = "book_ticker"
    book_status: str = "top_book"


@dataclass(frozen=True)
class QuoteIntent:
    symbol: str
    side: str
    price: float
    qty: float
    created_at_ns: int
    ttl_ms: int
    post_only: bool
    strategy_version: str
    strategy_tree_variant_id: str
    strategy_tree_parent_id: str
    strategy_tree_path: list[str] = field(default_factory=list)
    venue: str = "binance_usdm"
    quote_id: str | None = None


@dataclass
class LimitOrderState:
    order_id: str
    quote_id: str | None
    symbol: str
    side: str
    price: float
    qty: float
    remaining_qty: float
    status: str
    post_only: bool
    created_at_ns: int
    updated_at_ns: int
    expires_at_ns: int
    strategy_version: str
    strategy_tree_variant_id: str
    strategy_tree_parent_id: str
    strategy_tree_path: list[str] = field(default_factory=list)
    venue: str = "binance_usdm"


@dataclass
class InventoryState:
    symbol: str
    base_qty: float = 0.0
    quote_usdt: float = 0.0
    avg_price: float = 0.0
    realized_pnl_usdt: float = 0.0
    fees_usdt: float = 0.0


@dataclass(frozen=True)
class FillEvent:
    fill_id: str
    order_id: str
    symbol: str
    side: str
    price: float
    qty: float
    fee_usdt: float
    liquidity: str
    trade_id: str | int
    event_time_ms: int
    receive_time_ns: int
    inventory_base_qty: float
    inventory_quote_usdt: float
    strategy_version: str
    strategy_tree_variant_id: str
    strategy_tree_parent_id: str
    strategy_tree_path: list[str] = field(default_factory=list)
    venue: str = "binance_usdm"


@dataclass(frozen=True)
class LatencySample:
    name: str
    sample_time_ns: int
    latency_ms: float
    symbol: str
    venue: str = "binance_usdm"
    payload: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class OrderTruthEvent:
    symbol: str
    order_id: str
    client_order_id: str
    event_type: str
    order_status: str
    execution_type: str
    side: str
    price: float
    qty: float
    filled_qty: float
    last_fill_qty: float
    last_fill_price: float
    event_time_ms: int
    transaction_time_ms: int
    receive_time_ns: int
    truth_source: str
    strategy_version: str
    strategy_tree_variant_id: str
    strategy_tree_parent_id: str
    strategy_tree_path: list[str] = field(default_factory=list)
    venue: str = "binance_usdm"
    payload: dict[str, object] = field(default_factory=dict)

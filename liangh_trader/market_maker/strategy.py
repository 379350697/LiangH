from __future__ import annotations

from typing import Protocol

from .config import MarketMakerConfig
from .models import BookTick, InventoryState, QuoteIntent


class MarketMakerStrategy(Protocol):
    def generate_quotes(self, book: BookTick, inventory: InventoryState, now_ns: int) -> list[QuoteIntent]:
        ...


class ReferencePassiveMakerStrategy:
    def __init__(self, config: MarketMakerConfig) -> None:
        self.config = config

    def generate_quotes(self, book: BookTick, inventory: InventoryState, now_ns: int) -> list[QuoteIntent]:
        if book.source != "l2_depth" or book.book_status != "hot":
            return []
        if book.stale or book.sequence_gap:
            return []
        if book.best_bid is None or book.best_ask is None or book.best_bid <= 0 or book.best_ask <= 0:
            return []
        if book.best_bid >= book.best_ask:
            return []

        mid = (book.best_bid + book.best_ask) / 2.0
        half_spread = self.config.strategy.quote_spread_bps / 20_000.0
        buy_price = min(book.best_bid, mid * (1.0 - half_spread))
        sell_price = max(book.best_ask, mid * (1.0 + half_spread))
        max_inventory = self.config.risk.max_inventory_base
        skew_threshold = max_inventory * 0.8 if max_inventory > 0 else 0.0

        if inventory.base_qty >= skew_threshold > 0:
            sides = ["sell"]
        elif inventory.base_qty <= -skew_threshold < 0:
            sides = ["buy"]
        else:
            sides = ["buy", "sell"]

        quotes: list[QuoteIntent] = []
        for side in sides:
            price = buy_price if side == "buy" else sell_price
            qty = self.config.strategy.quote_size_usdt / price
            quotes.append(
                QuoteIntent(
                    symbol=self.config.symbol,
                    side=side,
                    price=price,
                    qty=qty,
                    created_at_ns=now_ns,
                    ttl_ms=self.config.strategy.order_ttl_ms,
                    post_only=True,
                    strategy_version=self.config.strategy.strategy_version,
                    strategy_tree_variant_id=self.config.strategy_tree.strategy_tree_variant_id,
                    strategy_tree_parent_id=self.config.strategy_tree.strategy_tree_parent_id,
                    strategy_tree_path=list(self.config.strategy_tree.strategy_tree_path),
                    venue=self.config.venue,
                    quote_id=f"{self.config.bot_id}-{now_ns}-{side}",
                )
            )
        return quotes


class OfiInventorySkewMakerStrategy:
    def __init__(self, config: MarketMakerConfig) -> None:
        self.config = config

    def generate_quotes(self, book: BookTick, inventory: InventoryState, now_ns: int) -> list[QuoteIntent]:
        if book.source != "l2_depth" or book.book_status != "hot":
            return []
        if book.stale or book.sequence_gap:
            return []
        if book.best_bid is None or book.best_ask is None or book.best_bid <= 0 or book.best_ask <= 0:
            return []
        if book.best_bid >= book.best_ask:
            return []
        spread_bps = book.spread_bps
        if spread_bps is None or spread_bps < self.config.strategy.min_quote_edge_bps:
            return []

        mid = (book.best_bid + book.best_ask) / 2.0
        half_spread = self.config.strategy.quote_spread_bps / 20_000.0
        buy_price = min(book.best_bid, mid * (1.0 - half_spread))
        sell_price = max(book.best_ask, mid * (1.0 + half_spread))
        sides = self._sides_for_inventory_and_ofi(book, inventory)
        quotes: list[QuoteIntent] = []
        for side in sides:
            price = buy_price if side == "buy" else sell_price
            qty = self.config.strategy.quote_size_usdt / price
            quotes.append(
                QuoteIntent(
                    symbol=self.config.symbol,
                    side=side,
                    price=price,
                    qty=qty,
                    created_at_ns=now_ns,
                    ttl_ms=self.config.strategy.order_ttl_ms,
                    post_only=True,
                    strategy_version=self.config.strategy.strategy_version,
                    strategy_tree_variant_id=self.config.strategy_tree.strategy_tree_variant_id,
                    strategy_tree_parent_id=self.config.strategy_tree.strategy_tree_parent_id,
                    strategy_tree_path=list(self.config.strategy_tree.strategy_tree_path),
                    venue=self.config.venue,
                    quote_id=f"{self.config.bot_id}-{now_ns}-{side}",
                )
            )
        return quotes

    def _sides_for_inventory_and_ofi(self, book: BookTick, inventory: InventoryState) -> list[str]:
        max_inventory = self.config.risk.max_inventory_base
        skew_threshold = max_inventory * 0.8 if max_inventory > 0 else 0.0
        if inventory.base_qty >= skew_threshold > 0:
            return ["sell"]
        if inventory.base_qty <= -skew_threshold < 0:
            return ["buy"]
        imbalance = _queue_imbalance(book)
        min_abs = abs(self.config.strategy.min_ofi_abs)
        if imbalance >= min_abs:
            return ["buy"]
        if imbalance <= -min_abs:
            return ["sell"]
        return ["buy", "sell"]


def reference_passive_maker(config: MarketMakerConfig) -> ReferencePassiveMakerStrategy:
    return ReferencePassiveMakerStrategy(config)


def _queue_imbalance(book: BookTick) -> float:
    bid_qty = float(book.best_bid_qty or 0.0)
    ask_qty = float(book.best_ask_qty or 0.0)
    total = bid_qty + ask_qty
    if total <= 0:
        return 0.0
    return (bid_qty - ask_qty) / total

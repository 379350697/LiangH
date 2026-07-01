from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class MarketMakerConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ExecutionConfig:
    allow_live_orders: bool = False
    primary_gateway: str = "paper"


@dataclass(frozen=True)
class SignalConfig:
    use_book_ticker: bool = True
    use_trade_flow: bool = True


@dataclass(frozen=True)
class LimitConfig:
    max_order_ops_per_minute: int = 1000
    max_order_ops_per_10s: int = 240


@dataclass(frozen=True)
class PaperConfig:
    initial_quote_usdt: float = 10_000.0
    maker_fee_bps: float = 2.0
    taker_fee_bps: float = 5.0
    queue_fill_ratio: float = 1.0


@dataclass(frozen=True)
class RiskConfig:
    max_inventory_base: float
    max_notional_usdt: float
    stale_feed_ms: int = 1500
    max_loop_lag_ms: int = 200
    max_spread_bps: float = 50.0


@dataclass(frozen=True)
class StrategyConfig:
    strategy_version: str
    variant_id: str
    quote_size_usdt: float
    quote_spread_bps: float
    order_ttl_ms: int = 1000
    quote_interval_ms: int = 250
    min_quote_edge_bps: float = 0.0
    min_ofi_abs: float = 0.20
    inventory_stop_bps: float = 20.0
    adverse_ofi_ticks: int = 3
    max_inventory_hold_ms: int = 30_000


@dataclass(frozen=True)
class StrategyTreeConfig:
    strategy_tree_variant_id: str
    strategy_tree_parent_id: str
    strategy_tree_path: list[str]


@dataclass(frozen=True)
class LedgerContext:
    run_id: str
    bot_id: str
    mode: str
    venue: str
    symbol: str
    strategy_version: str
    variant_id: str
    strategy_tree_variant_id: str
    strategy_tree_parent_id: str
    strategy_tree_path: list[str]


@dataclass(frozen=True)
class MarketMakerConfig:
    run_id: str
    bot_id: str
    mode: str
    venue: str
    symbol: str
    allowed_symbols: list[str]
    ledger_path: str
    execution: ExecutionConfig
    signals: SignalConfig
    limits: LimitConfig
    paper: PaperConfig
    risk: RiskConfig
    strategy: StrategyConfig
    strategy_tree: StrategyTreeConfig

    @property
    def ledger_context(self) -> LedgerContext:
        return LedgerContext(
            run_id=self.run_id,
            bot_id=self.bot_id,
            mode=self.mode,
            venue=self.venue,
            symbol=self.symbol,
            strategy_version=self.strategy.strategy_version,
            variant_id=self.strategy.variant_id,
            strategy_tree_variant_id=self.strategy_tree.strategy_tree_variant_id,
            strategy_tree_parent_id=self.strategy_tree.strategy_tree_parent_id,
            strategy_tree_path=list(self.strategy_tree.strategy_tree_path),
        )


def load_market_maker_config(path: str | Path) -> MarketMakerConfig:
    raw_path = Path(path)
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise MarketMakerConfigError("market maker config must be a JSON object")

    symbols = raw.get("symbols")
    if symbols is not None:
        if not isinstance(symbols, list) or len(symbols) != 1:
            raise MarketMakerConfigError("market maker config requires a single symbol")
        symbol = _normalize_symbol(symbols[0])
    else:
        symbol = _normalize_symbol(_required(raw, "symbol"))

    allowed_symbols = raw.get("allowed_symbols")
    if not isinstance(allowed_symbols, list) or not allowed_symbols:
        raise MarketMakerConfigError("allowed_symbols must be a non-empty list")
    allowed_symbols = [_normalize_symbol(value) for value in allowed_symbols]
    if symbol not in allowed_symbols:
        raise MarketMakerConfigError(f"symbol {symbol} must be present in allowed_symbols")

    mode = str(raw.get("mode", "paper")).lower()
    if mode not in {"paper", "live"}:
        raise MarketMakerConfigError("mode must be paper or live")

    venue = str(raw.get("venue", "binance_usdm")).lower()
    if venue != "binance_usdm":
        raise MarketMakerConfigError("market_maker_v1 currently supports venue=binance_usdm only")

    execution_raw = _mapping(raw.get("execution", {}), "execution")
    signals_raw = _mapping(raw.get("signals", {}), "signals")
    limits_raw = _mapping(raw.get("limits", {}), "limits")
    paper_raw = _mapping(raw.get("paper", {}), "paper")
    risk_raw = _mapping(_required(raw, "risk"), "risk")
    strategy_raw = _mapping(_required(raw, "strategy"), "strategy")
    tree_raw = _mapping(_required(raw, "strategy_tree"), "strategy_tree")

    tree_path = tree_raw.get("strategy_tree_path", [])
    if not isinstance(tree_path, list) or not tree_path:
        raise MarketMakerConfigError("strategy_tree.strategy_tree_path must be a non-empty list")

    primary_gateway = str(execution_raw.get("primary_gateway", "paper")).lower()
    if primary_gateway not in {"paper", "binance_ws_api"}:
        raise MarketMakerConfigError("execution.primary_gateway must be paper or binance_ws_api")

    return MarketMakerConfig(
        run_id=str(raw.get("run_id", "market-maker-v1")),
        bot_id=str(raw.get("bot_id", f"mm-{symbol.lower()}")),
        mode=mode,
        venue=venue,
        symbol=symbol,
        allowed_symbols=allowed_symbols,
        ledger_path=str(_required(raw, "ledger_path")),
        execution=ExecutionConfig(
            allow_live_orders=bool(execution_raw.get("allow_live_orders", False)),
            primary_gateway=primary_gateway,
        ),
        signals=SignalConfig(
            use_book_ticker=bool(signals_raw.get("use_book_ticker", True)),
            use_trade_flow=bool(signals_raw.get("use_trade_flow", True)),
        ),
        limits=LimitConfig(
            max_order_ops_per_minute=int(limits_raw.get("max_order_ops_per_minute", 1000)),
            max_order_ops_per_10s=int(limits_raw.get("max_order_ops_per_10s", 240)),
        ),
        paper=PaperConfig(
            initial_quote_usdt=float(paper_raw.get("initial_quote_usdt", 10_000.0)),
            maker_fee_bps=float(paper_raw.get("maker_fee_bps", 2.0)),
            taker_fee_bps=float(paper_raw.get("taker_fee_bps", 5.0)),
            queue_fill_ratio=float(paper_raw.get("queue_fill_ratio", 1.0)),
        ),
        risk=RiskConfig(
            max_inventory_base=float(_required(risk_raw, "max_inventory_base")),
            max_notional_usdt=float(_required(risk_raw, "max_notional_usdt")),
            stale_feed_ms=int(risk_raw.get("stale_feed_ms", 1500)),
            max_loop_lag_ms=int(risk_raw.get("max_loop_lag_ms", 200)),
            max_spread_bps=float(risk_raw.get("max_spread_bps", 50.0)),
        ),
        strategy=StrategyConfig(
            strategy_version=str(_required(strategy_raw, "strategy_version")),
            variant_id=str(_required(strategy_raw, "variant_id")),
            quote_size_usdt=float(_required(strategy_raw, "quote_size_usdt")),
            quote_spread_bps=float(_required(strategy_raw, "quote_spread_bps")),
            order_ttl_ms=int(strategy_raw.get("order_ttl_ms", 1000)),
            quote_interval_ms=int(strategy_raw.get("quote_interval_ms", 250)),
            min_quote_edge_bps=float(strategy_raw.get("min_quote_edge_bps", 0.0)),
            min_ofi_abs=float(strategy_raw.get("min_ofi_abs", 0.20)),
            inventory_stop_bps=float(strategy_raw.get("inventory_stop_bps", 20.0)),
            adverse_ofi_ticks=int(strategy_raw.get("adverse_ofi_ticks", 3)),
            max_inventory_hold_ms=int(strategy_raw.get("max_inventory_hold_ms", 30_000)),
        ),
        strategy_tree=StrategyTreeConfig(
            strategy_tree_variant_id=str(_required(tree_raw, "strategy_tree_variant_id")),
            strategy_tree_parent_id=str(_required(tree_raw, "strategy_tree_parent_id")),
            strategy_tree_path=[str(value) for value in tree_path],
        ),
    )


def _required(raw: dict[str, Any], key: str) -> Any:
    if key not in raw:
        raise MarketMakerConfigError(f"missing required field: {key}")
    return raw[key]


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MarketMakerConfigError(f"{name} must be a JSON object")
    return value


def _normalize_symbol(value: Any) -> str:
    symbol = str(value).strip().upper()
    if not symbol:
        raise MarketMakerConfigError("symbol must be non-empty")
    return symbol

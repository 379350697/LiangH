from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
import json
from pathlib import Path
from typing import Any, TypeVar


@dataclass(frozen=True)
class ExecutionConfig:
    mode: str = "paper"
    exchange: str = "okx"
    executor: str = "paper_okx"
    allow_live_orders: bool = False


@dataclass(frozen=True)
class RoutingConfig:
    shared_symbol_policy: str = "binance_first"


@dataclass(frozen=True)
class PaperConfig:
    initial_equity_usdt: float = 10_000.0
    fee_bps: float = 5.0
    slippage_bps: float = 10.0


@dataclass(frozen=True)
class RiskConfig:
    position_sizing_mode: str = "fixed_notional"
    active_capital_fraction: float = 0.30
    max_position_usdt: float = 1_000.0
    max_total_position_usdt: float | None = None
    max_open_positions: int | None = None
    max_daily_loss_usdt: float = 300.0
    default_leverage: int = 3
    alt_leverage: int = 5
    reference_leverage: int = 10
    max_slippage_bps: float = 10.0
    min_signal_strength: float = 0.35


@dataclass(frozen=True)
class MarketDataConfig:
    symbols: list[str] = field(default_factory=lambda: ["BTC-USDT-SWAP"])
    bars: list[str] = field(default_factory=lambda: ["1m", "5m", "15m", "1H", "1D"])
    candle_limit: int = 120
    max_fetch_workers: int = 1


@dataclass(frozen=True)
class UniverseConfig:
    mode: str = "static"
    provider: str = "okx"
    exclude_reference_symbols: bool = True
    reference_symbols: list[str] = field(default_factory=lambda: ["BTC-USDT-SWAP", "ETH-USDT-SWAP"])
    snapshot_path: str = "output/langlang_v1_2/universe_snapshot.json"
    liquidity_top_n: int = 200


@dataclass(frozen=True)
class SymbolSelectionConfig:
    enabled: bool = False
    top_n: int = 0
    min_score: float = 0.0
    min_daily_bars: int = 61
    style: str = "mixed"
    scoring_profile: str = "enhanced"
    long_top_n: int = 30
    short_top_n: int = 20
    min_long_score: float = 0.0
    min_short_score: float = 0.0


@dataclass(frozen=True)
class AppConfig:
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    paper: PaperConfig = field(default_factory=PaperConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    market_data: MarketDataConfig = field(default_factory=MarketDataConfig)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    selection: SymbolSelectionConfig = field(default_factory=SymbolSelectionConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    ledger_path: str = "runtime/langlang_trader.sqlite3"
    strategy_version: str = "rules_v01"


T = TypeVar("T")


def _from_dict(cls: type[T], values: dict[str, Any] | None) -> T:
    if values is None:
        return cls()  # type: ignore[call-arg]
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")
    known = {f.name: f for f in fields(cls)}
    kwargs: dict[str, Any] = {}
    for key, field_def in known.items():
        if key not in values:
            continue
        field_value = values[key]
        field_type = field_def.type
        if hasattr(field_type, "__dataclass_fields__"):
            kwargs[key] = _from_dict(field_type, field_value)
        else:
            kwargs[key] = field_value
    return cls(**kwargs)  # type: ignore[call-arg]


def load_config(path: str | Path) -> AppConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return AppConfig(
        execution=_from_dict(ExecutionConfig, raw.get("execution")),
        paper=_from_dict(PaperConfig, raw.get("paper")),
        risk=_from_dict(RiskConfig, raw.get("risk")),
        market_data=_from_dict(MarketDataConfig, raw.get("market_data")),
        universe=_from_dict(UniverseConfig, raw.get("universe")),
        selection=_from_dict(SymbolSelectionConfig, raw.get("selection")),
        routing=_from_dict(RoutingConfig, raw.get("routing")),
        ledger_path=raw.get("ledger_path", AppConfig().ledger_path),
        strategy_version=raw.get("strategy_version", AppConfig().strategy_version),
    )

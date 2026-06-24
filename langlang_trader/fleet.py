from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
import json
from pathlib import Path
from typing import Any

from langlang_trader.config import (
    ExecutionConfig,
    MarketDataConfig,
    PaperConfig,
    RiskConfig,
    RoutingConfig,
    SymbolSelectionConfig,
    UniverseConfig,
)
from langlang_trader.execution.paper import MultiExchangePaperExecutor, PaperExecutor
from langlang_trader.execution.routing import ExecutionRouter
from langlang_trader.features import DailyFeatureBuilder, FeatureSnapshot, MultiTimeframeFeatureBuilder
from langlang_trader.historical_patterns import HistoricalPatternMatcher, read_historical_patterns
from langlang_trader.ledger import Ledger
from langlang_trader.market_data import FallbackMarketData, MarketData
from langlang_trader.models import Position
from langlang_trader.risk import RiskEngine
from langlang_trader.strategy import (
    LangLangEnhancedVariant,
    LangLangNativeVariant,
    LangLangV1Variant,
    LangLangV1_1Variant,
    LangLangV1_3Variant,
    RulesLangLangEnhancedFinalStrategy,
    RulesLangLangEnhancedPayoffStrategy,
    RulesLangLangEnhancedStrategy,
    RulesLangLangNativeFinalStrategy,
    RulesLangLangNativePayoffStrategy,
    RulesLangLangNativeStrategy,
    RulesLangLangV1Strategy,
    RulesLangLangV1_1Strategy,
    RulesLangLangV1_2Strategy,
    RulesLangLangV1_3Strategy,
    RulesV01Strategy,
    StrategyVariant,
    strategy_from_version,
)
from langlang_trader.universe import (
    OkxBinanceUniverseProvider,
    OkxUniverseProvider,
    UniverseProvider,
    read_universe_snapshot,
    write_universe_snapshot,
)


V1_MULTI_TIMEFRAME_STRATEGIES = {
    RulesLangLangV1Strategy.version,
    RulesLangLangV1_1Strategy.version,
    RulesLangLangV1_2Strategy.version,
    RulesLangLangV1_3Strategy.version,
    RulesLangLangNativeStrategy.version,
    RulesLangLangEnhancedStrategy.version,
    RulesLangLangNativeFinalStrategy.version,
    RulesLangLangEnhancedFinalStrategy.version,
    RulesLangLangNativePayoffStrategy.version,
    RulesLangLangEnhancedPayoffStrategy.version,
}

HISTORICAL_MATCH_STRATEGIES = {
    RulesLangLangV1_1Strategy.version,
    RulesLangLangV1_2Strategy.version,
    RulesLangLangV1_3Strategy.version,
    RulesLangLangEnhancedStrategy.version,
    RulesLangLangEnhancedFinalStrategy.version,
    RulesLangLangEnhancedPayoffStrategy.version,
}


@dataclass(frozen=True)
class BotConfig:
    bot_id: str
    variant: StrategyVariant | LangLangV1Variant | LangLangV1_1Variant | LangLangNativeVariant | LangLangEnhancedVariant
    strategy_version: str | None = None
    selection_profile: str | None = None

    def to_dict(self) -> dict[str, Any]:
        row = {"bot_id": self.bot_id, "variant": self.variant.to_dict()}
        if self.strategy_version:
            row["strategy_version"] = self.strategy_version
        if self.selection_profile:
            row["selection_profile"] = self.selection_profile
        return row


@dataclass(frozen=True)
class FleetConfig:
    run_id: str
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    paper: PaperConfig = field(default_factory=PaperConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    market_data: MarketDataConfig = field(default_factory=MarketDataConfig)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    selection: SymbolSelectionConfig = field(default_factory=SymbolSelectionConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    ledger_path: str = "runtime/langlang_fleet.sqlite3"
    strategy_version: str = "rules_v01"
    historical_patterns_path: str = "output/langlang_distill/historical_patterns.csv"
    bots: list[BotConfig] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "execution": asdict(self.execution),
            "paper": asdict(self.paper),
            "risk": asdict(self.risk),
            "market_data": asdict(self.market_data),
            "universe": asdict(self.universe),
            "selection": asdict(self.selection),
            "routing": asdict(self.routing),
            "ledger_path": self.ledger_path,
            "strategy_version": self.strategy_version,
            "historical_patterns_path": self.historical_patterns_path,
            "bots": [bot.to_dict() for bot in self.bots],
        }


def load_fleet_config(path: str | Path) -> FleetConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return fleet_config_from_dict(raw)


def fleet_config_from_dict(raw: dict[str, Any]) -> FleetConfig:
    return FleetConfig(
        run_id=raw["run_id"],
        execution=ExecutionConfig(**raw.get("execution", {})),
        paper=PaperConfig(**raw.get("paper", {})),
        risk=RiskConfig(**raw.get("risk", {})),
        market_data=MarketDataConfig(**raw.get("market_data", {})),
        universe=UniverseConfig(**raw.get("universe", {})),
        selection=SymbolSelectionConfig(**raw.get("selection", {})),
        routing=RoutingConfig(**raw.get("routing", {})),
        ledger_path=raw.get("ledger_path", FleetConfig(run_id=raw["run_id"]).ledger_path),
        strategy_version=raw.get("strategy_version", "rules_v01"),
        historical_patterns_path=raw.get(
            "historical_patterns_path",
            FleetConfig(run_id=raw["run_id"]).historical_patterns_path,
        ),
        bots=[_bot_config_from_dict(row, raw.get("strategy_version", "rules_v01")) for row in raw.get("bots", [])],
    )


class FleetRunner:
    def __init__(
        self,
        *,
        config: FleetConfig,
        market_data: MarketData,
        ledger: Ledger,
        universe_provider: UniverseProvider | None = None,
    ):
        if config.execution.mode != "paper" or config.execution.executor not in {"paper_okx", "paper_multi"}:
            raise PermissionError("FleetRunner supports paper_okx or paper_multi execution only")
        if config.execution.executor == "paper_multi" and config.universe.mode not in {"okx_all_usdt_swap", "okx_binance_usdt_swap_observe"}:
            raise PermissionError("paper_multi requires an exchange-aware universe snapshot")
        self.config = config
        self.market_data = market_data
        self.ledger = ledger
        self.fleet_ledger = ledger.scoped(
            run_id=config.run_id,
            bot_id="fleet",
            variant_id="fleet",
            exchange=_fleet_event_exchange(config),
        )
        self.universe_provider = universe_provider
        self.feature_builder = DailyFeatureBuilder()
        self.multi_feature_builder = MultiTimeframeFeatureBuilder()
        self.strategy_versions = _fleet_strategy_versions(config)
        self.uses_multi_timeframe = any(version in V1_MULTI_TIMEFRAME_STRATEGIES for version in self.strategy_versions)
        patterns = (
            read_historical_patterns(config.historical_patterns_path)
            if any(version in HISTORICAL_MATCH_STRATEGIES for version in self.strategy_versions)
            else []
        )
        self.pattern_matcher = HistoricalPatternMatcher(patterns) if patterns else None

    def run_once(self) -> dict[str, int]:
        symbols, universe_snapshot = self._runtime_symbols()
        execution_router = (
            ExecutionRouter(
                universe_snapshot,
                shared_symbol_policy=self.config.routing.shared_symbol_policy,
            )
            if universe_snapshot is not None and self.config.execution.executor == "paper_multi"
            else None
        )
        okx_executable_symbols = (
            set(universe_snapshot.reference_symbols) | set(universe_snapshot.symbols)
            if universe_snapshot is not None
            else set(symbols)
        )
        routable_symbols = (
            _routable_symbols_for_executor(universe_snapshot, self.config.execution.executor)
            if universe_snapshot is not None
            else set(symbols)
        )
        market_data_by_symbol = _market_data_by_symbol(self.market_data, universe_snapshot)
        cycle = {
            "bots": len(self.config.bots),
            "symbols": len(symbols),
            "signals": 0,
            "intents": 0,
            "orders": 0,
            "fills": 0,
            "stop_exits": 0,
            "risk_rejections": 0,
            "selected_symbols": 0,
            "selection_skips": 0,
            "market_data_errors": 0,
            "errors": 0,
        }
        candles_by_symbol: dict[str, dict[str, Any]] = {}
        latest_prices: dict[str, float] = {}
        startup_bars = ["1D"] if self._should_stage_multi_timeframe_fetch() else None
        max_workers = max(1, int(getattr(self.config.market_data, "max_fetch_workers", 1) or 1))
        if max_workers > 1 and len(symbols) > 1:
            with ThreadPoolExecutor(max_workers=min(max_workers, len(symbols))) as pool:
                futures = {
                    pool.submit(
                        self._fetch_symbol_market_data,
                        symbol,
                        market_data_by_symbol.get(symbol, self.market_data),
                        startup_bars,
                        startup_bars is None,
                    ): symbol
                    for symbol in symbols
                }
                for future in as_completed(futures):
                    symbol = futures[future]
                    try:
                        symbol_candles, latest_price = future.result()
                        candles_by_symbol[symbol] = symbol_candles
                        latest_prices[symbol] = latest_price
                    except Exception as exc:
                        self.fleet_ledger.record_risk_event("market_data_error", {"error": repr(exc)}, symbol=symbol)
                        cycle["market_data_errors"] += 1
        else:
            for symbol in symbols:
                try:
                    symbol_candles, latest_price = self._fetch_symbol_market_data(
                        symbol,
                        market_data_by_symbol.get(symbol, self.market_data),
                        startup_bars,
                        startup_bars is None,
                    )
                    candles_by_symbol[symbol] = symbol_candles
                    latest_prices[symbol] = latest_price
                except Exception as exc:
                    self.fleet_ledger.record_risk_event("market_data_error", {"error": repr(exc)}, symbol=symbol)
                    cycle["market_data_errors"] += 1

        snapshots_by_symbol = self._build_snapshots(candles_by_symbol)
        selected_symbols = set(snapshots_by_symbol)
        long_selected_symbols: set[str] = set()
        short_selected_symbols: set[str] = set()
        selection_results = {}
        selection_results_by_side: dict[str, dict[str, Any]] = {"long": {}, "short": {}}
        selection_states_by_profile: dict[str, dict[str, Any]] = {}
        reference_symbols = set(universe_snapshot.reference_symbols if universe_snapshot is not None else self.config.universe.reference_symbols)
        if self.config.selection.enabled:
            if self.config.selection.style == "dual_board":
                selection_profiles = {
                    _bot_selection_profile(self.config, bot)
                    for bot in self.config.bots
                } or {self.config.selection.scoring_profile}
                selection_states_by_profile = _selection_states_for_profiles(
                    self.config.selection,
                    snapshots_by_symbol,
                    reference_symbols=reference_symbols,
                    profiles=selection_profiles,
                )
                selected_symbols = set()
                long_selected_symbols = set()
                short_selected_symbols = set()
                profiles_payload: dict[str, Any] = {}
                for profile, state in selection_states_by_profile.items():
                    selected_symbols.update(state["selected_symbols"])
                    long_selected_symbols.update(state["long_selected_symbols"])
                    short_selected_symbols.update(state["short_selected_symbols"])
                    profiles_payload[profile] = {
                        "selected_symbols": sorted(state["selected_symbols"]),
                        "executable_selected_symbols": sorted(state["selected_symbols"] & routable_symbols),
                        "routable_selected_symbols": sorted(state["selected_symbols"] & routable_symbols),
                        "okx_executable_selected_symbols": sorted(state["selected_symbols"] & okx_executable_symbols),
                        "long_selected_symbols": sorted(state["long_selected_symbols"]),
                        "short_selected_symbols": sorted(state["short_selected_symbols"]),
                        "ranked_long": [result.to_dict() for result in state["long_results"][:30]],
                        "ranked_short": [result.to_dict() for result in state["short_results"][:20]],
                    }
                default_state = selection_states_by_profile.get(
                    self.config.selection.scoring_profile,
                    next(iter(selection_states_by_profile.values()), None),
                )
                if default_state is not None:
                    selection_results_by_side = default_state["selection_results_by_side"]
                    selection_results = default_state["selection_results"]
                selectable = set(snapshots_by_symbol) - reference_symbols
                cycle["selection_skips"] = len(selectable - selected_symbols)
                self.fleet_ledger.record_risk_event(
                    "symbol_selection",
                    {
                        "style": self.config.selection.style,
                        "profiles": profiles_payload,
                        "long_top_n": self.config.selection.long_top_n,
                        "short_top_n": self.config.selection.short_top_n,
                        "selected_symbols": sorted(selected_symbols),
                        "executable_selected_symbols": sorted(selected_symbols & routable_symbols),
                        "routable_selected_symbols": sorted(selected_symbols & routable_symbols),
                        "okx_executable_selected_symbols": sorted(selected_symbols & okx_executable_symbols),
                        "long_selected_symbols": sorted(long_selected_symbols),
                        "short_selected_symbols": sorted(short_selected_symbols),
                    },
                )
            else:
                from langlang_trader.symbol_selection import SymbolSelector

                ranked = SymbolSelector(self.config.selection).rank(snapshots_by_symbol)
                selection_results = {result.symbol: result for result in ranked}
                selected_symbols = {result.symbol for result in ranked if result.selected}
                cycle["selection_skips"] = len(set(snapshots_by_symbol) - selected_symbols)
                self.fleet_ledger.record_risk_event(
                    "symbol_selection",
                    {
                        "top_n": self.config.selection.top_n,
                        "min_score": self.config.selection.min_score,
                        "selected_symbols": sorted(selected_symbols),
                        "ranked": [result.to_dict() for result in ranked[:25]],
                    },
                )
        cycle["selected_symbols"] = len(selected_symbols)
        if startup_bars is not None and selected_symbols:
            self._enrich_selected_multi_timeframe_data(
                symbols=sorted(selected_symbols & set(candles_by_symbol)),
                candles_by_symbol=candles_by_symbol,
                latest_prices=latest_prices,
                market_data_by_symbol=market_data_by_symbol,
                cycle=cycle,
            )
            snapshots_by_symbol = self._build_snapshots(candles_by_symbol)

        for bot in self.config.bots:
            allowed_side = _bot_allowed_side(bot.variant)
            bot_strategy_version = _bot_strategy_version(self.config, bot)
            bot_ledger = self.ledger.scoped(
                run_id=self.config.run_id,
                bot_id=bot.bot_id,
                variant_id=bot.variant.variant_id,
            )
            price_provider = lambda symbol, prices=latest_prices: prices[symbol]
            if self.config.execution.executor == "paper_multi" and execution_router is not None:
                executor = MultiExchangePaperExecutor(
                    ledger=bot_ledger,
                    paper_config=self.config.paper,
                    price_provider=price_provider,
                    router=execution_router,
                )
            else:
                executor = PaperExecutor(
                    ledger=bot_ledger,
                    paper_config=self.config.paper,
                    price_provider=price_provider,
                )
            _record_bot_account_snapshot(bot_ledger, executor, bot_strategy_version)
            strategy = strategy_from_version(bot_strategy_version, bot.variant)
            risk_engine = RiskEngine(self.config.risk)
            bot_selection_state = selection_states_by_profile.get(_bot_selection_profile(self.config, bot))
            for symbol in candles_by_symbol:
                try:
                    if (
                        universe_snapshot is not None
                        and self.config.execution.executor == "paper_okx"
                        and symbol not in okx_executable_symbols
                    ):
                        continue
                    if self._close_if_stop_loss_hit(
                        ledger=bot_ledger,
                        executor=executor,
                        symbol=symbol,
                        latest_price=latest_prices[symbol],
                        cycle=cycle,
                    ):
                        continue
                    snapshot = snapshots_by_symbol.get(symbol)
                    if snapshot is None:
                        continue
                    if symbol in reference_symbols and self.config.selection.style == "dual_board":
                        continue
                    selection_result = selection_results.get(symbol)
                    if self.config.selection.enabled and self.config.selection.style == "dual_board":
                        bot_long_selected = (
                            bot_selection_state["long_selected_symbols"]
                            if bot_selection_state is not None
                            else long_selected_symbols
                        )
                        bot_short_selected = (
                            bot_selection_state["short_selected_symbols"]
                            if bot_selection_state is not None
                            else short_selected_symbols
                        )
                        allowed_symbols = _selected_symbols_for_side(
                            allowed_side=allowed_side,
                            long_selected=bot_long_selected,
                            short_selected=bot_short_selected,
                        )
                        if symbol not in allowed_symbols:
                            continue
                        selection_result = _selection_result_for_side(
                            symbol=symbol,
                            allowed_side=allowed_side,
                            by_side=(
                                bot_selection_state["selection_results_by_side"]
                                if bot_selection_state is not None
                                else selection_results_by_side
                            ),
                        )
                    elif self.config.selection.enabled and symbol not in selected_symbols:
                        continue
                    snapshot = _with_selection_features(snapshot, selection_result)
                    if bot_strategy_version in HISTORICAL_MATCH_STRATEGIES and self.pattern_matcher is not None:
                        snapshot = _with_historical_match(snapshot, self.pattern_matcher)
                    signal = strategy.generate_from_features(snapshot)
                    if signal is None:
                        continue
                    signal_id = bot_ledger.record_signal(signal, bot_strategy_version)
                    cycle["signals"] += 1
                    intent = risk_engine.intent_from_signal(
                        signal=signal,
                        account=executor.get_account(),
                        latest_price=latest_prices[symbol],
                        existing_position=_position_for_symbol(executor.get_positions(), symbol),
                    )
                    if intent is None:
                        bot_ledger.record_risk_event(
                            "intent_rejected",
                            {"signal_id": signal_id, "strength": signal.strength},
                            symbol=symbol,
                        )
                        cycle["risk_rejections"] += 1
                        continue
                    routed_intent = execution_router.route(intent) if execution_router is not None else None
                    if execution_router is not None and routed_intent is None:
                        bot_ledger.record_risk_event(
                            "execution_route_rejected",
                            {"signal_id": signal_id, "reason": execution_router.rejection_reason(intent)},
                            symbol=symbol,
                        )
                        cycle["risk_rejections"] += 1
                        continue
                    intent_ledger = (
                        bot_ledger.scoped(
                            run_id=self.config.run_id,
                            bot_id=bot.bot_id,
                            variant_id=bot.variant.variant_id,
                            exchange=routed_intent.exchange,
                        )
                        if routed_intent is not None
                        else bot_ledger
                    )
                    intent_ledger.record_order_intent(intent, signal_id=signal_id)
                    cycle["intents"] += 1
                    result = executor.place_order(intent, route=routed_intent)
                    if result.status in {"filled", "accepted", "submitted"}:
                        cycle["orders"] += 1
                    if result.filled_qty > 0:
                        cycle["fills"] += 1
                except Exception as exc:  # pragma: no cover - operational safety path
                    bot_ledger.record_risk_event("fleet_runner_error", {"error": repr(exc)}, symbol=symbol)
                    cycle["errors"] += 1
        return cycle

    def _runtime_symbols(self) -> tuple[list[str], Any | None]:
        if self.config.universe.mode in {"okx_all_usdt_swap", "okx_binance_usdt_swap_observe"}:
            provider = self.universe_provider or _universe_provider_for_config(self.config.universe)
            try:
                snapshot = provider.list_symbols()
            except Exception as exc:
                if not self.config.universe.snapshot_path:
                    raise
                snapshot_path = Path(self.config.universe.snapshot_path)
                if not snapshot_path.exists():
                    raise
                snapshot = read_universe_snapshot(snapshot_path)
                self.fleet_ledger.record_risk_event(
                    "universe_snapshot_fallback",
                    {
                        "error": repr(exc),
                        "snapshot_path": self.config.universe.snapshot_path,
                        "snapshot_generated_at": snapshot.generated_at,
                    },
                )
            if self.config.universe.snapshot_path:
                write_universe_snapshot(self.config.universe.snapshot_path, snapshot)
            universe_symbols = (
                snapshot.observed_symbols
                if self.config.universe.mode == "okx_binance_usdt_swap_observe" and snapshot.observed_symbols
                else snapshot.symbols
            )
            symbols = list(dict.fromkeys([*snapshot.reference_symbols, *universe_symbols]))
            self.fleet_ledger.record_risk_event(
                "universe_snapshot",
                {
                    "mode": snapshot.mode,
                    "symbols": len(snapshot.symbols),
                    "observed_symbols": len(getattr(snapshot, "observed_symbols", [])),
                    "reference_symbols": snapshot.reference_symbols,
                    "snapshot_path": self.config.universe.snapshot_path,
                    "summary": snapshot.raw_payload.get("summary", {}),
                },
            )
            return symbols, snapshot
        return list(self.config.market_data.symbols), None

    def _fetch_symbol_market_data(
        self,
        symbol: str,
        market_data: MarketData | None = None,
        bars: list[str] | None = None,
        fetch_latest_ticker: bool = True,
    ) -> tuple[dict[str, Any], float]:
        market_data = market_data or self.market_data
        bars_to_fetch = bars or self.config.market_data.bars
        if self.uses_multi_timeframe:
            candles_by_bar = {
                bar: market_data.get_candles(
                    symbol,
                    bar=bar,
                    limit=self.config.market_data.candle_limit,
                )
                for bar in bars_to_fetch
            }
        else:
            candles_by_bar = {
                "1D": market_data.get_candles(
                    symbol,
                    bar="1D",
                    limit=self.config.market_data.candle_limit,
                )
            }
        if fetch_latest_ticker:
            try:
                latest_price = market_data.latest_price(symbol)
            except Exception as exc:
                latest_price = _latest_close_from_candles(candles_by_bar)
                self.fleet_ledger.record_risk_event(
                    "latest_price_fallback_to_candle_close",
                    {"error": repr(exc), "latest_price": latest_price},
                    symbol=symbol,
                )
        else:
            latest_price = _latest_close_from_candles(candles_by_bar)
        return candles_by_bar, latest_price

    def _should_stage_multi_timeframe_fetch(self) -> bool:
        return self.uses_multi_timeframe and self.config.selection.enabled and self.config.selection.style == "dual_board"

    def _enrich_selected_multi_timeframe_data(
        self,
        *,
        symbols: list[str],
        candles_by_symbol: dict[str, dict[str, Any]],
        latest_prices: dict[str, float],
        market_data_by_symbol: dict[str, MarketData],
        cycle: dict[str, int],
    ) -> None:
        intraday_bars = [bar for bar in self.config.market_data.bars if bar != "1D"]
        if not intraday_bars or not symbols:
            return
        max_workers = max(1, int(getattr(self.config.market_data, "max_fetch_workers", 1) or 1))

        def fetch_intraday(symbol: str) -> tuple[str, dict[str, Any]]:
            market_data = market_data_by_symbol.get(symbol, self.market_data)
            rows = {
                bar: market_data.get_candles(
                    symbol,
                    bar=bar,
                    limit=self.config.market_data.candle_limit,
                )
                for bar in intraday_bars
            }
            return symbol, rows

        if max_workers > 1 and len(symbols) > 1:
            with ThreadPoolExecutor(max_workers=min(max_workers, len(symbols))) as pool:
                futures = {pool.submit(fetch_intraday, symbol): symbol for symbol in symbols}
                for future in as_completed(futures):
                    symbol = futures[future]
                    try:
                        _, rows = future.result()
                        candles_by_symbol.setdefault(symbol, {}).update(rows)
                        latest_prices[symbol] = _latest_close_from_candles(rows)
                    except Exception as exc:
                        self.fleet_ledger.record_risk_event(
                            "market_data_error",
                            {"error": repr(exc), "phase": "selected_intraday_enrichment"},
                            symbol=symbol,
                        )
                        cycle["market_data_errors"] += 1
            return
        for symbol in symbols:
            try:
                _, rows = fetch_intraday(symbol)
                candles_by_symbol.setdefault(symbol, {}).update(rows)
                latest_prices[symbol] = _latest_close_from_candles(rows)
            except Exception as exc:
                self.fleet_ledger.record_risk_event(
                    "market_data_error",
                    {"error": repr(exc), "phase": "selected_intraday_enrichment"},
                    symbol=symbol,
                )
                cycle["market_data_errors"] += 1

    def _close_if_stop_loss_hit(
        self,
        *,
        ledger: Ledger,
        executor: PaperExecutor,
        symbol: str,
        latest_price: float,
        cycle: dict[str, int],
    ) -> bool:
        position = _position_for_symbol(executor.get_positions(), symbol)
        if position is None:
            return False
        stop_loss = ledger.latest_stop_loss(symbol, exchange=position.exchange)
        if stop_loss is None:
            return False
        triggered = (
            latest_price <= stop_loss
            if position.side.value == "long"
            else latest_price >= stop_loss
        )
        if not triggered:
            return False
        result = executor.close_position(symbol, reason=f"stop_loss:{stop_loss}")
        event_ledger = ledger.scoped(
            run_id=ledger.run_id,
            bot_id=ledger.bot_id,
            variant_id=ledger.variant_id,
            exchange=position.exchange,
        )
        event_ledger.record_risk_event(
            "stop_loss_exit",
            {
                "latest_price": latest_price,
                "stop_loss": stop_loss,
                "status": result.status,
                "exchange_order_id": result.exchange_order_id,
            },
            symbol=symbol,
        )
        if result.status in {"filled", "accepted", "submitted"}:
            cycle["orders"] += 1
        if result.filled_qty > 0:
            cycle["fills"] += 1
        cycle["stop_exits"] += 1
        return True

    def _build_snapshots(self, candles_by_symbol: dict[str, dict[str, Any]]) -> dict[str, FeatureSnapshot]:
        snapshots: dict[str, FeatureSnapshot] = {}
        for symbol, candles_by_bar in candles_by_symbol.items():
            if self.uses_multi_timeframe:
                snapshot = self.multi_feature_builder.build(symbol, candles_by_bar)
            else:
                snapshot = self.feature_builder.build(symbol, candles_by_bar["1D"])
            if snapshot is not None:
                snapshots[symbol] = snapshot
        return snapshots


def _position_for_symbol(positions: list[Position], symbol: str) -> Position | None:
    for position in positions:
        if position.symbol == symbol:
            return position
    return None


def _latest_close_from_candles(candles_by_bar: dict[str, list[Any]]) -> float:
    latest = None
    for rows in candles_by_bar.values():
        for candle in rows:
            if latest is None or candle.ts > latest.ts:
                latest = candle
    if latest is None:
        raise ValueError("no candles available for latest price fallback")
    return float(latest.close)


def _bot_strategy_version(config: FleetConfig, bot: BotConfig) -> str:
    return bot.strategy_version or config.strategy_version


def _fleet_event_exchange(config: FleetConfig) -> str:
    if config.execution.executor == "paper_multi" or config.execution.exchange == "multi":
        return "multi"
    return config.execution.exchange


def _bot_selection_profile(config: FleetConfig, bot: BotConfig) -> str:
    return bot.selection_profile or config.selection.scoring_profile


def _fleet_strategy_versions(config: FleetConfig) -> set[str]:
    versions = {config.strategy_version}
    versions.update(bot.strategy_version for bot in config.bots if bot.strategy_version)
    return versions


def _universe_provider_for_config(config: UniverseConfig) -> UniverseProvider:
    if config.mode == "okx_binance_usdt_swap_observe" or config.provider == "okx_binance":
        return OkxBinanceUniverseProvider(config=config)
    return OkxUniverseProvider(config=config)


def _with_selection_features(snapshot: FeatureSnapshot, selection_result: Any | None) -> FeatureSnapshot:
    if selection_result is None:
        return snapshot
    features = dict(snapshot.features)
    features.update(selection_result.features)
    features["selection_reason_codes"] = selection_result.reason_codes
    features["selection_filter_codes"] = getattr(selection_result, "filter_codes", [])
    features["selection_mode"] = getattr(selection_result, "selection_mode", "mixed")
    features["selection_market_env"] = getattr(selection_result, "market_env", {})
    features["selection_selected"] = selection_result.selected
    return FeatureSnapshot(
        symbol=snapshot.symbol,
        bar=snapshot.bar,
        last_ts=snapshot.last_ts,
        features=features,
        created_at=snapshot.created_at,
    )


def _best_selection_result(left: Any | None, right: Any | None) -> Any | None:
    if left is None:
        return right
    if right is None:
        return left
    return left if left.selection_score >= right.selection_score else right


def _bot_allowed_side(variant: Any) -> str:
    allowed = str(getattr(variant, "allowed_side", "both") or "both").lower()
    variant_id = str(getattr(variant, "variant_id", "")).lower()
    if allowed in {"long", "short"}:
        return allowed
    if variant_id.startswith(("llv1_1_long_", "llv1_2_long_")):
        return "long"
    if variant_id.startswith(("llv1_1_short_", "llv1_2_short_")):
        return "short"
    return "both"


def _selected_symbols_for_side(
    *,
    allowed_side: str,
    long_selected: set[str],
    short_selected: set[str],
) -> set[str]:
    if allowed_side == "long":
        return set(long_selected)
    if allowed_side == "short":
        return set(short_selected)
    return set(long_selected) | set(short_selected)


def _selection_result_for_side(
    *,
    symbol: str,
    allowed_side: str,
    by_side: dict[str, dict[str, Any]],
) -> Any | None:
    if allowed_side == "long":
        return by_side.get("long", {}).get(symbol)
    if allowed_side == "short":
        return by_side.get("short", {}).get(symbol)
    return _best_selection_result(
        by_side.get("long", {}).get(symbol),
        by_side.get("short", {}).get(symbol),
    )


def _selection_states_for_profiles(
    config: SymbolSelectionConfig,
    snapshots_by_symbol: dict[str, FeatureSnapshot],
    *,
    reference_symbols: set[str],
    profiles: set[str],
) -> dict[str, dict[str, Any]]:
    from langlang_trader.symbol_selection import SelectionEngine

    states: dict[str, dict[str, Any]] = {}
    for profile in sorted(profiles):
        profile_config = replace(config, scoring_profile=profile)
        boards = SelectionEngine(profile_config).rank_all_market(
            snapshots_by_symbol,
            reference_symbols=list(reference_symbols),
        )
        long_results = boards["long_main_wave"]
        short_results = boards["short_waterfall"]
        selection_results_by_side = {
            "long": {result.symbol: result for result in long_results},
            "short": {result.symbol: result for result in short_results},
        }
        long_selected_symbols = {result.symbol for result in long_results if result.selected}
        short_selected_symbols = {result.symbol for result in short_results if result.selected}
        selected_symbols = long_selected_symbols | short_selected_symbols
        states[profile] = {
            "long_results": long_results,
            "short_results": short_results,
            "selection_results_by_side": selection_results_by_side,
            "long_selected_symbols": long_selected_symbols,
            "short_selected_symbols": short_selected_symbols,
            "selected_symbols": selected_symbols,
            "selection_results": {
                symbol: _best_selection_result(
                    selection_results_by_side["long"].get(symbol),
                    selection_results_by_side["short"].get(symbol),
                )
                for symbol in selected_symbols
            },
        }
    return states


def _with_historical_match(snapshot: FeatureSnapshot, matcher: HistoricalPatternMatcher) -> FeatureSnapshot:
    features = dict(snapshot.features)
    if features.get("matched_trade_examples"):
        return snapshot
    ret_20d = _float(features.get("ret_20d"))
    if ret_20d >= 0:
        side = "long"
        regime = "first_divergence"
        setup = "small_divergence_entry"
    else:
        side = "short"
        regime = "weak_waterfall"
        setup = "waterfall_continuation"
    match = matcher.match(side=side, regime=regime, setup=setup, features=features)
    features["historical_match_score"] = match.score
    features["matched_trade_examples"] = match.examples
    features["big_loss_overlap_count"] = match.big_loss_overlap_count
    return FeatureSnapshot(
        symbol=snapshot.symbol,
        bar=snapshot.bar,
        last_ts=snapshot.last_ts,
        features=features,
        created_at=snapshot.created_at,
    )


def _market_data_by_symbol(market_data: MarketData, universe_snapshot: Any | None) -> dict[str, MarketData]:
    if universe_snapshot is None or not isinstance(market_data, FallbackMarketData):
        return {}
    routed: dict[str, MarketData] = {}
    for row in universe_snapshot.rows:
        if row.source_exchange == "okx" and (row.tradable or row.is_reference):
            routed[row.symbol] = market_data.primary
    for row in universe_snapshot.rows:
        if row.source_exchange == "binance" and row.symbol not in routed:
            routed[row.symbol] = market_data.fallback
    return routed


def _routable_symbols_for_executor(universe_snapshot: Any, executor: str) -> set[str]:
    reference_symbols = set(getattr(universe_snapshot, "reference_symbols", []) or [])
    if executor == "paper_multi":
        routable = set(reference_symbols)
        for row in getattr(universe_snapshot, "rows", []) or []:
            if getattr(row, "is_reference", False):
                continue
            if getattr(row, "tradable", False):
                routable.add(row.symbol)
                continue
            if getattr(row, "source_exchange", "") == "binance" and getattr(row, "filter_reason", "") in {
                "okx_executable_overlap",
                "binance_observed_only_not_okx_executable",
            }:
                routable.add(row.symbol)
                continue
            if getattr(row, "execution_symbol", ""):
                routable.add(row.symbol)
        return routable
    return reference_symbols | set(getattr(universe_snapshot, "symbols", []) or [])


def _record_bot_account_snapshot(
    ledger: Ledger,
    executor: PaperExecutor | MultiExchangePaperExecutor,
    strategy_version: str,
) -> None:
    snapshot_exchange = "multi" if isinstance(executor, MultiExchangePaperExecutor) else executor.exchange
    snapshot_ledger = ledger.scoped(
        run_id=ledger.run_id,
        bot_id=ledger.bot_id,
        variant_id=ledger.variant_id,
        exchange=snapshot_exchange,
    )
    try:
        snapshot = executor.get_account()
    except Exception as exc:  # pragma: no cover - operational diagnostic path
        snapshot_ledger.record_risk_event(
            "bot_account_snapshot_failed",
            {"error": repr(exc), "strategy_version": strategy_version},
        )
        return
    snapshot_ledger.record_equity_snapshot(
        snapshot,
        raw={"source": "fleet_bot_tick", "strategy_version": strategy_version},
        strategy_version=strategy_version,
    )


def _bot_config_from_dict(row: dict[str, Any], strategy_version: str) -> BotConfig:
    bot_strategy_version = row.get("strategy_version") or strategy_version
    if bot_strategy_version in {
        RulesLangLangEnhancedStrategy.version,
        RulesLangLangEnhancedFinalStrategy.version,
        RulesLangLangEnhancedPayoffStrategy.version,
    }:
        variant = LangLangEnhancedVariant(**row["variant"])
    elif bot_strategy_version in {
        RulesLangLangNativeStrategy.version,
        RulesLangLangNativeFinalStrategy.version,
        RulesLangLangNativePayoffStrategy.version,
    }:
        variant = LangLangNativeVariant(**row["variant"])
    elif bot_strategy_version == RulesLangLangV1_3Strategy.version:
        variant = LangLangV1_3Variant(**row["variant"])
    elif bot_strategy_version in {RulesLangLangV1_1Strategy.version, RulesLangLangV1_2Strategy.version}:
        variant = LangLangV1_1Variant(**row["variant"])
    elif bot_strategy_version == RulesLangLangV1Strategy.version:
        variant = LangLangV1Variant(**row["variant"])
    else:
        variant = StrategyVariant(**row["variant"])
    return BotConfig(
        bot_id=row["bot_id"],
        variant=variant,
        strategy_version=row.get("strategy_version"),
        selection_profile=row.get("selection_profile"),
    )


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value in {None, ""}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default

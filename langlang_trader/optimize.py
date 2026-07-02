from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
from typing import Any

from langlang_trader.config import ExecutionConfig, MarketDataConfig, PaperConfig, RiskConfig, SymbolSelectionConfig, UniverseConfig
from langlang_trader.distill_v1 import TradeLabel, TradeLabeler
from langlang_trader.execution.paper import PaperExecutor
from langlang_trader.features import DailyFeatureBuilder, FeatureSnapshot, MultiTimeframeFeatureBuilder
from langlang_trader.fleet import BotConfig, FleetConfig
from langlang_trader.historical_patterns import HistoricalPatternMatcher, build_historical_patterns, write_historical_patterns
from langlang_trader.ledger import Ledger
from langlang_trader.models import Candle, ExitPlan, OrderIntent, Side, StrategyAction
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
    default_langlang_enhanced_grid,
    default_langlang_v1_grid,
    default_langlang_v1_1_grid,
    default_langlang_v1_3_grid,
    default_langlang_native_grid,
    default_variant_grid,
    strategy_from_version,
)
from langlang_trader.strategy_library import DEFAULT_DB_PATH, DEFAULT_REGISTRY_PATH, ingest_leaderboard


EVENT_REPLAY_STRATEGIES = {
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

MULTI_EXCHANGE_SELECTION_STRATEGIES = {
    RulesLangLangV1_2Strategy.version,
    RulesLangLangV1_3Strategy.version,
    RulesLangLangNativeStrategy.version,
    RulesLangLangEnhancedStrategy.version,
    RulesLangLangNativeFinalStrategy.version,
    RulesLangLangEnhancedFinalStrategy.version,
    RulesLangLangNativePayoffStrategy.version,
    RulesLangLangEnhancedPayoffStrategy.version,
}

PAYOFF_STRATEGIES = {
    RulesLangLangNativePayoffStrategy.version,
    RulesLangLangEnhancedPayoffStrategy.version,
}


@dataclass(frozen=True)
class OptimizerConfig:
    trades_csv: str
    kline_cache_dir: str
    out_dir: str
    strategy_version: str = "rules_v01"
    variants: list[StrategyVariant | LangLangV1Variant | LangLangV1_1Variant | LangLangV1_3Variant | LangLangNativeVariant | LangLangEnhancedVariant] | None = None
    top_n: int = 10
    max_variants: int | None = None
    min_validation_signals: int = 5
    max_validation_signals: int = 300
    train_start: str = "2022-05-24"
    train_end: str = "2023-12-31"
    validation_start: str = "2024-01-01"
    validation_end: str = "2024-06-20"
    strategy_library_registry_path: str | None = DEFAULT_REGISTRY_PATH
    strategy_library_db_path: str | None = DEFAULT_DB_PATH
    data_snapshot_id: str = "unspecified"
    feature_profile: str = "wyckoff_enhanced_v1_3"
    experiment_label: str = ""


@dataclass(frozen=True)
class OptimizerResult:
    leaderboard: list[dict[str, Any]]
    selected_config_path: str
    leaderboard_path: str
    report_path: str


@dataclass
class _ReplayPositionState:
    symbol: str
    side: Side
    entry_time: datetime
    entry_price: float
    remaining_qty: float
    leverage: int
    stop_loss: float
    risk_per_unit: float
    partial_take_profit_price: float
    runner_take_profit_price: float
    partial_exit_fraction: float
    partial_taken: bool = False


class HistoricalReplayOptimizer:
    def __init__(self, config: OptimizerConfig):
        self.config = config

    def run(self) -> OptimizerResult:
        out_dir = Path(self.config.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        trades = _read_trades(Path(self.config.trades_csv))
        symbols = sorted({trade["symbol"] for trade in trades})
        candles_by_symbol = {
            symbol: _read_daily_candles(Path(self.config.kline_cache_dir), symbol)
            for symbol in symbols
        }
        candles_by_symbol_bar = (
            {
                symbol: {
                    bar: _read_cached_candles(Path(self.config.kline_cache_dir), symbol, bar)
                    for bar in ("1D", "1H", "15m", "5m", "1m")
                }
                for symbol in symbols
            }
            if self.config.strategy_version in EVENT_REPLAY_STRATEGIES
            else {}
        )
        available_symbols = [symbol for symbol, candles in candles_by_symbol.items() if candles]
        trade_labels = TradeLabeler().label_trades(trades)
        excel_events_by_trade = _read_excel_event_evidence(Path(self.config.trades_csv))
        big_wins, big_losses = _classify_big_trades(trades, trade_labels)
        variants = self.config.variants or _default_variants(self.config.strategy_version)
        if self.config.max_variants is not None:
            variants = variants[: self.config.max_variants]
        historical_patterns = build_historical_patterns(trades)
        write_historical_patterns(out_dir / "historical_patterns.csv", historical_patterns)
        raw_rows = [
            self._score_variant(
                variant,
                candles_by_symbol,
                big_wins,
                big_losses,
                candles_by_symbol_bar,
                historical_patterns,
                excel_events_by_trade,
            )
            for variant in variants
        ]
        leaderboard = _rank_rows(raw_rows)
        eligible = [row for row in leaderboard if row["eligible"]]
        selected = eligible[: self.config.top_n]

        suffix = _output_suffix(self.config.strategy_version)
        leaderboard_path = out_dir / f"leaderboard{suffix}.csv"
        _write_leaderboard(leaderboard_path, leaderboard)
        fleet_config = self._build_fleet_config(selected, available_symbols, out_dir)
        selected_config_path = out_dir / f"selected_fleet_config{suffix}.json"
        selected_config_path.write_text(json.dumps(fleet_config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        report_path = out_dir / f"optimizer_report{suffix}.md"
        report_text = _render_report(leaderboard, selected)
        report_path.write_text(report_text, encoding="utf-8")
        self._append_strategy_library_run(
            leaderboard_path=leaderboard_path,
            selected_config_path=selected_config_path,
            report_path=report_path,
        )
        if _is_final_strategy(self.config.strategy_version):
            _write_final_distillation_artifacts(
                out_dir=out_dir,
                strategy_version=self.config.strategy_version,
                trades=trades,
                trade_labels=trade_labels,
                candles_by_symbol=candles_by_symbol,
                leaderboard=leaderboard,
                selected=selected,
            )
        if _is_payoff_strategy(self.config.strategy_version):
            _write_payoff_distillation_artifacts(
                out_dir=out_dir,
                strategy_version=self.config.strategy_version,
                trades=trades,
                trade_labels=trade_labels,
                candles_by_symbol=candles_by_symbol,
                excel_evidence=_read_excel_evidence(Path(self.config.trades_csv)),
                excel_events=excel_events_by_trade,
                leaderboard=leaderboard,
                selected=selected,
            )
        if self.config.strategy_version in {
            RulesLangLangNativeStrategy.version,
            RulesLangLangNativeFinalStrategy.version,
            RulesLangLangNativePayoffStrategy.version,
        }:
            (out_dir / "native_fit_report_v1.md").write_text(report_text, encoding="utf-8")
            if self.config.strategy_version == RulesLangLangNativeFinalStrategy.version:
                (out_dir / "native_fit_report_final.md").write_text(
                    _render_final_fit_report("native", leaderboard, selected),
                    encoding="utf-8",
                )
            if self.config.strategy_version == RulesLangLangNativePayoffStrategy.version:
                (out_dir / "native_payoff_fit_report_v1.md").write_text(
                    _render_payoff_fit_report("native_payoff", leaderboard, selected),
                    encoding="utf-8",
                )
        elif self.config.strategy_version in {
            RulesLangLangEnhancedStrategy.version,
            RulesLangLangEnhancedFinalStrategy.version,
            RulesLangLangEnhancedPayoffStrategy.version,
        }:
            (out_dir / "enhanced_fit_report_v1.md").write_text(report_text, encoding="utf-8")
            if self.config.strategy_version == RulesLangLangEnhancedFinalStrategy.version:
                (out_dir / "enhanced_fit_report_final.md").write_text(
                    _render_final_fit_report("enhanced", leaderboard, selected),
                    encoding="utf-8",
                )
            if self.config.strategy_version == RulesLangLangEnhancedPayoffStrategy.version:
                (out_dir / "enhanced_payoff_fit_report_v1.md").write_text(
                    _render_payoff_fit_report("enhanced_payoff", leaderboard, selected),
                    encoding="utf-8",
                )
        return OptimizerResult(
            leaderboard=leaderboard,
            selected_config_path=str(selected_config_path),
            leaderboard_path=str(leaderboard_path),
            report_path=str(report_path),
        )

    def _score_variant(
        self,
        variant: StrategyVariant,
        candles_by_symbol: dict[str, list[Candle]],
        big_wins: list[dict[str, Any]],
        big_losses: list[dict[str, Any]],
        candles_by_symbol_bar: dict[str, dict[str, list[Candle]]] | None = None,
        historical_patterns: list[dict[str, Any]] | None = None,
        excel_events_by_trade: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if self.config.strategy_version in EVENT_REPLAY_STRATEGIES:
            return self._score_variant_event_replay(
                variant,
                candles_by_symbol,
                big_wins,
                big_losses,
                candles_by_symbol_bar or {},
                historical_patterns or [],
                excel_events_by_trade or {},
            )
        strategy = RulesV01Strategy(variant)
        validation_start = _parse_day(self.config.validation_start)
        validation_end = _parse_day(self.config.validation_end)
        signal_events: list[dict[str, Any]] = []
        trade_returns: list[float] = []
        equity = 1.0
        peak = 1.0
        max_drawdown = 0.0
        for symbol, candles in candles_by_symbol.items():
            if len(candles) < 62:
                continue
            for idx in range(60, len(candles) - 1):
                current = candles[idx]
                current_day = _datetime_from_ms(current.ts)
                if current_day < validation_start or current_day > validation_end:
                    continue
                signal = strategy.generate(symbol, candles[: idx + 1])
                if signal is None:
                    continue
                next_close = candles[idx + 1].close
                ret = (next_close / current.close) - 1.0
                trade_returns.append(ret)
                equity *= 1.0 + ret
                peak = max(peak, equity)
                max_drawdown = max(max_drawdown, (peak - equity) / peak if peak else 0.0)
                signal_events.append({"symbol": symbol, "side": "long", "time": current_day})

        gross_profit = sum(ret for ret in trade_returns if ret > 0)
        gross_loss = abs(sum(ret for ret in trade_returns if ret < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)
        big_win_recall = _event_overlap_ratio(signal_events, big_wins)
        big_loss_overlap = _event_overlap_ratio(signal_events, big_losses)
        signal_count = len(signal_events)
        min_signals = _effective_min_validation_signals(self.config.strategy_version, self.config.min_validation_signals)
        eligible = min_signals <= signal_count <= self.config.max_validation_signals
        return {
            "variant_id": variant.variant_id,
            "eligible": eligible,
            "validation_signals": signal_count,
            "raw_validation_signals": signal_count,
            "validation_net_pnl": sum(trade_returns),
            "validation_profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
            "big_win_recall": big_win_recall,
            "big_loss_overlap": big_loss_overlap,
            "validation_realized_pnl_usdt": sum(trade_returns) * 10_000,
            "replay_mode": "next_close",
            "variant": variant,
            "excel_event_support_score": _excel_event_support_score(signal_events, big_wins, big_losses, excel_events_by_trade or {}),
        }

    def _score_variant_event_replay(
        self,
        variant: LangLangV1Variant | LangLangV1_1Variant,
        candles_by_symbol: dict[str, list[Candle]],
        big_wins: list[dict[str, Any]],
        big_losses: list[dict[str, Any]],
        candles_by_symbol_bar: dict[str, dict[str, list[Candle]]],
        historical_patterns: list[dict[str, Any]],
        excel_events_by_trade: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        strategy = strategy_from_version(self.config.strategy_version, variant)
        feature_builder = MultiTimeframeFeatureBuilder()
        pattern_matcher = HistoricalPatternMatcher(historical_patterns) if historical_patterns else None
        validation_start = _parse_day(self.config.validation_start)
        validation_end = _parse_day(self.config.validation_end)
        signal_events: list[dict[str, Any]] = []
        position_windows: list[dict[str, Any]] = []
        active_positions: dict[str, _ReplayPositionState] = {}
        raw_signal_count = 0
        diagnostics = _new_experiment_diagnostics()
        initial_equity = 10_000.0
        current_prices: dict[str, float] = {}
        risk = RiskEngine(RiskConfig(max_position_usdt=1_000.0, max_daily_loss_usdt=1_000.0, default_leverage=3))

        with tempfile.TemporaryDirectory() as tmp:
            ledger = Ledger(
                Path(tmp) / "replay.sqlite3",
                run_id="optimizer-v1",
                bot_id=variant.variant_id,
                variant_id=variant.variant_id,
            )
            executor = PaperExecutor(
                ledger=ledger,
                paper_config=PaperConfig(initial_equity_usdt=initial_equity, fee_bps=5, slippage_bps=5),
                price_provider=lambda symbol: current_prices[symbol],
            )
            for symbol, candles in candles_by_symbol.items():
                if len(candles) < 62:
                    continue
                for idx in range(60, len(candles)):
                    current = candles[idx]
                    current_day = _datetime_from_ms(current.ts)
                    if current_day < validation_start or current_day > validation_end:
                        continue
                    current_prices[symbol] = current.close
                    if _close_if_stop_hit_from_candle(
                        executor,
                        ledger,
                        current,
                        current_prices,
                        active_positions,
                        position_windows,
                        current_day,
                    ):
                        continue
                    snapshot = feature_builder.build(
                        symbol,
                        _feature_bars_for_replay(
                            symbol=symbol,
                            current_ts=current.ts,
                            daily_candles=candles[: idx + 1],
                            candles_by_symbol_bar=candles_by_symbol_bar,
                        ),
                    )
                    if snapshot is None:
                        continue
                    if self.config.strategy_version in HISTORICAL_MATCH_STRATEGIES and pattern_matcher is not None:
                        snapshot = _with_v1_1_historical_match(snapshot, pattern_matcher)
                    snapshot = _apply_experiment_feature_profile(snapshot, self.config.feature_profile)
                    _record_snapshot_attribution(diagnostics, snapshot.features)
                    if _apply_v1_exit_plan(
                        executor=executor,
                        ledger=ledger,
                        candle=current,
                        current_prices=current_prices,
                        active_positions=active_positions,
                        position_windows=position_windows,
                        current_day=current_day,
                        features=snapshot.features,
                        variant=variant,
                    ):
                        continue
                    if hasattr(strategy, "decide"):
                        decision = strategy.decide(snapshot)
                        _record_decision_diagnostic(diagnostics, decision)
                        signal = decision.signal if decision.action is StrategyAction.ENTER else None
                    else:
                        signal = strategy.generate_from_features(snapshot)
                    if signal is None:
                        continue
                    raw_signal_count += 1
                    if symbol in active_positions:
                        continue
                    intent = risk.intent_from_signal(
                        signal=signal,
                        account=executor.get_account(),
                        latest_price=current.close,
                        existing_position=None,
                    )
                    if intent is None:
                        ledger.record_risk_event(
                            "intent_rejected",
                            {"strength": signal.strength},
                            symbol=symbol,
                        )
                        continue
                    signal_id = ledger.record_signal(signal, self.config.strategy_version)
                    signal_events.append({"symbol": symbol, "side": signal.side.value, "time": current_day})
                    if len(signal_events) > self.config.max_validation_signals:
                        return _early_stopped_event_replay_row(
                            variant=variant,
                            validation_signals=len(signal_events),
                            raw_signal_count=raw_signal_count,
                            signal_events=signal_events,
                            position_windows=position_windows,
                            big_wins=big_wins,
                            big_losses=big_losses,
                            excel_events_by_trade=excel_events_by_trade,
                            diagnostics=diagnostics,
                        )
                    ledger.record_order_intent(intent, signal_id=signal_id)
                    result = executor.place_order(intent)
                    if result.filled_qty > 0:
                        _record_signal_diagnostic(diagnostics, signal)
                        active_positions[symbol] = _state_from_signal(
                            signal=signal,
                            entry_price=result.avg_price or current.close,
                            qty=result.filled_qty,
                            leverage=intent.leverage,
                            entry_time=current_day,
                            variant=variant,
                        )

            for position in list(executor.get_positions()):
                if position.symbol not in current_prices:
                    symbol_candles = candles_by_symbol.get(position.symbol, [])
                    if not symbol_candles:
                        continue
                    current_prices[position.symbol] = symbol_candles[-1].close
                state = active_positions.pop(position.symbol, None)
                if state is not None:
                    position_windows.append(_window_from_state(state, validation_end))
                executor.close_position(position.symbol, reason="validation_end")

            equities = [initial_equity] + [
                float(row["equity_usdt"])
                for row in ledger.list_rows("equity_snapshots", run_id="optimizer-v1", bot_id=variant.variant_id)
            ]
            realized_pnl = equities[-1] - initial_equity if equities else 0.0
            deltas = [(equities[idx] - equities[idx - 1]) / initial_equity for idx in range(1, len(equities))]
            gross_profit = sum(delta for delta in deltas if delta > 0)
            gross_loss = abs(sum(delta for delta in deltas if delta < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)
            peak = initial_equity
            max_drawdown = 0.0
            for equity in equities:
                peak = max(peak, equity)
                max_drawdown = max(max_drawdown, (peak - equity) / peak if peak else 0.0)

        signal_count = len(signal_events)
        min_signals = _effective_min_validation_signals(self.config.strategy_version, self.config.min_validation_signals)
        eligible = min_signals <= signal_count <= self.config.max_validation_signals
        big_win_recall = _capture_overlap_ratio(signal_events, position_windows, big_wins)
        big_loss_overlap = _capture_overlap_ratio(signal_events, position_windows, big_losses)
        payoff_metrics = _payoff_metrics(
            deltas=deltas,
            signal_events=signal_events,
            position_windows=position_windows,
            big_wins=big_wins,
            big_losses=big_losses,
        )
        return {
            "variant_id": variant.variant_id,
            "eligible": eligible,
            "validation_signals": signal_count,
            "raw_validation_signals": raw_signal_count,
            "validation_net_pnl": realized_pnl / initial_equity,
            "validation_profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
            "big_win_recall": big_win_recall,
            "big_loss_overlap": big_loss_overlap,
            "validation_realized_pnl_usdt": realized_pnl,
            "replay_mode": "event_replay",
            "variant": variant,
            "experiment_label": self.config.experiment_label or self.config.feature_profile,
            "feature_profile": self.config.feature_profile,
            "experiment_diagnostics": _freeze_experiment_diagnostics(diagnostics),
            "excel_event_support_score": _excel_event_support_score(
                signal_events,
                big_wins,
                big_losses,
                excel_events_by_trade,
            ),
            **payoff_metrics,
        }

    def _append_strategy_library_run(
        self,
        *,
        leaderboard_path: Path,
        selected_config_path: Path,
        report_path: Path,
    ) -> None:
        if not self.config.strategy_library_registry_path or not self.config.strategy_library_db_path:
            return
        registry_path = Path(self.config.strategy_library_registry_path)
        if not registry_path.exists():
            return
        run_id = f"optimizer:{self.config.strategy_version}:{Path(self.config.out_dir).name}"
        ingest_leaderboard(
            registry_path=registry_path,
            leaderboard_path=leaderboard_path,
            db_path=self.config.strategy_library_db_path,
            run_id=run_id,
            strategy_version=self.config.strategy_version,
            data_snapshot_id=self.config.data_snapshot_id,
            artifact_paths={
                "leaderboard": str(leaderboard_path),
                "selected_fleet_config": str(selected_config_path),
                "optimizer_report": str(report_path),
            },
        )

    def _build_fleet_config(self, selected: list[dict[str, Any]], symbols: list[str], out_dir: Path) -> FleetConfig:
        now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        bots = [
            BotConfig(bot_id=f"bot_{idx:02d}_{row['variant_id'][:24]}", variant=row["variant"])
            for idx, row in enumerate(selected, start=1)
        ]
        return FleetConfig(
            run_id=f"fleet-{now}",
            execution=(
                ExecutionConfig(mode="paper", exchange="multi", executor="paper_multi", allow_live_orders=False)
                if self.config.strategy_version in MULTI_EXCHANGE_SELECTION_STRATEGIES
                else ExecutionConfig(mode="paper", exchange="okx", executor="paper_okx", allow_live_orders=False)
            ),
            paper=PaperConfig(),
            risk=RiskConfig(),
            market_data=MarketDataConfig(
                symbols=[] if self.config.strategy_version in MULTI_EXCHANGE_SELECTION_STRATEGIES else (symbols or ["BTC-USDT-SWAP"]),
                bars=["1m", "5m", "15m", "1H", "4H", "1D"],
                max_fetch_workers=8 if self.config.strategy_version in EVENT_REPLAY_STRATEGIES else 1,
                cache_enabled=self.config.strategy_version in MULTI_EXCHANGE_SELECTION_STRATEGIES,
                cache_dir=str(out_dir / "kline_cache"),
                market_snapshot_cache_enabled=self.config.strategy_version in MULTI_EXCHANGE_SELECTION_STRATEGIES,
            ),
            universe=(
                UniverseConfig(
                    mode="okx_binance_usdt_swap_observe",
                    provider="okx_binance",
                    snapshot_path=str(out_dir / "universe_snapshot.json"),
                )
                if self.config.strategy_version in MULTI_EXCHANGE_SELECTION_STRATEGIES
                else UniverseConfig()
            ),
            selection=(
                SymbolSelectionConfig(
                    enabled=True,
                    style="dual_board",
                    scoring_profile=(
                        "native"
                        if self.config.strategy_version
                        in {
                            RulesLangLangNativeStrategy.version,
                            RulesLangLangNativeFinalStrategy.version,
                            RulesLangLangNativePayoffStrategy.version,
                        }
                        else "enhanced"
                    ),
                    top_n=0,
                    long_top_n=30,
                    short_top_n=20,
                )
                if self.config.strategy_version in MULTI_EXCHANGE_SELECTION_STRATEGIES
                else SymbolSelectionConfig(enabled=True, top_n=20)
            ),
            ledger_path=str(out_dir / "fleet.sqlite3"),
            strategy_version=self.config.strategy_version,
            historical_patterns_path=str(out_dir / "historical_patterns.csv"),
            bots=bots,
        )


def _read_trades(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    parsed = []
    for row in rows:
        parsed.append(
            {
                "trade_id": row.get("trade_id", ""),
                "entry_time": _parse_datetime(row["entry_time"]),
                "exit_time": _parse_datetime(row["exit_time"]) if row.get("exit_time") else None,
                "symbol": row["symbol"],
                "side": row["side"],
                "pnl_usdt": _float(row.get("pnl_usdt")),
                "return_rate": _float(row.get("return_rate")),
                "hold_minutes": _float(row.get("hold_minutes")),
            }
        )
    return parsed


def _read_daily_candles(cache_dir: Path, symbol: str) -> list[Candle]:
    return _read_cached_candles(cache_dir, symbol, "1D")


def _read_cached_candles(cache_dir: Path, symbol: str, bar: str) -> list[Candle]:
    paths = sorted((cache_dir / bar).glob(f"{symbol}_*.csv"))
    if not paths:
        return []
    rows: list[Candle] = []
    seen: set[int] = set()
    for path in paths:
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ts = int(float(row["ts"]))
                if ts in seen:
                    continue
                seen.add(ts)
                rows.append(
                    Candle(
                        symbol=symbol,
                        bar=bar,
                        ts=ts,
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=float(row.get("vol") or row.get("volume") or 0.0),
                    )
                )
    return sorted(rows, key=lambda candle: candle.ts)


def _classify_big_trades(
    trades: list[dict[str, Any]],
    trade_labels: dict[str, list[str]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if trade_labels is None:
        trade_labels = TradeLabeler().label_trades(trades)
    big_wins = [
        trade
        for trade in trades
        if TradeLabel.BIG_WIN.value in trade_labels.get(str(trade.get("trade_id", "")), [])
    ]
    big_losses = [
        trade
        for trade in trades
        if TradeLabel.BIG_LOSS.value in trade_labels.get(str(trade.get("trade_id", "")), [])
    ]
    return big_wins, big_losses


def _event_overlap_ratio(signal_events: list[dict[str, Any]], target_trades: list[dict[str, Any]]) -> float:
    if not target_trades:
        return 0.0
    hits = 0
    for trade in target_trades:
        for event in signal_events:
            if event["symbol"] != trade["symbol"] or event["side"] != trade["side"]:
                continue
            if abs((event["time"] - trade["entry_time"]).total_seconds()) <= 24 * 60 * 60:
                hits += 1
                break
    return hits / len(target_trades)


def _capture_overlap_ratio(
    signal_events: list[dict[str, Any]],
    position_windows: list[dict[str, Any]],
    target_trades: list[dict[str, Any]],
) -> float:
    if not target_trades:
        return 0.0
    hits = 0
    for trade in target_trades:
        if _trade_has_nearby_signal(signal_events, trade) or _trade_is_inside_position_window(position_windows, trade):
            hits += 1
    return hits / len(target_trades)


def _trade_has_nearby_signal(signal_events: list[dict[str, Any]], trade: dict[str, Any]) -> bool:
    for event in signal_events:
        if event["symbol"] != trade["symbol"] or event["side"] != trade["side"]:
            continue
        if abs((event["time"] - trade["entry_time"]).total_seconds()) <= 24 * 60 * 60:
            return True
    return False


def _trade_is_inside_position_window(position_windows: list[dict[str, Any]], trade: dict[str, Any]) -> bool:
    for window in position_windows:
        if window["symbol"] != trade["symbol"] or window["side"] != trade["side"]:
            continue
        if window["open_time"] <= trade["entry_time"] <= window["close_time"]:
            return True
    return False


def _payoff_metrics(
    *,
    deltas: list[float],
    signal_events: list[dict[str, Any]],
    position_windows: list[dict[str, Any]],
    big_wins: list[dict[str, Any]],
    big_losses: list[dict[str, Any]],
) -> dict[str, float]:
    wins = [delta for delta in deltas if delta > 0]
    losses = [abs(delta) for delta in deltas if delta < 0]
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else (avg_win if avg_win > 0 else 0.0)
    right_tail_capture = _capture_overlap_ratio(signal_events, position_windows, big_wins)
    big_loss_overlap = _capture_overlap_ratio(signal_events, position_windows, big_losses)
    right_tail_return_capture = _captured_return_share(signal_events, position_windows, big_wins, positive_only=True)
    left_tail_avoidance = 1.0 - big_loss_overlap
    payoff_asymmetry = min(1.0, win_loss_ratio / 3.0)
    max_single_loss = max(losses) if losses else 0.0
    loss_cap_score = max(0.0, 1.0 - min(1.0, max_single_loss / 0.10))
    return {
        "right_tail_capture_score": right_tail_capture,
        "right_tail_return_capture": right_tail_return_capture,
        "loss_suppression_score": left_tail_avoidance,
        "payoff_asymmetry_score": payoff_asymmetry,
        "avg_win_loss_ratio": win_loss_ratio,
        "max_single_loss": max_single_loss,
        "loss_cap_score": loss_cap_score,
    }


def _captured_return_share(
    signal_events: list[dict[str, Any]],
    position_windows: list[dict[str, Any]],
    target_trades: list[dict[str, Any]],
    *,
    positive_only: bool = False,
) -> float:
    if not target_trades:
        return 0.0
    total = 0.0
    captured = 0.0
    for trade in target_trades:
        value = _float(trade.get("return_rate"))
        if positive_only:
            value = max(0.0, value)
        else:
            value = abs(value)
        total += value
        if _trade_has_nearby_signal(signal_events, trade) or _trade_is_inside_position_window(position_windows, trade):
            captured += value
    return captured / total if total > 0 else 0.0


def _rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if rows and "right_tail_capture_score" in rows[0]:
        return _rank_payoff_rows(rows)
    net_pct = _percentiles([row["validation_net_pnl"] for row in rows])
    pf_pct = _percentiles([row["validation_profit_factor"] for row in rows])
    dd_pct = _percentiles([-row["max_drawdown"] for row in rows])
    recall_pct = _percentiles([row["big_win_recall"] for row in rows])
    loss_pct = _percentiles([-row["big_loss_overlap"] for row in rows])
    ranked: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        score = (
            0.30 * net_pct[idx]
            + 0.20 * pf_pct[idx]
            + 0.20 * dd_pct[idx]
            + 0.20 * recall_pct[idx]
            + 0.10 * loss_pct[idx]
        )
        item = dict(row)
        item.update(
            {
                "score": score if row["eligible"] else 0.0,
                "validation_net_pnl_percentile": net_pct[idx],
                "validation_profit_factor_percentile": pf_pct[idx],
                "inverse_max_drawdown_percentile": dd_pct[idx],
                "big_win_recall_percentile": recall_pct[idx],
                "inverse_big_loss_overlap_percentile": loss_pct[idx],
            }
        )
        ranked.append(item)
    ranked.sort(key=lambda row: _rank_sort_key(row, row["validation_net_pnl"]), reverse=True)
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return ranked


def _rank_payoff_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    right_tail_pct = _percentiles([row["right_tail_capture_score"] for row in rows])
    right_tail_return_pct = _percentiles([row["right_tail_return_capture"] for row in rows])
    loss_pct = _percentiles([row["loss_suppression_score"] for row in rows])
    asym_pct = _percentiles([row["payoff_asymmetry_score"] for row in rows])
    net_pct = _percentiles([row["validation_net_pnl"] for row in rows])
    structure_pct = _percentiles([row["big_win_recall"] - row["big_loss_overlap"] for row in rows])
    excel_event_pct = _percentiles([float(row.get("excel_event_support_score", 0.0)) for row in rows])
    ranked: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        score = (
            0.28 * right_tail_pct[idx]
            + 0.18 * right_tail_return_pct[idx]
            + 0.18 * loss_pct[idx]
            + 0.14 * asym_pct[idx]
            + 0.10 * net_pct[idx]
            + 0.05 * structure_pct[idx]
            + 0.07 * excel_event_pct[idx]
        )
        item = dict(row)
        item.update(
            {
                "score": score if row["eligible"] else 0.0,
                "validation_net_pnl_percentile": net_pct[idx],
                "validation_profit_factor_percentile": _percentiles(
                    [candidate["validation_profit_factor"] for candidate in rows]
                )[idx],
                "inverse_max_drawdown_percentile": _percentiles([-candidate["max_drawdown"] for candidate in rows])[idx],
                "big_win_recall_percentile": right_tail_pct[idx],
                "inverse_big_loss_overlap_percentile": loss_pct[idx],
                "right_tail_capture_percentile": right_tail_pct[idx],
                "right_tail_return_capture_percentile": right_tail_return_pct[idx],
                "loss_suppression_percentile": loss_pct[idx],
                "payoff_asymmetry_percentile": asym_pct[idx],
                "structure_fit_score": max(0.0, row["big_win_recall"] - row["big_loss_overlap"]),
                "excel_event_support_percentile": excel_event_pct[idx],
            }
        )
        ranked.append(item)
    ranked.sort(
        key=lambda row: _rank_sort_key(row, row["right_tail_capture_score"], row["validation_net_pnl"]),
        reverse=True,
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return ranked


def _rank_sort_key(row: dict[str, Any], *extra: Any) -> tuple[Any, ...]:
    has_signal = int(row.get("validation_signals") or 0) > 0
    return (has_signal, row.get("score", 0.0), *extra)


def _write_leaderboard(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "rank",
        "variant_id",
        "score",
        "eligible",
        "validation_signals",
        "raw_validation_signals",
        "validation_net_pnl",
        "validation_profit_factor",
        "max_drawdown",
        "big_win_recall",
        "big_loss_overlap",
        "validation_realized_pnl_usdt",
        "replay_mode",
        "validation_net_pnl_percentile",
        "validation_profit_factor_percentile",
        "inverse_max_drawdown_percentile",
        "big_win_recall_percentile",
        "inverse_big_loss_overlap_percentile",
    ]
    optional_fields = [
        "right_tail_capture_score",
        "right_tail_return_capture",
        "loss_suppression_score",
        "payoff_asymmetry_score",
        "avg_win_loss_ratio",
        "max_single_loss",
        "loss_cap_score",
        "right_tail_capture_percentile",
        "right_tail_return_capture_percentile",
        "loss_suppression_percentile",
        "payoff_asymmetry_percentile",
        "structure_fit_score",
        "excel_event_support_score",
        "excel_event_support_percentile",
    ]
    for field in optional_fields:
        if any(field in row for row in rows):
            fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_final_distillation_artifacts(
    *,
    out_dir: Path,
    strategy_version: str,
    trades: list[dict[str, Any]],
    trade_labels: dict[str, list[str]],
    candles_by_symbol: dict[str, list[Candle]],
    leaderboard: list[dict[str, Any]],
    selected: list[dict[str, Any]],
) -> None:
    distill_rows: list[dict[str, Any]] = []
    explanation_rows: list[dict[str, Any]] = []
    native_strategy = RulesLangLangNativeFinalStrategy(LangLangNativeVariant(variant_id="native_final_explainer"))
    enhanced_strategy = RulesLangLangEnhancedFinalStrategy(
        LangLangEnhancedVariant(variant_id="enhanced_final_explainer", exploratory=True)
    )
    for trade in trades:
        trade_id = str(trade.get("trade_id") or len(distill_rows))
        labels = trade_labels.get(trade_id, [])
        candles = candles_by_symbol.get(str(trade.get("symbol", "")), [])
        snapshot = _snapshot_before_trade(trade, candles)
        data_status = "available" if snapshot is not None else _trade_data_status(candles, trade)
        outcome_label = _expert_trade_label(trade, labels)
        mfe, mae = _trade_mfe_mae(trade, candles)
        state_payload = snapshot.features if snapshot is not None else {}
        native_explanation = "skip:data_unavailable"
        enhanced_explanation = "skip:data_unavailable"
        why_stop_or_hold = "data_unavailable_no_exit_plan"
        native_action = "skip"
        enhanced_action = "skip"
        entry_position_id = ""
        if snapshot is not None:
            native_decision = native_strategy.decide(snapshot)
            enhanced_decision = enhanced_strategy.decide(snapshot)
            native_action = native_decision.action.value
            enhanced_action = enhanced_decision.action.value
            native_explanation = native_decision.explanation
            enhanced_explanation = enhanced_decision.explanation
            if native_decision.signal is not None:
                entry_position_id = str(native_decision.signal.decision_trace.get("entry_position_id", ""))
                why_stop_or_hold = json.dumps(
                    {
                        "stop_loss": native_decision.signal.stop_loss,
                        "take_profit_plan": native_decision.signal.take_profit_plan,
                        "hold_plan": native_decision.signal.hold_plan,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            else:
                why_stop_or_hold = "|".join(code.value for code in native_decision.filter_codes) or native_explanation
        distill_rows.append(
            {
                "trade_id": trade_id,
                "symbol": trade.get("symbol", ""),
                "side": trade.get("side", ""),
                "entry_time": _iso(trade.get("entry_time")),
                "exit_time": _iso(trade.get("exit_time")),
                "state_action_outcome_status": data_status,
                "state_json": json.dumps(_compact_state(state_payload), ensure_ascii=False, sort_keys=True),
                "expert_action": "enter",
                "expert_trade_label": outcome_label,
                "raw_trade_labels": "|".join(labels),
                "pnl_usdt": trade.get("pnl_usdt", 0.0),
                "return_rate": trade.get("return_rate", 0.0),
                "mfe_pct": mfe,
                "mae_pct": mae,
                "entry_position_id": entry_position_id,
                "eligible_for_enhanced_learning": outcome_label in {"rational_win", "rational_loss", "right_tail"},
            }
        )
        explanation_rows.append(
            {
                "trade_id": trade_id,
                "symbol": trade.get("symbol", ""),
                "side": trade.get("side", ""),
                "data_status": data_status,
                "expert_trade_label": outcome_label,
                "native_action": native_action,
                "native_explanation": native_explanation,
                "enhanced_action": enhanced_action,
                "enhanced_explanation": enhanced_explanation,
                "why_stop_or_hold": why_stop_or_hold,
            }
        )
    _write_csv(out_dir / "distill_dataset_final.csv", distill_rows)
    _write_csv(out_dir / "trade_explanation_matrix_final.csv", explanation_rows)
    (out_dir / "overfit_audit_final.md").write_text(
        _render_overfit_audit(strategy_version, leaderboard, selected),
        encoding="utf-8",
    )


def _write_payoff_distillation_artifacts(
    *,
    out_dir: Path,
    strategy_version: str,
    trades: list[dict[str, Any]],
    trade_labels: dict[str, list[str]],
    candles_by_symbol: dict[str, list[Candle]],
    excel_evidence: dict[str, dict[str, str]] | None,
    excel_events: dict[str, dict[str, Any]] | None,
    leaderboard: list[dict[str, Any]],
    selected: list[dict[str, Any]],
) -> None:
    wave_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    explanation_rows: list[dict[str, Any]] = []
    strategy = (
        RulesLangLangNativePayoffStrategy(LangLangNativeVariant(variant_id="native_payoff_explainer"))
        if strategy_version == RulesLangLangNativePayoffStrategy.version
        else RulesLangLangEnhancedPayoffStrategy(
            LangLangEnhancedVariant(variant_id="enhanced_payoff_explainer", exploratory=True)
        )
    )
    for trade in trades:
        trade_id = str(trade.get("trade_id") or len(wave_rows))
        evidence = (excel_evidence or {}).get(trade_id, {})
        event_summary = (excel_events or {}).get(trade_id, {})
        candles = candles_by_symbol.get(str(trade.get("symbol", "")), [])
        snapshot = _snapshot_before_trade(trade, candles)
        data_status = "available" if snapshot is not None else _trade_data_status(candles, trade)
        labels = trade_labels.get(trade_id, [])
        outcome_label = _expert_trade_label(trade, labels)
        features = snapshot.features if snapshot is not None else {}
        wave_stage = _classify_wave_stage(features) if snapshot is not None else "data_unavailable"
        entry_position_id = _entry_position_from_wave(wave_stage, str(trade.get("side", "")), _float(trade.get("return_rate")))
        selection_reason_codes = _selection_reason_codes(features, str(trade.get("side", "")), wave_stage)
        selection_filter_codes = _selection_filter_codes(features, wave_stage)
        native_action = "skip"
        why_entry = "data_unavailable"
        why_exit = "data_unavailable"
        if snapshot is not None:
            decision = strategy.decide(
                FeatureSnapshot(
                    symbol=snapshot.symbol,
                    bar=snapshot.bar,
                    last_ts=snapshot.last_ts,
                    created_at=snapshot.created_at,
                    features={
                        **features,
                        "symbol_cycle": wave_stage,
                        "entry_position_id": entry_position_id,
                        "selection_reason_codes": selection_reason_codes,
                        "selection_filter_codes": selection_filter_codes,
                        **_replay_side_features(str(trade.get("side", ""))),
                    },
                )
            )
            native_action = decision.action.value
            why_entry = decision.explanation
            why_exit = _why_exit_or_hold(entry_position_id, outcome_label, _float(trade.get("return_rate")))
        base = {
            "trade_id": trade_id,
            "symbol": trade.get("symbol", ""),
            "side": trade.get("side", ""),
            "entry_time": _iso(trade.get("entry_time")),
            "exit_time": _iso(trade.get("exit_time")),
            "data_status": data_status,
            "return_rate": trade.get("return_rate", 0.0),
            "expert_trade_label": outcome_label,
            "excel_evidence_status": evidence.get("sheet_label_status", "excel_evidence_unavailable"),
            "excel_event_count": event_summary.get("event_count", 0),
            "excel_event_weight": event_summary.get("event_weight", 0.0),
        }
        wave_rows.append(
            {
                **base,
                "wave_stage": wave_stage,
                "entry_position_id": entry_position_id,
                "excel_sheet_membership": evidence.get("sheet_membership", "excel_evidence_unavailable"),
                "excel_btc_cycle_label": evidence.get("btc_cycle_label", ""),
                "excel_entry_position_label": evidence.get("entry_position_label", ""),
                "excel_manual_review_text": evidence.get("manual_review_text", ""),
                "excel_event_sources": event_summary.get("sources", ""),
                "excel_event_semantics": event_summary.get("semantics", ""),
                "market_time_guard": "features_ts_lte_entry_time",
                "ret_20d": features.get("ret_20d", ""),
                "ret_60d": features.get("ret_60d", ""),
                "pos_20d": features.get("pos_20d", ""),
                "pullback_from_20d_high": features.get("pullback_from_20d_high", ""),
                "vol_ratio_20d": features.get("vol_ratio_20d", ""),
            }
        )
        selection_rows.append(
            {
                **base,
                "selection_mode": "long_main_wave" if str(trade.get("side")) == "long" else "short_waterfall",
                "wave_stage": wave_stage,
                "selection_reason_codes": "|".join(selection_reason_codes),
                "selection_filter_codes": "|".join(selection_filter_codes) or "none",
                "excel_sheet_membership": evidence.get("sheet_membership", "excel_evidence_unavailable"),
                "excel_space_bucket": evidence.get("space_bucket", ""),
                "excel_stop_loss_bucket": evidence.get("stop_loss_bucket", ""),
                "excel_hold_time_bucket": evidence.get("hold_time_bucket", ""),
                "excel_symbol_profit_rank": evidence.get("symbol_profit_rank", ""),
                "excel_event_sources": event_summary.get("sources", ""),
                "excel_event_semantics": event_summary.get("semantics", ""),
                "ranking_data_status": data_status,
                "liquidity_scope": "top200_or_terminal_status",
            }
        )
        explanation_rows.append(
            {
                **base,
                "why_this_symbol": "|".join(selection_reason_codes) or "no_selection_reason",
                "why_this_stage": wave_stage,
                "why_this_price": entry_position_id,
                "excel_sheet_membership": evidence.get("sheet_membership", "excel_evidence_unavailable"),
                "excel_btc_cycle_label": evidence.get("btc_cycle_label", ""),
                "excel_entry_position_label": evidence.get("entry_position_label", ""),
                "excel_manual_review_text": evidence.get("manual_review_text", ""),
                "excel_space_bucket": evidence.get("space_bucket", ""),
                "excel_stop_loss_bucket": evidence.get("stop_loss_bucket", ""),
                "excel_hold_time_bucket": evidence.get("hold_time_bucket", ""),
                "excel_event_sources": event_summary.get("sources", ""),
                "excel_event_semantics": event_summary.get("semantics", ""),
                "why_entry": why_entry,
                "why_stop_or_take_profit": why_exit,
                "native_or_enhanced_action": native_action,
                "anti_future_function_guard": "snapshot_at_or_before_entry",
            }
        )
    _write_csv(out_dir / "wave_stage_dataset_v1.csv", wave_rows)
    _write_csv(out_dir / "symbol_selection_cross_section_v1.csv", selection_rows)
    _write_csv(out_dir / "trade_wave_explanation_matrix_v1.csv", explanation_rows)
    (out_dir / "right_tail_capture_report_v1.md").write_text(
        _render_right_tail_report(leaderboard, trades),
        encoding="utf-8",
    )
    (out_dir / "loss_suppression_report_v1.md").write_text(
        _render_loss_suppression_report(leaderboard, trades),
        encoding="utf-8",
    )


def _read_excel_evidence(trades_csv: Path) -> dict[str, dict[str, str]]:
    candidates = [
        trades_csv.parent / "excel_digest" / "trade_sheet_label_matrix.csv",
        trades_csv.parent.parent / "langlang_v1_3" / "excel_digest" / "trade_sheet_label_matrix.csv",
        Path("output/langlang_v1_3/excel_digest/trade_sheet_label_matrix.csv"),
    ]
    evidence_path = next((path for path in candidates if path.exists()), None)
    if evidence_path is None:
        return {}
    with evidence_path.open(newline="", encoding="utf-8") as handle:
        return {
            str(row.get("trade_id", "")): row
            for row in csv.DictReader(handle)
            if row.get("trade_id")
        }


def _read_excel_event_evidence(trades_csv: Path) -> dict[str, dict[str, Any]]:
    candidates = [
        trades_csv.parent / "excel_digest" / "excel_evidence_event_dataset.csv",
        trades_csv.parent.parent / "langlang_v1_3" / "excel_digest" / "excel_evidence_event_dataset.csv",
        Path("output/langlang_v1_3/excel_digest/excel_evidence_event_dataset.csv"),
    ]
    evidence_path = next((path for path in candidates if path.exists()), None)
    if evidence_path is None:
        return {}
    grouped: dict[str, dict[str, Any]] = {}
    with evidence_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            trade_id = str(row.get("trade_id") or "").strip()
            if not trade_id:
                continue
            item = grouped.setdefault(
                trade_id,
                {
                    "event_count": 0,
                    "event_weight": 0.0,
                    "sources_set": set(),
                    "semantics_set": set(),
                    "roles_set": set(),
                },
            )
            item["event_count"] += 1
            item["event_weight"] += _float(row.get("evidence_weight"), 1.0)
            if row.get("sheet_name"):
                item["sources_set"].add(str(row["sheet_name"]))
            if row.get("event_semantic"):
                item["semantics_set"].add(str(row["event_semantic"]))
            if row.get("evidence_role"):
                item["roles_set"].add(str(row["evidence_role"]))
    for item in grouped.values():
        item["sources"] = "|".join(sorted(item.pop("sources_set")))
        item["semantics"] = "|".join(sorted(item.pop("semantics_set")))
        item["roles"] = "|".join(sorted(item.pop("roles_set")))
    return grouped


def _excel_event_support_score(
    signal_events: list[dict[str, Any]],
    big_wins: list[dict[str, Any]],
    big_losses: list[dict[str, Any]],
    excel_events_by_trade: dict[str, dict[str, Any]],
) -> float:
    if not signal_events or not excel_events_by_trade:
        return 0.0
    support = 0.0
    risk = 0.0
    for trade in big_wins:
        if _trade_has_nearby_signal(signal_events, trade):
            summary = excel_events_by_trade.get(str(trade.get("trade_id", "")), {})
            support += float(summary.get("event_weight", 0.0))
    for trade in big_losses:
        if _trade_has_nearby_signal(signal_events, trade):
            summary = excel_events_by_trade.get(str(trade.get("trade_id", "")), {})
            risk += float(summary.get("event_weight", 0.0))
    denominator = support + risk
    if denominator <= 0:
        return 0.0
    return max(0.0, min(1.0, support / denominator))


def _classify_wave_stage(features: dict[str, Any]) -> str:
    ret_20d = _float(features.get("ret_20d"))
    ret_60d = _float(features.get("ret_60d"))
    pos_20d = _float(features.get("pos_20d"), 0.5)
    pullback = _float(features.get("pullback_from_20d_high"))
    vol_ratio = _float(features.get("vol_ratio_20d"), 1.0)
    h1_ret_24 = _float(features.get("h1_ret_24"))
    if ret_20d <= -0.16 or ret_60d <= -0.28 or h1_ret_24 <= -0.06:
        return "weak_waterfall"
    if abs(ret_20d) <= 0.06 and 0.32 <= pos_20d <= 0.68:
        return "base_platform"
    if pos_20d >= 0.94 and pullback >= -0.025 and 0.10 <= ret_20d <= 0.22:
        return "exhaustion"
    if ret_20d >= 0.24 and ret_60d >= 0.55 and pullback >= -0.08 and pos_20d >= 0.65:
        return "main_wave"
    if ret_20d >= 0.16 and ret_60d >= 0.38 and pullback >= -0.06 and vol_ratio >= 1.15:
        return "wave_start"
    if ret_20d >= 0.12 and -0.12 <= pullback <= -0.025:
        return "small_divergence_1" if pos_20d >= 0.55 else "small_divergence_2"
    if ret_60d >= 0.35 and pullback < -0.12:
        return "large_divergence"
    if ret_20d >= 0.10 and ret_60d >= 0.45 and 0.45 <= pos_20d <= 0.78:
        return "second_wave"
    return "box_chop"


def _entry_position_from_wave(wave_stage: str, side: str, return_rate: float = 0.0) -> str:
    side = side.lower()
    if side == "short":
        if wave_stage in {"weak_waterfall", "box_chop"}:
            return "short_waterfall_continuation"
        if wave_stage in {"large_divergence", "exhaustion"}:
            return "3_or_5_countertrend_top_short"
        return "short_rebound_failure"
    if wave_stage in {"wave_start", "main_wave", "base_platform"}:
        return "1_startup_long"
    if wave_stage.startswith("small_divergence"):
        return "2_small_divergence_low_absorb"
    if wave_stage in {"large_divergence", "second_wave"}:
        return "4_second_wave_start_long"
    if wave_stage == "box_chop":
        return "6_box_rebound_long"
    if wave_stage == "exhaustion":
        return "skip_high_position_breakout"
    return "unclassified_entry"


def _selection_reason_codes(features: dict[str, Any], side: str, wave_stage: str) -> list[str]:
    if not features:
        return []
    ret_20d = _float(features.get("ret_20d"))
    ret_60d = _float(features.get("ret_60d"))
    pos_20d = _float(features.get("pos_20d"), 0.5)
    vol_ratio = _float(features.get("vol_ratio_20d"), 1.0)
    reasons: list[str] = []
    if side == "long":
        if ret_20d >= 0.16 and ret_60d >= 0.38:
            reasons.append("leader_altcoin")
        if pos_20d >= 0.55:
            reasons.append("daily_structure_strong")
        if vol_ratio >= 1.15:
            reasons.append("turnover_expansion")
        if wave_stage in {"wave_start", "main_wave", "second_wave"}:
            reasons.append("main_wave_candidate")
    else:
        if wave_stage == "weak_waterfall":
            reasons.append("waterfall_weak_coin")
        if ret_20d <= -0.12:
            reasons.append("relative_weakness")
    return reasons


def _selection_filter_codes(features: dict[str, Any], wave_stage: str) -> list[str]:
    filters: list[str] = []
    if wave_stage == "exhaustion":
        filters.append("high_position_breakout_risk")
    if wave_stage == "box_chop":
        filters.append("box_chop_low_quality")
    if _float(features.get("pullback_from_20d_high")) < -0.18:
        filters.append("structure_deep_pullback")
    return filters


def _replay_side_features(side: str) -> dict[str, str]:
    normalized = str(side or "").lower()
    if normalized == "short":
        return {
            "requested_side": "short",
            "selection_bias": "short",
            "selection_mode": "short_waterfall",
            "symbol_selection_tag": "short_waterfall",
        }
    return {
        "requested_side": "long",
        "selection_bias": "long",
        "selection_mode": "long_main_wave",
    }


def _why_exit_or_hold(entry_position_id: str, outcome_label: str, return_rate: float) -> str:
    if entry_position_id in {"1_startup_long", "4_second_wave_start_long"}:
        return "allow_runner_for_right_tail" if return_rate > 0 else "structure_or_mae_stop"
    if "small_divergence" in entry_position_id:
        return "short_hold_take_profit_or_fast_stop"
    if entry_position_id in {"6_box_rebound_long", "skip_high_position_breakout"}:
        return "short_hold_or_skip_low_quality"
    if "short" in entry_position_id:
        return "waterfall_follow_or_rebound_failure_stop"
    return f"outcome_label:{outcome_label}"


def _snapshot_before_trade(trade: dict[str, Any], candles: list[Candle]) -> FeatureSnapshot | None:
    entry_time = trade.get("entry_time")
    if not isinstance(entry_time, datetime):
        return None
    entry_ts = int(entry_time.timestamp() * 1000)
    rows = [candle for candle in candles if candle.ts <= entry_ts]
    return DailyFeatureBuilder().build(str(trade.get("symbol", "")), rows)


def _trade_data_status(candles: list[Candle], trade: dict[str, Any]) -> str:
    if not candles:
        return "exchange_unavailable"
    entry_time = trade.get("entry_time")
    if isinstance(entry_time, datetime) and candles and int(entry_time.timestamp() * 1000) < candles[0].ts:
        return "listing_boundary"
    return "indicator_warmup"


def _expert_trade_label(trade: dict[str, Any], labels: list[str]) -> str:
    pnl = _float(trade.get("pnl_usdt"))
    if TradeLabel.RIGHT_TAIL.value in labels:
        return "right_tail"
    if TradeLabel.FAST_FAILURE.value in labels or TradeLabel.CHASE_FAILURE.value in labels:
        return "emotional_loss"
    if TradeLabel.BIG_LOSS.value in labels:
        return "quick_fail" if _float(trade.get("hold_minutes")) <= 60 else "rational_loss"
    if pnl > 0:
        return "rational_win"
    if pnl < 0:
        return "rational_loss"
    return "noise_or_unexplained"


def _trade_mfe_mae(trade: dict[str, Any], candles: list[Candle]) -> tuple[float, float]:
    entry_time = trade.get("entry_time")
    exit_time = trade.get("exit_time")
    if not isinstance(entry_time, datetime) or not candles:
        return 0.0, 0.0
    start_ts = int(entry_time.timestamp() * 1000)
    end_dt = exit_time if isinstance(exit_time, datetime) else entry_time
    end_ts = max(start_ts, int(end_dt.timestamp() * 1000))
    window = [candle for candle in candles if start_ts <= candle.ts <= end_ts]
    if not window:
        return 0.0, 0.0
    entry_price = window[0].open or window[0].close
    if entry_price <= 0:
        return 0.0, 0.0
    side = Side.from_value(str(trade.get("side", "long")))
    highs = [(candle.high / entry_price - 1.0) * side.sign for candle in window]
    lows = [(candle.low / entry_price - 1.0) * side.sign for candle in window]
    return max(highs + lows), min(highs + lows)


def _compact_state(features: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "ret_3d",
        "ret_7d",
        "ret_20d",
        "ret_60d",
        "pos_20d",
        "pullback_from_20d_high",
        "ma_5",
        "ma_20",
        "ma_60",
        "macd_hist",
        "atr_14",
        "rsi_14",
        "vol_ratio_20d",
        "latest_close",
    )
    return {key: features[key] for key in keys if key in features}


def _render_final_fit_report(strategy_line: str, leaderboard: list[dict[str, Any]], selected: list[dict[str, Any]]) -> str:
    best = leaderboard[0] if leaderboard else {}
    return "\n".join(
        [
            f"# LangLang {strategy_line.title()} Final Fit Report",
            "",
            f"- strategy_line: {strategy_line}",
            f"- variants_scored: {len(leaderboard)}",
            f"- selected_bots: {len(selected)}",
            f"- best_variant_id: {best.get('variant_id', '')}",
            f"- best_score: {float(best.get('score', 0.0)):.6f}",
            f"- best_validation_profit_factor: {float(best.get('validation_profit_factor', 0.0)):.6f}",
            f"- best_big_win_recall: {float(best.get('big_win_recall', 0.0)):.6f}",
            f"- best_big_loss_overlap: {float(best.get('big_loss_overlap', 0.0)):.6f}",
            "",
            "## Interpretation",
            "- native focuses on document-fit and explanation coverage.",
            "- enhanced keeps the native direction and only filters, reduces size, or limits holding plans.",
        ]
    ) + "\n"


def _render_payoff_fit_report(strategy_line: str, leaderboard: list[dict[str, Any]], selected: list[dict[str, Any]]) -> str:
    best = leaderboard[0] if leaderboard else {}
    return "\n".join(
        [
            f"# LangLang {strategy_line.title()} Fit Report",
            "",
            f"- strategy_line: {strategy_line}",
            "- objective: asymmetric_payoff_boda_sunxiao",
            f"- variants_scored: {len(leaderboard)}",
            f"- selected_bots: {len(selected)}",
            f"- best_variant_id: {best.get('variant_id', '')}",
            f"- best_score: {float(best.get('score', 0.0)):.6f}",
            f"- right_tail_capture_score: {float(best.get('right_tail_capture_score', 0.0)):.6f}",
            f"- right_tail_return_capture: {float(best.get('right_tail_return_capture', 0.0)):.6f}",
            f"- loss_suppression_score: {float(best.get('loss_suppression_score', 0.0)):.6f}",
            f"- payoff_asymmetry_score: {float(best.get('payoff_asymmetry_score', 0.0)):.6f}",
            f"- avg_win_loss_ratio: {float(best.get('avg_win_loss_ratio', 0.0)):.6f}",
            "",
            "## Interpretation",
            "- Native payoff evaluates document logic by right-tail and loss-shape fit.",
            "- Enhanced payoff keeps native direction and can only skip, reduce size, short-hold, or allow runner.",
        ]
    ) + "\n"


def _render_right_tail_report(leaderboard: list[dict[str, Any]], trades: list[dict[str, Any]]) -> str:
    returns = sorted((_float(trade.get("return_rate")) for trade in trades), reverse=True)
    top_5_count = max(1, int(len(returns) * 0.05)) if returns else 0
    top_5_sum = sum(returns[:top_5_count])
    total_sum = sum(returns)
    best = leaderboard[0] if leaderboard else {}
    return "\n".join(
        [
            "# Right Tail Capture Report",
            "",
            f"- trades: {len(trades)}",
            f"- top_5_percent_count: {top_5_count}",
            f"- top_5_percent_return_sum: {top_5_sum:.6f}",
            f"- total_return_sum: {total_sum:.6f}",
            f"- top_5_percent_contribution_ratio: {(top_5_sum / total_sum if total_sum else 0.0):.6f}",
            f"- best_right_tail_capture_score: {float(best.get('right_tail_capture_score', 0.0)):.6f}",
            f"- best_right_tail_return_capture: {float(best.get('right_tail_return_capture', 0.0)):.6f}",
            "",
            "## Gate",
            "- Strategy is not accepted unless right-tail capture improves without creating large-loss overlap.",
        ]
    ) + "\n"


def _render_loss_suppression_report(leaderboard: list[dict[str, Any]], trades: list[dict[str, Any]]) -> str:
    losses = sorted((_float(trade.get("return_rate")) for trade in trades if _float(trade.get("return_rate")) < 0))
    bottom_20_count = max(1, int(len(trades) * 0.20)) if trades else 0
    bottom_20_sum = sum(losses[:bottom_20_count])
    best = leaderboard[0] if leaderboard else {}
    return "\n".join(
        [
            "# Loss Suppression Report",
            "",
            f"- losing_trades: {len(losses)}",
            f"- bottom_20_percent_count: {bottom_20_count}",
            f"- bottom_20_percent_return_sum: {bottom_20_sum:.6f}",
            f"- best_loss_suppression_score: {float(best.get('loss_suppression_score', 0.0)):.6f}",
            f"- best_big_loss_overlap: {float(best.get('big_loss_overlap', 0.0)):.6f}",
            f"- best_max_single_loss: {float(best.get('max_single_loss', 0.0)):.6f}",
            f"- best_avg_win_loss_ratio: {float(best.get('avg_win_loss_ratio', 0.0)):.6f}",
            "",
            "## Gate",
            "- Enhanced is not accepted unless it lowers big-loss overlap or improves payoff asymmetry versus native.",
        ]
    ) + "\n"


def _render_overfit_audit(strategy_version: str, leaderboard: list[dict[str, Any]], selected: list[dict[str, Any]]) -> str:
    trial_count = len(leaderboard)
    best = leaderboard[0] if leaderboard else {}
    effective_trials = max(1, trial_count)
    best_score = float(best.get("score", 0.0))
    deflated_score_proxy = best_score / (1.0 + effective_trials ** 0.5 / 10.0)
    return "\n".join(
        [
            "# LangLang Final Overfit Audit",
            "",
            f"- strategy_version: {strategy_version}",
            "- validation_method: event_replay_with_fixed_train_validation_split",
            "- train_window: 2022-05-24..2023-12-31",
            "- validation_window: 2024-01-01..2024-06-20",
            f"- variants_scored: {trial_count}",
            f"- selected_bots: {len(selected)}",
            f"- best_score: {best_score:.6f}",
            f"- deflated_score_proxy: {deflated_score_proxy:.6f}",
            "- multiple_testing_policy: record_all_trials_and_deflate_best_score_proxy",
            "- leakage_guard: features_use_candles_at_or_before_signal_time",
            "",
            "## Notes",
            "- This audit is a lightweight Deflated-Sharpe-style proxy, not a full CPCV implementation.",
            "- Final paper acceptance still requires out-of-sample fleet results before any live authorization.",
        ]
    ) + "\n"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _is_final_strategy(strategy_version: str) -> bool:
    return strategy_version in {RulesLangLangNativeFinalStrategy.version, RulesLangLangEnhancedFinalStrategy.version}


def _is_payoff_strategy(strategy_version: str) -> bool:
    return strategy_version in PAYOFF_STRATEGIES


def _effective_min_validation_signals(strategy_version: str, configured_min: int) -> int:
    if _is_final_strategy(strategy_version) or _is_payoff_strategy(strategy_version):
        return max(1, configured_min)
    return configured_min


def _iso(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return ""


def _render_report(leaderboard: list[dict[str, Any]], selected: list[dict[str, Any]]) -> str:
    replay_modes = sorted({row.get("replay_mode", "next_close") for row in leaderboard})
    lines = [
        "# Fleet Optimizer Report",
        "",
        f"- variants_scored: {len(leaderboard)}",
        f"- selected_bots: {len(selected)}",
        f"- replay_modes: {', '.join(replay_modes)}",
        "",
        "## Selected",
    ]
    for row in selected:
        lines.append(
            f"- rank {row['rank']}: {row['variant_id']} score={row['score']:.4f} "
            f"signals={row['validation_signals']} net={row['validation_net_pnl']:.4f}"
        )
    return "\n".join(lines) + "\n"


def _percentiles(values: list[float]) -> list[float]:
    if not values:
        return []
    unique = sorted(values)
    if len(unique) == 1 or unique[0] == unique[-1]:
        return [0.5 for _ in values]
    return [(sum(1 for value in unique if value <= current) - 1) / (len(unique) - 1) for current in values]


def _parse_datetime(value: str) -> datetime:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"unsupported datetime: {value}")


def _parse_day(value: str) -> datetime:
    return _parse_datetime(value)


def _datetime_from_ms(value: int) -> datetime:
    return datetime.fromtimestamp(value / 1000, tz=timezone.utc)


def _float(value: Any, default: float = 0.0) -> float:
    if value in {None, ""}:
        return default
    return float(value)


def _default_variants(strategy_version: str) -> list[StrategyVariant | LangLangV1Variant | LangLangV1_1Variant | LangLangV1_3Variant | LangLangNativeVariant | LangLangEnhancedVariant]:
    if strategy_version in {
        RulesLangLangEnhancedStrategy.version,
        RulesLangLangEnhancedFinalStrategy.version,
        RulesLangLangEnhancedPayoffStrategy.version,
    }:
        return default_langlang_enhanced_grid()
    if strategy_version in {
        RulesLangLangNativeStrategy.version,
        RulesLangLangNativeFinalStrategy.version,
        RulesLangLangNativePayoffStrategy.version,
    }:
        return default_langlang_native_grid()
    if strategy_version == RulesLangLangV1_3Strategy.version:
        return default_langlang_v1_3_grid()
    if strategy_version == RulesLangLangV1_2Strategy.version:
        return default_langlang_v1_1_grid()
    if strategy_version == RulesLangLangV1_1Strategy.version:
        return default_langlang_v1_1_grid()
    if strategy_version == RulesLangLangV1Strategy.version:
        return default_langlang_v1_grid()
    return default_variant_grid()


def _output_suffix(strategy_version: str) -> str:
    if strategy_version == RulesLangLangEnhancedPayoffStrategy.version:
        return "_enhanced_payoff_v1"
    if strategy_version == RulesLangLangNativePayoffStrategy.version:
        return "_native_payoff_v1"
    if strategy_version == RulesLangLangEnhancedFinalStrategy.version:
        return "_enhanced_final"
    if strategy_version == RulesLangLangNativeFinalStrategy.version:
        return "_native_final"
    if strategy_version == RulesLangLangEnhancedStrategy.version:
        return "_enhanced_v1"
    if strategy_version == RulesLangLangNativeStrategy.version:
        return "_native_v1"
    if strategy_version == RulesLangLangV1_3Strategy.version:
        return "_v1_3"
    if strategy_version == RulesLangLangV1_2Strategy.version:
        return "_v1_2"
    if strategy_version == RulesLangLangV1_1Strategy.version:
        return "_v1_1"
    return ""


def _state_from_signal(
    *,
    signal: Any,
    entry_price: float,
    qty: float,
    leverage: int,
    entry_time: datetime,
    variant: LangLangV1Variant,
) -> _ReplayPositionState:
    stop_loss = float(signal.stop_loss)
    risk_per_unit = max(abs(entry_price - stop_loss), entry_price * 0.002)
    take_profit_plan = getattr(signal, "take_profit_plan", {}) or {}
    partial_r = float(take_profit_plan.get("partial_r", variant.partial_take_profit_r))
    runner_r = float(take_profit_plan.get("runner_r", variant.runner_take_profit_r))
    partial_fraction = float(take_profit_plan.get("partial_exit_fraction", variant.partial_exit_fraction))
    partial_fraction = min(1.0, max(0.0, partial_fraction))
    return _ReplayPositionState(
        symbol=signal.symbol,
        side=signal.side,
        entry_time=entry_time,
        entry_price=entry_price,
        remaining_qty=qty,
        leverage=leverage,
        stop_loss=stop_loss,
        risk_per_unit=risk_per_unit,
        partial_take_profit_price=entry_price + signal.side.sign * risk_per_unit * partial_r,
        runner_take_profit_price=entry_price + signal.side.sign * risk_per_unit * runner_r,
        partial_exit_fraction=partial_fraction,
    )


def _apply_experiment_feature_profile(snapshot: FeatureSnapshot, profile: str) -> FeatureSnapshot:
    if not profile:
        return snapshot
    from langlang_trader.experiment_matrix import apply_feature_profile

    return apply_feature_profile(snapshot, profile)


def _new_experiment_diagnostics() -> dict[str, Any]:
    return {
        "snapshot_count": 0,
        "skip_filter_counts": Counter(),
        "skip_explanation_counts": Counter(),
        "action_counts": Counter(),
        "entry_position_counts": Counter(),
        "signal_reason_counts": Counter(),
        "strong_pattern_tag_counts": Counter(),
        "risk_pattern_tag_counts": Counter(),
        "wyckoff_phase_counts": Counter(),
        "wyckoff_long_setup_counts": Counter(),
        "wyckoff_short_setup_counts": Counter(),
        "strong_pattern_score_bins": Counter(),
        "risk_pattern_score_bins": Counter(),
        "wyckoff_long_score_bins": Counter(),
        "wyckoff_short_score_bins": Counter(),
        "wyckoff_exit_score_bins": Counter(),
        "strong_pattern_released_signals": 0,
        "wyckoff_released_signals": 0,
        "risk_filtered_skips": 0,
    }


def _record_snapshot_attribution(diagnostics: dict[str, Any], features: dict[str, Any]) -> None:
    diagnostics["snapshot_count"] += 1
    diagnostics["strong_pattern_tag_counts"][_tag(features.get("strong_pattern_tag"))] += 1
    diagnostics["risk_pattern_tag_counts"][_tag(features.get("risk_pattern_tag"))] += 1
    diagnostics["wyckoff_phase_counts"][_tag(features.get("wyckoff_phase_tag"), default="none")] += 1
    diagnostics["wyckoff_long_setup_counts"][_tag(features.get("wyckoff_long_setup_tag"))] += 1
    diagnostics["wyckoff_short_setup_counts"][_tag(features.get("wyckoff_short_setup_tag"))] += 1
    for field in (
        "strong_pattern_score",
        "risk_pattern_score",
        "wyckoff_long_score",
        "wyckoff_short_score",
        "wyckoff_exit_score",
    ):
        diagnostics[f"{field}_bins"][_score_bin(features.get(field, 0.0))] += 1


def _record_decision_diagnostic(diagnostics: dict[str, Any], decision: Any) -> None:
    action = getattr(getattr(decision, "action", None), "value", str(getattr(decision, "action", ""))) or "unknown"
    diagnostics["action_counts"][action] += 1
    if action == StrategyAction.SKIP.value:
        explanation = str(getattr(decision, "explanation", "skip:unknown"))
        diagnostics["skip_explanation_counts"][explanation] += 1
        filters = [
            getattr(code, "value", str(code))
            for code in getattr(decision, "filter_codes", [])
        ]
        if not filters:
            filters = ["unknown_skip_filter"]
        for code in filters:
            diagnostics["skip_filter_counts"][code] += 1
            if code in {
                "wyckoff_risk",
                "five_wave_late_risk",
                "false_breakout_after_contraction",
                "third_small_divergence",
            }:
                diagnostics["risk_filtered_skips"] += 1


def _record_signal_diagnostic(diagnostics: dict[str, Any], signal: Any) -> None:
    features = getattr(signal, "features", {}) or {}
    trace = getattr(signal, "decision_trace", {}) or {}
    entry_position_id = str(trace.get("entry_position_id") or features.get("entry_position_id") or "unknown_entry")
    diagnostics["entry_position_counts"][entry_position_id] += 1
    reason_codes = [str(code) for code in getattr(signal, "reason_codes", [])]
    for code in reason_codes:
        diagnostics["signal_reason_counts"][code] += 1
    if any(code.startswith("strong_pattern") or "strong_pattern" in code for code in reason_codes):
        diagnostics["strong_pattern_released_signals"] += 1
    if any(code.startswith("wyckoff") or "wyckoff" in code for code in reason_codes):
        diagnostics["wyckoff_released_signals"] += 1


def _freeze_experiment_diagnostics(diagnostics: dict[str, Any]) -> dict[str, Any]:
    frozen: dict[str, Any] = {}
    for key, value in diagnostics.items():
        if isinstance(value, Counter):
            frozen[key] = dict(value)
        else:
            frozen[key] = value
    return frozen


def _tag(value: Any, *, default: str = "none") -> str:
    text = str(value or "").strip()
    return text or default


def _score_bin(value: Any) -> str:
    number = _float(value)
    if number < 0.45:
        return "score_000_045"
    if number < 0.65:
        return "score_045_065"
    if number < 0.70:
        return "score_065_070"
    return "score_070_100"


def _early_stopped_event_replay_row(
    *,
    variant: LangLangV1Variant,
    validation_signals: int,
    raw_signal_count: int,
    signal_events: list[dict[str, Any]],
    position_windows: list[dict[str, Any]],
    big_wins: list[dict[str, Any]],
    big_losses: list[dict[str, Any]],
    excel_events_by_trade: dict[str, dict[str, Any]] | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payoff_metrics = _payoff_metrics(
        deltas=[],
        signal_events=signal_events,
        position_windows=position_windows,
        big_wins=big_wins,
        big_losses=big_losses,
    )
    return {
        "variant_id": variant.variant_id,
        "eligible": False,
        "validation_signals": validation_signals,
        "raw_validation_signals": raw_signal_count,
        "validation_net_pnl": 0.0,
        "validation_profit_factor": 0.0,
        "max_drawdown": 1.0,
        "big_win_recall": _capture_overlap_ratio(signal_events, position_windows, big_wins),
        "big_loss_overlap": _capture_overlap_ratio(signal_events, position_windows, big_losses),
        "validation_realized_pnl_usdt": 0.0,
        "replay_mode": "event_replay_early_stop",
        "variant": variant,
        "experiment_diagnostics": _freeze_experiment_diagnostics(diagnostics or _new_experiment_diagnostics()),
        "excel_event_support_score": _excel_event_support_score(
            signal_events,
            big_wins,
            big_losses,
            excel_events_by_trade or {},
        ),
        **payoff_metrics,
    }


def _window_from_state(state: _ReplayPositionState, close_time: datetime) -> dict[str, Any]:
    return {
        "symbol": state.symbol,
        "side": state.side.value,
        "open_time": state.entry_time,
        "close_time": close_time,
    }


def _close_if_stop_hit_from_candle(
    executor: PaperExecutor,
    ledger: Ledger,
    candle: Candle,
    current_prices: dict[str, float],
    active_positions: dict[str, _ReplayPositionState] | None = None,
    position_windows: list[dict[str, Any]] | None = None,
    current_day: datetime | None = None,
) -> bool:
    state = active_positions.get(candle.symbol) if active_positions is not None else None
    if active_positions is not None and state is None:
        return False
    side = state.side if state is not None else None
    stop_loss = state.stop_loss if state is not None else None
    if side is None or stop_loss is None:
        position = ledger.get_position(candle.symbol)
        if position is None:
            return False
        side = position.side
        stop_loss = ledger.latest_stop_loss(candle.symbol)
    if stop_loss is None:
        return False
    if side is Side.LONG and candle.low <= stop_loss:
        current_prices[candle.symbol] = stop_loss
        executor.close_position(candle.symbol, reason="stop_loss_exit")
        _finish_active_position(candle.symbol, active_positions, position_windows, current_day)
        return True
    elif side is Side.SHORT and candle.high >= stop_loss:
        current_prices[candle.symbol] = stop_loss
        executor.close_position(candle.symbol, reason="stop_loss_exit")
        _finish_active_position(candle.symbol, active_positions, position_windows, current_day)
        return True
    return False


def _apply_v1_exit_plan(
    *,
    executor: PaperExecutor,
    ledger: Ledger,
    candle: Candle,
    current_prices: dict[str, float],
    active_positions: dict[str, _ReplayPositionState],
    position_windows: list[dict[str, Any]],
    current_day: datetime,
    features: dict[str, Any],
    variant: LangLangV1Variant,
) -> bool:
    state = active_positions.get(candle.symbol)
    if state is None:
        return False

    if not state.partial_taken and _price_touched(candle, state.side, state.partial_take_profit_price):
        reduce_qty = min(state.remaining_qty, state.remaining_qty * state.partial_exit_fraction)
        if reduce_qty > 0:
            current_prices[candle.symbol] = state.partial_take_profit_price
            result = executor.place_order(
                OrderIntent(
                    symbol=candle.symbol,
                    side=_opposite_side(state.side),
                    order_type="market",
                    qty=reduce_qty,
                    leverage=state.leverage,
                    reduce_only=True,
                    entry_reason=f"reduce:{ExitPlan.PARTIAL_TAKE_PROFIT.value}",
                    stop_loss=None,
                    max_slippage_bps=executor.paper_config.slippage_bps,
                )
            )
            ledger.record_risk_event(
                "partial_take_profit_exit",
                {
                    "target_price": state.partial_take_profit_price,
                    "status": result.status,
                    "exchange_order_id": result.exchange_order_id,
                    "filled_qty": result.filled_qty,
                },
                symbol=candle.symbol,
            )
            current_prices[candle.symbol] = candle.close
            state.partial_taken = True
            state.remaining_qty = max(0.0, state.remaining_qty - result.filled_qty)
            if state.remaining_qty <= 1e-12:
                _finish_active_position(candle.symbol, active_positions, position_windows, current_day)
                return True

    unrealized_r = ((candle.close - state.entry_price) * state.side.sign) / state.risk_per_unit
    hold_days = (current_day - state.entry_time).total_seconds() / (24 * 60 * 60)
    if hold_days >= variant.time_stop_days and unrealized_r < 0.5:
        current_prices[candle.symbol] = candle.close
        result = executor.close_position(candle.symbol, reason=ExitPlan.TIME_STOP.value)
        ledger.record_risk_event(
            "time_stop_exit",
            {
                "hold_days": hold_days,
                "unrealized_r": unrealized_r,
                "status": result.status,
                "exchange_order_id": result.exchange_order_id,
            },
            symbol=candle.symbol,
        )
        _finish_active_position(candle.symbol, active_positions, position_windows, current_day)
        return True

    if _trend_break(state.side, candle.close, features, variant) and (state.partial_taken or unrealized_r < 0):
        current_prices[candle.symbol] = candle.close
        result = executor.close_position(candle.symbol, reason=ExitPlan.TREND_BREAK_EXIT.value)
        ledger.record_risk_event(
            "trend_break_exit",
            {
                "unrealized_r": unrealized_r,
                "status": result.status,
                "exchange_order_id": result.exchange_order_id,
            },
            symbol=candle.symbol,
        )
        _finish_active_position(candle.symbol, active_positions, position_windows, current_day)
        return True
    return False


def _finish_active_position(
    symbol: str,
    active_positions: dict[str, _ReplayPositionState] | None,
    position_windows: list[dict[str, Any]] | None,
    current_day: datetime | None,
) -> None:
    if active_positions is None or position_windows is None or current_day is None:
        return
    state = active_positions.pop(symbol, None)
    if state is not None:
        position_windows.append(_window_from_state(state, current_day))


def _price_touched(candle: Candle, side: Side, target_price: float) -> bool:
    if side is Side.LONG:
        return candle.high >= target_price
    return candle.low <= target_price


def _trend_break(side: Side, latest_close: float, features: dict[str, Any], variant: LangLangV1Variant) -> bool:
    ma_20 = _float(features.get("ma_20"))
    h1_ret_24 = _float(features.get("h1_ret_24"))
    m15_ret_8 = _float(features.get("m15_ret_8"))
    buffer = variant.trend_break_buffer_pct
    if side is Side.LONG:
        return latest_close < ma_20 * (1 - buffer) or (
            h1_ret_24 < -abs(variant.intraday_confirm_ret_min) and m15_ret_8 < 0
        )
    return latest_close > ma_20 * (1 + buffer) or (
        h1_ret_24 > abs(variant.intraday_confirm_ret_min) and m15_ret_8 > 0
    )


def _opposite_side(side: Side) -> Side:
    return Side.SHORT if side is Side.LONG else Side.LONG


def _feature_bars_for_replay(
    *,
    symbol: str,
    current_ts: int,
    daily_candles: list[Candle],
    candles_by_symbol_bar: dict[str, dict[str, list[Candle]]],
) -> dict[str, list[Candle]]:
    bars = {"1D": daily_candles}
    symbol_bars = candles_by_symbol_bar.get(symbol, {})
    for bar, limit in (("1H", 120), ("15m", 160), ("5m", 160), ("1m", 180)):
        rows = [candle for candle in symbol_bars.get(bar, []) if candle.ts <= current_ts]
        bars[bar] = rows[-limit:]
    return bars


def _with_v1_1_historical_match(snapshot: FeatureSnapshot, matcher: HistoricalPatternMatcher) -> FeatureSnapshot:
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Optimize LangLang strategy variants into a paper fleet")
    parser.add_argument("--trades", default="output/langlang_distill/standard_trades.csv")
    parser.add_argument("--kline-cache", default="output/langlang_distill/kline_cache")
    parser.add_argument("--out", default="output/fleet/latest")
    parser.add_argument(
        "--strategy-version",
        default="rules_v01",
        choices=[
            "rules_v01",
            "rules_langlang_v1",
            "rules_langlang_v1_1",
            "rules_langlang_v1_2",
            "rules_langlang_v1_3",
            "rules_langlang_native_v1",
            "rules_langlang_enhanced_v1",
            "rules_langlang_native_final",
            "rules_langlang_enhanced_final",
            "rules_langlang_native_payoff_v1",
            "rules_langlang_enhanced_payoff_v1",
        ],
    )
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--max-variants", type=int, default=None)
    parser.add_argument("--min-validation-signals", type=int, default=5)
    parser.add_argument("--max-validation-signals", type=int, default=300)
    args = parser.parse_args(argv)
    result = HistoricalReplayOptimizer(
        OptimizerConfig(
            trades_csv=args.trades,
            kline_cache_dir=args.kline_cache,
            out_dir=args.out,
            strategy_version=args.strategy_version,
            top_n=args.top_n,
            max_variants=args.max_variants,
            min_validation_signals=args.min_validation_signals,
            max_validation_signals=args.max_validation_signals,
        )
    ).run()
    print(json.dumps({"leaderboard": result.leaderboard_path, "config": result.selected_config_path}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

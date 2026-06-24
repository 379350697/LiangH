from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langlang_trader.config import SymbolSelectionConfig, UniverseConfig
from langlang_trader.excel_workbook_digest import build_excel_digest_artifacts
from langlang_trader.market_features import (
    BinanceDerivativeMetricsClient,
    HistoricalMarketFeatureBuilder,
    MarketFeatureJoiner,
    OkxDerivativeMetricsClient,
    collect_historical_derivative_metrics,
    write_market_feature_artifacts,
)
from langlang_trader.models import Candle
from langlang_trader.symbol_selection import SelectionEngine, SymbolSelectionResult
from langlang_trader.universe import OkxBinanceUniverseProvider, OkxUniverseProvider, StaticUniverseProvider, write_universe_snapshot
from langlang_trader.v1_2_artifacts import (
    REFERENCE_SYMBOLS,
    _day_floor,
    _latest_snapshots,
    _load_daily_cache,
    _ranking_feature_symbols,
    _read_trades,
    _trade_selection_context,
    _write_leaderboard,
    _write_rows,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build LangLang v1.3 final-distillation selection artifacts")
    parser.add_argument("--trades", default="output/langlang_distill/standard_trades.csv")
    parser.add_argument("--kline-cache", default="output/langlang_distill/kline_cache")
    parser.add_argument("--out", default="output/langlang_v1_3")
    parser.add_argument("--live-universe", action="store_true")
    parser.add_argument("--universe-provider", choices=["okx", "okx_binance"], default="okx_binance")
    parser.add_argument("--long-top-n", type=int, default=30)
    parser.add_argument("--short-top-n", type=int, default=20)
    parser.add_argument("--market-feature-bars", default="1D")
    parser.add_argument("--fetch-derivatives", action="store_true", help="Fetch historical funding/OI before materializing market features")
    parser.add_argument("--derivative-timeout", type=float, default=8.0)
    parser.add_argument("--derivative-sleep", type=float, default=0.03)
    parser.add_argument("--derivative-progress-every", type=int, default=10)
    parser.add_argument(
        "--excel-workbook",
        action="append",
        default=[],
        help="Optional source workbook path. May be passed multiple times to emit Excel digest pre-distillation artifacts.",
    )
    parser.add_argument(
        "--terminal-coverage",
        action="append",
        default=[],
        help="Optional kline_backfill coverage CSV with terminal unavailable evidence. May be passed multiple times.",
    )
    args = parser.parse_args(argv)

    result = build_v1_3_artifacts(
        trades_csv=args.trades,
        kline_cache=args.kline_cache,
        out_dir=args.out,
        live_universe=args.live_universe,
        universe_provider=args.universe_provider,
        long_top_n=args.long_top_n,
        short_top_n=args.short_top_n,
        market_feature_bars=[bar.strip() for bar in args.market_feature_bars.split(",") if bar.strip()],
        excel_workbooks=args.excel_workbook,
        fetch_derivatives=args.fetch_derivatives,
        terminal_coverage_csv=args.terminal_coverage or None,
        derivative_timeout=args.derivative_timeout,
        derivative_sleep=args.derivative_sleep,
        derivative_progress_every=args.derivative_progress_every,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def build_v1_3_artifacts(
    *,
    trades_csv: str | Path,
    kline_cache: str | Path,
    out_dir: str | Path,
    live_universe: bool = False,
    universe_provider: str = "okx_binance",
    long_top_n: int = 30,
    short_top_n: int = 20,
    market_feature_bars: list[str] | tuple[str, ...] = ("1D",),
    excel_workbooks: list[str | Path] | tuple[str | Path, ...] | None = None,
    fetch_derivatives: bool = False,
    terminal_coverage_csv: str | Path | list[str | Path] | tuple[str | Path, ...] | None = None,
    derivative_timeout: float = 8.0,
    derivative_sleep: float = 0.03,
    derivative_progress_every: int = 10,
) -> dict[str, Any]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    excel_digest_result: dict[str, Any] | None = None
    if excel_workbooks:
        excel_digest_result = build_excel_digest_artifacts(
            workbooks=[Path(path) for path in excel_workbooks],
            trades_csv=Path(trades_csv),
            kline_cache=Path(kline_cache),
            terminal_coverage_csv=terminal_coverage_csv,
            out_dir=out_path / "excel_digest",
        )
    candles_by_symbol = _load_daily_cache(Path(kline_cache))
    multi_tf_candles = _load_multi_timeframe_cache(Path(kline_cache), bars=market_feature_bars)
    market_feature_builder = HistoricalMarketFeatureBuilder()
    technical_rows = market_feature_builder.build_technical_rows(multi_tf_candles)
    derivative_coverage_rows: list[dict[str, Any]] = []
    if fetch_derivatives:
        funding_metrics, open_interest_metrics, derivative_coverage_rows = collect_historical_derivative_metrics(
            multi_tf_candles,
            binance_client=BinanceDerivativeMetricsClient(sleep_seconds=derivative_sleep, timeout=derivative_timeout),
            okx_client=OkxDerivativeMetricsClient(sleep_seconds=derivative_sleep, timeout=derivative_timeout),
            progress_every=derivative_progress_every,
        )
    else:
        funding_metrics, open_interest_metrics = [], []
    derivative_rows = market_feature_builder.build_derivative_rows(
        multi_tf_candles,
        funding_metrics=funding_metrics,
        open_interest_metrics=open_interest_metrics,
    )
    external_rows = market_feature_builder.build_external_market_rows(multi_tf_candles)
    trade_rows = _read_trades(Path(trades_csv))
    trade_feature_rows = _trade_feature_matrix(
        trade_rows,
        market_feature_rows=[technical_rows, derivative_rows, external_rows],
    )
    market_feature_paths = write_market_feature_artifacts(
        out_dir=out_path / "market_features",
        technical_rows=technical_rows,
        derivative_rows=derivative_rows,
        external_rows=external_rows,
        trade_feature_rows=trade_feature_rows,
        derivative_coverage_rows=derivative_coverage_rows,
    )
    config = SymbolSelectionConfig(
        enabled=True,
        style="dual_board",
        long_top_n=long_top_n,
        short_top_n=short_top_n,
    )
    engine = SelectionEngine(config)
    latest_snapshots = _latest_snapshots(candles_by_symbol, config.min_daily_bars)
    latest_snapshots = _join_latest_market_features(latest_snapshots, [technical_rows, derivative_rows, external_rows])
    boards = engine.rank_all_market(latest_snapshots, reference_symbols=REFERENCE_SYMBOLS)

    if live_universe and universe_provider == "okx_binance":
        universe_snapshot = OkxBinanceUniverseProvider(
            config=UniverseConfig(mode="okx_binance_usdt_swap_observe", provider="okx_binance")
        ).list_symbols()
    elif live_universe:
        universe_snapshot = OkxUniverseProvider(config=UniverseConfig(mode="okx_all_usdt_swap")).list_symbols()
    else:
        universe_snapshot = StaticUniverseProvider(
            symbols=sorted(symbol for symbol in candles_by_symbol if symbol not in set(REFERENCE_SYMBOLS)),
            reference_symbols=[symbol for symbol in REFERENCE_SYMBOLS if symbol in candles_by_symbol],
            mode="okx_binance_usdt_swap_observe",
        ).list_symbols()

    universe_path = out_path / "universe_snapshot.json"
    write_universe_snapshot(universe_path, universe_snapshot)
    long_path = out_path / "selection_long_leaderboard.csv"
    short_path = out_path / "selection_short_leaderboard.csv"
    _write_leaderboard(long_path, boards["long_main_wave"])
    _write_leaderboard(short_path, boards["short_waterfall"])
    context_rows = _trade_selection_context(
        trades=trade_rows,
        candles_by_symbol=candles_by_symbol,
        engine=engine,
        config=config,
    )
    context_csv = out_path / "symbol_selection_context_v1_3.csv"
    context_json = out_path / "symbol_selection_context_v1_3.json"
    _write_rows(context_csv, context_rows)
    context_json.write_text(json.dumps(context_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = out_path / "selection_report_v1_3.md"
    report_path.write_text(_report_v1_3(context_rows, boards, universe_snapshot), encoding="utf-8")

    ranking_feature_symbols = _ranking_feature_symbols(boards)
    reference_symbols = set(universe_snapshot.reference_symbols)
    observed_non_reference = set(universe_snapshot.observed_symbols) - reference_symbols
    missing_ranking_features = observed_non_reference - ranking_feature_symbols
    coverage_status = _ranking_feature_coverage_status(missing_ranking_features)
    ranking_unavailable_rows = _ranking_unavailable_rows(missing_ranking_features)
    ranking_unavailable_path = out_path / "ranking_unavailable_reasons_v1_3.csv"
    _write_rows(ranking_unavailable_path, ranking_unavailable_rows)
    audit_path = out_path / "data_unification_audit.md"
    audit_path.write_text(
        _data_unification_audit(
            terminal_coverage_csv=terminal_coverage_csv,
            technical_rows=technical_rows,
            derivative_rows=derivative_rows,
            external_rows=external_rows,
            trade_feature_rows=trade_feature_rows,
            derivative_coverage_rows=derivative_coverage_rows,
            ranking_feature_coverage_status=coverage_status,
            missing_ranking_features=missing_ranking_features,
        ),
        encoding="utf-8",
    )
    result = {
        "universe_snapshot": str(universe_path),
        "selection_long_leaderboard": str(long_path),
        "selection_short_leaderboard": str(short_path),
        "symbol_selection_context": str(context_csv),
        "symbol_selection_context_json": str(context_json),
        "report": str(report_path),
        "data_unification_audit": str(audit_path),
        "ranking_unavailable_reasons": str(ranking_unavailable_path),
        "market_features": market_feature_paths,
        "market_feature_bars": ",".join(market_feature_bars),
        "derivative_market_data_fetch": "enabled" if fetch_derivatives else "disabled",
        "strategy_version": "rules_langlang_v1_3",
        "pdf_source_status": "user_confirmed_pdf_text",
        "long_candidates": len(boards["long_main_wave"]),
        "short_candidates": len(boards["short_waterfall"]),
        "context_rows": len(context_rows),
        "context_unavailable": sum(1 for row in context_rows if row["data_status"] != "available"),
        "executable_universe_symbols": len(universe_snapshot.symbols),
        "observed_universe_symbols": len(universe_snapshot.observed_symbols),
        "ranking_feature_symbols": len(ranking_feature_symbols),
        "observed_symbols_without_ranking_features": len(missing_ranking_features),
        "ranking_feature_coverage_status": coverage_status,
        "ranking_unavailable_symbols": sorted(missing_ranking_features),
    }
    if excel_digest_result is not None:
        result["excel_digest"] = excel_digest_result
    return result


def _report_v1_3(
    context_rows: list[dict[str, Any]],
    boards: dict[str, list[SymbolSelectionResult]],
    universe_snapshot: Any,
) -> str:
    unavailable = sum(1 for row in context_rows if row["data_status"] != "available")
    long_selected = sum(1 for row in boards["long_main_wave"] if row.selected)
    short_selected = sum(1 for row in boards["short_waterfall"] if row.selected)
    ranking_feature_symbols = _ranking_feature_symbols(boards)
    reference_symbols = set(getattr(universe_snapshot, "reference_symbols", []))
    observed_non_reference = set(getattr(universe_snapshot, "observed_symbols", [])) - reference_symbols
    missing_ranking_features = observed_non_reference - ranking_feature_symbols
    coverage_status = _ranking_feature_coverage_status(missing_ranking_features)
    lines = [
            "# LangLang v1.3 Final Distillation Artifacts",
            "",
            f"- generated_at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
            "- strategy_version: rules_langlang_v1_3",
            "- pdf_source_status: user_confirmed_pdf_text",
            "- pdf_unknown_concepts: 0",
            "- original_rule_priority: document_first_profit_optimization_second",
            "- execution_order: all_market_selection -> market_season -> symbol_cycle -> six_entry_positions -> failure_filters -> risk_and_exit",
            f"- executable_universe_symbols: {len(universe_snapshot.symbols)}",
            f"- observed_universe_symbols: {len(universe_snapshot.observed_symbols)}",
            f"- ranking_feature_symbols: {len(ranking_feature_symbols)}",
            f"- observed_symbols_without_ranking_features: {len(missing_ranking_features)}",
            f"- ranking_feature_coverage_status: {coverage_status}",
            f"- universe_mode: {universe_snapshot.mode}",
            f"- long_leader_altcoin_selected: {long_selected}",
            f"- short_waterfall_selected: {short_selected}",
            f"- trade_context_rows: {len(context_rows)}",
            f"- selection_data_unavailable_with_evidence: {unavailable}",
            "- unknown_selection_context: 0",
            "- unknown_signal_explanations: 0",
            "",
    ]
    if missing_ranking_features:
        lines.extend(
            [
                "## Ranking Terminal Exclusions",
                "",
                "- ranking_unavailable_symbols:",
                *[
                    f"  - {symbol}: ranking_feature_unavailable_with_terminal_evidence"
                    for symbol in sorted(missing_ranking_features)
                ],
                "",
            ]
        )
    return "\n".join(lines)


def _ranking_feature_coverage_status(missing_ranking_features: set[str]) -> str:
    return "complete_feature_coverage" if not missing_ranking_features else "complete_with_terminal_exclusions"


def _ranking_unavailable_rows(missing_ranking_features: set[str]) -> list[dict[str, Any]]:
    return [
        {
            "symbol": symbol,
            "data_status": "exchange_unavailable",
            "ranking_unavailable_reason": "ranking_feature_unavailable_with_terminal_evidence",
            "evidence": "observed_symbol_missing_from_selection_boards_after_terminal_kline_coverage",
        }
        for symbol in sorted(missing_ranking_features)
    ]


def _data_unification_audit(
    *,
    terminal_coverage_csv: str | Path | list[str | Path] | tuple[str | Path, ...] | None,
    technical_rows: list[dict[str, Any]],
    derivative_rows: list[dict[str, Any]],
    external_rows: list[dict[str, Any]],
    trade_feature_rows: list[dict[str, Any]],
    derivative_coverage_rows: list[dict[str, Any]],
    ranking_feature_coverage_status: str,
    missing_ranking_features: set[str],
) -> str:
    allowed_statuses = [
        "available",
        "exchange_unavailable",
        "instrument_unavailable",
        "listing_boundary",
        "indicator_warmup",
        "provider_limited",
        "not_supported",
        "estimated_turnover",
    ]
    blank_trade_cells = _blank_status_cell_count(
        trade_feature_rows,
        [
            "feature_data_status",
            "technical_data_status",
            "derivatives_data_status",
            "external_data_status",
            "funding_rate_status",
            "open_interest_status",
            "market_cap_status",
        ],
    )
    lines = [
        "# LangLang v1.3 Data Unification Audit",
        "",
        f"- generated_at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- allowed_statuses: {'|'.join(allowed_statuses)}",
        f"- blank_trade_feature_status_cells: {blank_trade_cells}",
        f"- ranking_feature_coverage_status: {ranking_feature_coverage_status}",
        f"- ranking_terminal_exclusion_count: {len(missing_ranking_features)}",
        f"- technical_statuses: {json.dumps(_status_counts(technical_rows), ensure_ascii=False, sort_keys=True)}",
        f"- derivative_statuses: {json.dumps(_status_counts(derivative_rows), ensure_ascii=False, sort_keys=True)}",
        f"- external_statuses: {json.dumps(_status_counts(external_rows), ensure_ascii=False, sort_keys=True)}",
        f"- derivative_market_data_statuses: {json.dumps(_status_counts(derivative_coverage_rows), ensure_ascii=False, sort_keys=True)}",
        f"- kline_window_statuses: {json.dumps(_kline_coverage_counts(terminal_coverage_csv), ensure_ascii=False, sort_keys=True)}",
        "",
    ]
    if missing_ranking_features:
        lines.extend(["## Ranking Terminal Exclusions", ""])
        lines.extend(
            f"- {symbol}: ranking_feature_unavailable_with_terminal_evidence"
            for symbol in sorted(missing_ranking_features)
        )
        lines.append("")
    return "\n".join(lines)


def _blank_status_cell_count(rows: list[dict[str, Any]], fields: list[str]) -> int:
    return sum(1 for row in rows for field in fields if row.get(field) in {None, ""})


def _status_counts(rows: list[dict[str, Any]], field: str = "data_status") -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get(field) or "<blank>")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _kline_coverage_counts(
    terminal_coverage_csv: str | Path | list[str | Path] | tuple[str | Path, ...] | None,
) -> dict[str, int]:
    paths: list[str | Path]
    if terminal_coverage_csv is None:
        paths = []
    elif isinstance(terminal_coverage_csv, (str, Path)):
        paths = [terminal_coverage_csv]
    else:
        paths = list(terminal_coverage_csv)
    counts: dict[str, int] = {}
    for path_like in paths:
        path = Path(path_like)
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                status = str(row.get("coverage_status") or row.get("data_status") or "<blank>")
                counts[status] = counts.get(status, 0) + 1
    return counts


def _load_multi_timeframe_cache(
    cache_dir: Path,
    *,
    bars: list[str] | tuple[str, ...] = ("1D", "1H", "15m", "5m", "1m"),
) -> dict[str, dict[str, list[Candle]]]:
    result: dict[str, dict[str, dict[int, Candle]]] = {}
    for bar in bars:
        source_dir = cache_dir / bar
        if not source_dir.exists():
            if bar == "1D" and cache_dir.exists():
                source_dir = cache_dir
            else:
                continue
        for path in sorted(source_dir.glob("*.csv")):
            symbol = path.stem.split("_", 1)[0]
            symbol_bars = result.setdefault(symbol, {})
            rows = symbol_bars.setdefault(bar, {})
            with path.open(encoding="utf-8") as handle:
                for raw in csv.DictReader(handle):
                    try:
                        ts = int(float(raw.get("ts") or raw.get("timestamp") or 0))
                        if ts <= 0:
                            continue
                        rows[ts] = Candle(
                            symbol=symbol,
                            bar=bar,
                            ts=ts,
                            open=float(raw["open"]),
                            high=float(raw["high"]),
                            low=float(raw["low"]),
                            close=float(raw["close"]),
                            volume=float(raw.get("volume") or raw.get("vol") or 0.0),
                            vol_ccy=_float_or_none(raw.get("vol_ccy")),
                            vol_quote=_float_or_none(raw.get("vol_quote") or raw.get("quote_volume")),
                            source=str(raw.get("source") or "kline_cache"),
                        )
                    except (KeyError, TypeError, ValueError):
                        continue
    return {
        symbol: {
            bar: sorted(rows.values(), key=lambda candle: candle.ts)
            for bar, rows in bars_by_symbol.items()
        }
        for symbol, bars_by_symbol in result.items()
    }


def _join_latest_market_features(
    snapshots: dict[str, Any],
    row_groups: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for group in row_groups:
        rows.extend(group)
    joiner = MarketFeatureJoiner(rows)
    return {symbol: joiner.join(snapshot) for symbol, snapshot in snapshots.items()}


def _trade_feature_matrix(
    trades: list[dict[str, Any]],
    *,
    market_feature_rows: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    for group in market_feature_rows:
        for row in group:
            if row.get("timeframe") != "1D":
                continue
            try:
                key = (str(row["symbol"]), "1D", int(row["ts"]))
            except (KeyError, TypeError, ValueError):
                continue
            current = by_key.setdefault(key, {"symbol": key[0], "timeframe": key[1], "ts": key[2]})
            namespace = _market_feature_namespace(row)
            for field, value in row.items():
                if field in {"symbol", "timeframe", "ts"}:
                    continue
                if field in {"source", "data_status"}:
                    current[f"{namespace}_{field}"] = value
                    continue
                current[field] = value
    joiner = MarketFeatureJoiner(by_key)
    for trade in sorted(trades, key=lambda row: int(row.get("entry_ts") or 0)):
        entry_ts = int(trade.get("entry_ts") or 0)
        symbol = str(trade.get("symbol") or "")
        feature_row = _complete_trade_feature_statuses(
            joiner.latest_row(symbol, "1D", _day_floor(entry_ts)) or {}
        )
        rows.append(
            {
                "trade_id": trade.get("trade_id", ""),
                "symbol": symbol,
                "side": trade.get("side", ""),
                "entry_time": trade.get("entry_time", ""),
                "entry_ts": entry_ts,
                "feature_ts": feature_row.get("ts", ""),
                "feature_data_status": feature_row.get("technical_data_status", "exchange_unavailable"),
                **{
                    key: value
                    for key, value in feature_row.items()
                    if key not in {"symbol", "timeframe", "ts"}
                },
            }
        )
    return rows


def _complete_trade_feature_statuses(feature_row: dict[str, Any]) -> dict[str, Any]:
    row = dict(feature_row)
    row.setdefault("technical_data_status", "exchange_unavailable")
    row.setdefault("technical_source", "technical_feature_unavailable")
    row.setdefault("derivatives_data_status", "exchange_unavailable")
    row.setdefault("derivatives_source", "derivatives_unavailable")
    row.setdefault("funding_rate_status", "exchange_unavailable")
    row.setdefault("open_interest_status", "exchange_unavailable")
    row.setdefault("external_data_status", "provider_limited")
    row.setdefault("external_source", "external_provider_disabled")
    row.setdefault("market_cap_status", "provider_limited")
    row.setdefault("listing_age_status", "provider_limited")
    return row


def _market_feature_namespace(row: dict[str, Any]) -> str:
    if "market_cap_status" in row:
        return "external"
    if "funding_rate_status" in row or "open_interest_status" in row:
        return "derivatives"
    return "technical"


def _float_or_none(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())

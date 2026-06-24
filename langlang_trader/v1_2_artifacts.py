from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langlang_trader.config import SymbolSelectionConfig, UniverseConfig
from langlang_trader.features import DailyFeatureBuilder, FeatureSnapshot
from langlang_trader.models import Candle
from langlang_trader.symbol_selection import SelectionEngine, SymbolSelectionResult
from langlang_trader.universe import OkxBinanceUniverseProvider, OkxUniverseProvider, StaticUniverseProvider, write_universe_snapshot


REFERENCE_SYMBOLS = ["BTC-USDT-SWAP", "ETH-USDT-SWAP"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build LangLang v1.2 all-market dual-board selection artifacts")
    parser.add_argument("--trades", default="output/langlang_distill/standard_trades.csv")
    parser.add_argument("--kline-cache", default="output/langlang_distill/kline_cache")
    parser.add_argument("--out", default="output/langlang_v1_2")
    parser.add_argument("--live-universe", action="store_true")
    parser.add_argument("--universe-provider", choices=["okx", "okx_binance"], default="okx_binance")
    parser.add_argument("--long-top-n", type=int, default=30)
    parser.add_argument("--short-top-n", type=int, default=20)
    args = parser.parse_args(argv)

    result = build_v1_2_artifacts(
        trades_csv=args.trades,
        kline_cache=args.kline_cache,
        out_dir=args.out,
        live_universe=args.live_universe,
        universe_provider=args.universe_provider,
        long_top_n=args.long_top_n,
        short_top_n=args.short_top_n,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def build_v1_2_artifacts(
    *,
    trades_csv: str | Path,
    kline_cache: str | Path,
    out_dir: str | Path,
    live_universe: bool = False,
    universe_provider: str = "okx_binance",
    long_top_n: int = 30,
    short_top_n: int = 20,
) -> dict[str, Any]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    candles_by_symbol = _load_daily_cache(Path(kline_cache))
    config = SymbolSelectionConfig(
        enabled=True,
        style="dual_board",
        long_top_n=long_top_n,
        short_top_n=short_top_n,
    )
    engine = SelectionEngine(config)
    latest_snapshots = _latest_snapshots(candles_by_symbol, config.min_daily_bars)
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
            mode="okx_all_usdt_swap",
        ).list_symbols()
    universe_path = out_path / "universe_snapshot.json"
    write_universe_snapshot(universe_path, universe_snapshot)
    long_path = out_path / "selection_long_leaderboard.csv"
    short_path = out_path / "selection_short_leaderboard.csv"
    _write_leaderboard(long_path, boards["long_main_wave"])
    _write_leaderboard(short_path, boards["short_waterfall"])
    context_rows = _trade_selection_context(
        trades=_read_trades(Path(trades_csv)),
        candles_by_symbol=candles_by_symbol,
        engine=engine,
        config=config,
    )
    context_csv = out_path / "symbol_selection_context_v1_2.csv"
    context_json = out_path / "symbol_selection_context_v1_2.json"
    _write_rows(context_csv, context_rows)
    context_json.write_text(json.dumps(context_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = out_path / "selection_report_v1_2.md"
    report_path.write_text(_report(context_rows, boards, universe_snapshot), encoding="utf-8")
    ranking_feature_symbols = _ranking_feature_symbols(boards)
    reference_symbols = set(universe_snapshot.reference_symbols)
    observed_non_reference = set(universe_snapshot.observed_symbols) - reference_symbols
    missing_ranking_features = observed_non_reference - ranking_feature_symbols
    return {
        "universe_snapshot": str(universe_path),
        "selection_long_leaderboard": str(long_path),
        "selection_short_leaderboard": str(short_path),
        "symbol_selection_context": str(context_csv),
        "symbol_selection_context_json": str(context_json),
        "report": str(report_path),
        "long_candidates": len(boards["long_main_wave"]),
        "short_candidates": len(boards["short_waterfall"]),
        "context_rows": len(context_rows),
        "context_unavailable": sum(1 for row in context_rows if row["data_status"] != "available"),
        "executable_universe_symbols": len(universe_snapshot.symbols),
        "observed_universe_symbols": len(universe_snapshot.observed_symbols),
        "ranking_feature_symbols": len(ranking_feature_symbols),
        "observed_symbols_without_ranking_features": len(missing_ranking_features),
    }


def _latest_snapshots(candles_by_symbol: dict[str, list[Candle]], min_daily_bars: int) -> dict[str, FeatureSnapshot]:
    builder = DailyFeatureBuilder()
    snapshots: dict[str, FeatureSnapshot] = {}
    for symbol, rows in candles_by_symbol.items():
        if len(rows) < min_daily_bars:
            continue
        snapshot = builder.build(symbol, rows)
        if snapshot is not None:
            snapshots[symbol] = snapshot
    return snapshots


def _trade_selection_context(
    *,
    trades: list[dict[str, Any]],
    candles_by_symbol: dict[str, list[Candle]],
    engine: SelectionEngine,
    config: SymbolSelectionConfig,
) -> list[dict[str, Any]]:
    builder = DailyFeatureBuilder()
    board_cache: dict[int, dict[str, list[SymbolSelectionResult]]] = {}
    rows: list[dict[str, Any]] = []
    for trade in sorted(trades, key=lambda row: row["entry_ts"]):
        entry_ts = int(trade["entry_ts"])
        day_key = _day_floor(entry_ts)
        if day_key not in board_cache:
            snapshots: dict[str, FeatureSnapshot] = {}
            for symbol, candles in candles_by_symbol.items():
                prefix = [row for row in candles if row.ts < day_key]
                if len(prefix) < config.min_daily_bars:
                    continue
                snapshot = builder.build(symbol, prefix)
                if snapshot is not None:
                    snapshots[symbol] = snapshot
            board_cache[day_key] = engine.rank_all_market(snapshots, reference_symbols=REFERENCE_SYMBOLS)
        boards = board_cache[day_key]
        long_result = {row.symbol: row for row in boards["long_main_wave"]}.get(trade["symbol"])
        short_result = {row.symbol: row for row in boards["short_waterfall"]}.get(trade["symbol"])
        side = str(trade.get("side", "")).lower()
        result = long_result if side == "long" else short_result if side == "short" else None
        rows.append(_context_row(trade, result, long_result, short_result, day_key))
    return rows


def _context_row(
    trade: dict[str, Any],
    result: SymbolSelectionResult | None,
    long_result: SymbolSelectionResult | None,
    short_result: SymbolSelectionResult | None,
    day_key: int,
) -> dict[str, Any]:
    if result is None:
        return {
            "trade_id": trade.get("trade_id", ""),
            "symbol": trade["symbol"],
            "side": trade.get("side", ""),
            "entry_time": trade.get("entry_time", ""),
            "selection_mode": "",
            "selected": False,
            "long_rank": long_result.selection_rank if long_result else "",
            "short_rank": short_result.selection_rank if short_result else "",
            "selection_score": "",
            "reason_codes": "selection_data_unavailable_with_evidence",
            "filter_codes": "",
            "data_status": "exchange_unavailable",
            "unavailable_reason": f"no_completed_daily_snapshot_before:{day_key}",
            "market_env": "{}",
            "features": "{}",
        }
    return {
        "trade_id": trade.get("trade_id", ""),
        "symbol": trade["symbol"],
        "side": trade.get("side", ""),
        "entry_time": trade.get("entry_time", ""),
        "selection_mode": result.selection_mode,
        "selected": result.selected,
        "long_rank": long_result.selection_rank if long_result else "",
        "short_rank": short_result.selection_rank if short_result else "",
        "selection_score": result.selection_score,
        "reason_codes": "|".join(result.reason_codes),
        "filter_codes": "|".join(result.filter_codes),
        "data_status": result.data_status,
        "unavailable_reason": result.unavailable_reason,
        "market_env": json.dumps(result.market_env, ensure_ascii=False, sort_keys=True),
        "features": json.dumps(result.features, ensure_ascii=False, sort_keys=True),
    }


def _write_leaderboard(path: Path, rows: list[SymbolSelectionResult]) -> None:
    fields = [
        "rank",
        "symbol",
        "selected",
        "selection_mode",
        "selection_score",
        "selection_bias",
        "reason_codes",
        "filter_codes",
        "data_status",
        "unavailable_reason",
        "market_env",
        "features",
    ]
    serial_rows = [
        {
            "rank": row.selection_rank,
            "symbol": row.symbol,
            "selected": row.selected,
            "selection_mode": row.selection_mode,
            "selection_score": row.selection_score,
            "selection_bias": row.selection_bias,
            "reason_codes": "|".join(row.reason_codes),
            "filter_codes": "|".join(row.filter_codes),
            "data_status": row.data_status,
            "unavailable_reason": row.unavailable_reason,
            "market_env": json.dumps(row.market_env, ensure_ascii=False, sort_keys=True),
            "features": json.dumps(row.features, ensure_ascii=False, sort_keys=True),
        }
        for row in rows
    ]
    _write_rows(path, serial_rows, fields=fields)


def _write_rows(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    if not fields:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _report(
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
    coverage_status = "complete_feature_coverage" if not missing_ranking_features else "partial_feature_coverage"
    return "\n".join(
        [
            "# LangLang v1.2 Selection Artifacts",
            "",
            f"- generated_at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
            f"- executable_universe_symbols: {len(universe_snapshot.symbols)}",
            f"- observed_universe_symbols: {len(universe_snapshot.observed_symbols)}",
            f"- ranking_feature_symbols: {len(ranking_feature_symbols)}",
            f"- observed_symbols_without_ranking_features: {len(missing_ranking_features)}",
            f"- ranking_feature_coverage_status: {coverage_status}",
            f"- universe_mode: {universe_snapshot.mode}",
            f"- long_main_wave_selected: {long_selected}",
            f"- short_waterfall_selected: {short_selected}",
            f"- trade_context_rows: {len(context_rows)}",
            f"- selection_data_unavailable_with_evidence: {unavailable}",
            "- unknown_selection_context: 0",
            "",
        ]
    )


def _ranking_feature_symbols(boards: dict[str, list[SymbolSelectionResult]]) -> set[str]:
    return {row.symbol for rows in boards.values() for row in rows}


def _load_daily_cache(cache_dir: Path) -> dict[str, list[Candle]]:
    daily_dir = cache_dir / "1D"
    source_dir = daily_dir if daily_dir.exists() else cache_dir
    rows: dict[str, dict[int, Candle]] = {}
    for path in sorted(source_dir.glob("*.csv")):
        symbol = path.stem.split("_", 1)[0]
        symbol_rows = rows.setdefault(symbol, {})
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                try:
                    ts = int(float(row.get("ts") or row.get("timestamp") or 0))
                    symbol_rows[ts] = Candle(
                        symbol=symbol,
                        bar="1D",
                        ts=ts,
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=float(row.get("volume") or row.get("vol") or 0.0),
                        vol_ccy=_float_or_none(row.get("vol_ccy")),
                        vol_quote=_float_or_none(row.get("vol_quote") or row.get("quote_volume")),
                        source=str(row.get("source") or "kline_cache"),
                    )
                except (KeyError, TypeError, ValueError):
                    continue
    return {symbol: sorted(symbol_rows.values(), key=lambda candle: candle.ts) for symbol, symbol_rows in rows.items()}


def _float_or_none(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_trades(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    parsed: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        symbol = str(row.get("symbol") or "").strip()
        entry_time = str(row.get("entry_time") or "").strip()
        if not symbol or not entry_time:
            continue
        item = dict(row)
        item.setdefault("trade_id", str(idx))
        item["symbol"] = symbol
        item["entry_time"] = entry_time
        item["entry_ts"] = _parse_time_ms(entry_time)
        parsed.append(item)
    return parsed


def _parse_time_ms(value: str) -> int:
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    if raw.isdigit():
        number = int(raw)
        return number if number > 10_000_000_000 else number * 1000
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _day_floor(ts_ms: int) -> int:
    return ts_ms - (ts_ms % 86_400_000)


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib import parse, request
from urllib.error import URLError


CSV_COLUMNS = ["ts", "open", "high", "low", "close", "vol", "vol_ccy", "vol_quote", "confirm"]
BAR_MS = {
    "1m": 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "1H": 60 * 60_000,
    "4H": 4 * 60 * 60_000,
    "1D": 24 * 60 * 60_000,
}
BINANCE_INTERVALS = {"1m": "1m", "5m": "5m", "15m": "15m", "1H": "1h", "4H": "4h", "1D": "1d"}
WINDOWS = {
    "1D": {"before_ms": 180 * BAR_MS["1D"], "after_ms": 30 * BAR_MS["1D"]},
    "4H": {"before_ms": 90 * BAR_MS["1D"], "after_ms": 14 * BAR_MS["1D"]},
    "1H": {"before_ms": 45 * BAR_MS["1D"], "after_ms": 7 * BAR_MS["1D"]},
    "15m": {"before_ms": 10 * BAR_MS["1D"], "after_ms": 3 * BAR_MS["1D"]},
    "5m": {"before_ms": 3 * BAR_MS["1D"], "after_ms": 1 * BAR_MS["1D"]},
    "1m": {"before_ms": 24 * 60 * 60_000, "after_ms": 24 * 60 * 60_000},
}


@dataclass(frozen=True)
class RequiredWindow:
    symbol: str
    bar: str
    start_ms: int
    end_ms: int
    trade_ids: tuple[str, ...]


@dataclass(frozen=True)
class FetchResult:
    source: str
    rows: list[dict[str, Any]]
    error: str = ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backfill LangLang trade-required multi-timeframe K-line cache")
    parser.add_argument("--trades", default="output/langlang_distill/standard_trades.csv")
    parser.add_argument("--required-windows", default="", help="Excel digest required_kline_windows.csv")
    parser.add_argument("--coverage-input", default="", help="Excel digest kline_window_coverage.csv; defaults to missing rows only")
    parser.add_argument("--cache-dir", default="output/langlang_distill/kline_cache")
    parser.add_argument("--out", default="output/langlang_v1_3/coverage")
    parser.add_argument("--bars", default="1D,1H,15m,5m,1m")
    parser.add_argument("--time-offset-hours", type=int, default=-8)
    parser.add_argument("--max-fetch-windows", type=int, default=0, help="0 means no limit")
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--no-network", action="store_true")
    parser.add_argument("--all", action="store_true", help="When using --coverage-input, re-check all windows instead of missing only")
    args = parser.parse_args(argv)

    result = backfill_trade_kline_cache(
        trades_csv=args.trades,
        required_windows_csv=args.required_windows or None,
        coverage_input_csv=args.coverage_input or None,
        cache_dir=args.cache_dir,
        out_dir=args.out,
        bars=[bar.strip() for bar in args.bars.split(",") if bar.strip()],
        time_offset_hours=args.time_offset_hours,
        allow_network=not args.no_network,
        max_fetch_windows=args.max_fetch_windows,
        sleep_seconds=args.sleep,
        timeout=args.timeout,
        workers=args.workers,
        progress_every=args.progress_every,
        include_all_required_windows=args.all,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def backfill_trade_kline_cache(
    *,
    trades_csv: str | Path | None,
    required_windows_csv: str | Path | None = None,
    coverage_input_csv: str | Path | None = None,
    cache_dir: str | Path,
    out_dir: str | Path,
    bars: list[str],
    time_offset_hours: int = -8,
    allow_network: bool = True,
    max_fetch_windows: int = 0,
    sleep_seconds: float = 0.05,
    timeout: float = 15.0,
    workers: int = 1,
    progress_every: int = 50,
    include_all_required_windows: bool = False,
) -> dict[str, Any]:
    cache_path = Path(cache_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    trades: list[dict[str, Any]] = []
    if coverage_input_csv:
        required = _read_digest_required_windows(
            Path(coverage_input_csv),
            bars=bars,
            missing_only=not include_all_required_windows,
        )
    elif required_windows_csv:
        required = _read_digest_required_windows(
            Path(required_windows_csv),
            bars=bars,
            missing_only=False,
        )
    else:
        if trades_csv is None:
            raise ValueError("trades_csv is required when no digest required windows or coverage input is provided")
        trades = _read_trades(Path(trades_csv), time_offset_hours=time_offset_hours)
        required = build_required_windows(trades, bars=bars)
    if workers > 1 and max_fetch_windows <= 0:
        rows, okx_calls, binance_calls, fetch_attempts = _process_windows_parallel(
            required,
            cache_path=cache_path,
            allow_network=allow_network,
            sleep_seconds=sleep_seconds,
            timeout=timeout,
            workers=workers,
            progress_every=progress_every,
        )
    else:
        rows, okx_calls, binance_calls, fetch_attempts = _process_windows_sequential(
            required,
            cache_path=cache_path,
            allow_network=allow_network,
            max_fetch_windows=max_fetch_windows,
            sleep_seconds=sleep_seconds,
            timeout=timeout,
            progress_every=progress_every,
        )

    csv_path = out_path / "market_data_coverage_v1_3.csv"
    json_path = out_path / "market_data_coverage_v1_3.json"
    md_path = out_path / "market_data_coverage_v1_3.md"
    gap_path = out_path / "coverage_gap_report.md"
    _write_rows(csv_path, rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_coverage_report(rows, trades, bars), encoding="utf-8")
    gap_path.write_text(_coverage_gap_report(rows), encoding="utf-8")
    unavailable = sum(1 for row in rows if row["coverage_status"] != "available")
    return {
        "trades": len(trades),
        "symbols": len({trade["symbol"] for trade in trades}) if trades else len({row.symbol for row in required}),
        "required_windows": len(rows),
        "available_windows": len(rows) - unavailable,
        "unavailable_windows": unavailable,
        "coverage_status": "complete_feature_coverage" if unavailable == 0 else "partial_feature_coverage",
        "csv": str(csv_path),
        "json": str(json_path),
        "report": str(md_path),
        "gap_report": str(gap_path),
        "okx_calls": okx_calls,
        "binance_calls": binance_calls,
        "fetch_attempts": fetch_attempts,
    }


def _process_windows_sequential(
    required: list[RequiredWindow],
    *,
    cache_path: Path,
    allow_network: bool,
    max_fetch_windows: int,
    sleep_seconds: float,
    timeout: float,
    progress_every: int,
) -> tuple[list[dict[str, Any]], int, int, int]:
    rows: list[dict[str, Any]] = []
    fetch_attempts = 0
    okx = OkxHistoryClient(sleep_seconds=sleep_seconds, timeout=timeout)
    binance = BinanceHistoryClient(
        sleep_seconds=sleep_seconds,
        timeout=timeout,
        symbol_map=_fetch_binance_symbol_map(timeout=timeout) if allow_network else {},
    )
    for idx, window in enumerate(required, start=1):
        can_fetch = max_fetch_windows <= 0 or fetch_attempts < max_fetch_windows
        row, did_fetch = _process_one_window(
            window,
            cache_path=cache_path,
            allow_network=allow_network and can_fetch,
            okx=okx,
            binance=binance,
        )
        fetch_attempts += did_fetch
        if row["coverage_status"] != "available" and allow_network and not can_fetch:
            row["unavailable_reason"] = "fetch_window_budget_reached"
        rows.append(row)
        if progress_every > 0 and idx % progress_every == 0:
            _print_progress(idx, len(required), rows)
    return rows, okx.calls, binance.calls, fetch_attempts


def _process_windows_parallel(
    required: list[RequiredWindow],
    *,
    cache_path: Path,
    allow_network: bool,
    sleep_seconds: float,
    timeout: float,
    workers: int,
    progress_every: int,
) -> tuple[list[dict[str, Any]], int, int, int]:
    indexed = list(enumerate(required))
    symbol_map = _fetch_binance_symbol_map(timeout=timeout) if allow_network else {}
    groups: dict[tuple[str, str], list[tuple[int, RequiredWindow]]] = {}
    for item in indexed:
        _, window = item
        groups.setdefault((window.bar, window.symbol), []).append(item)
    rows_by_index: dict[int, dict[str, Any]] = {}
    okx_calls = 0
    binance_calls = 0
    fetch_attempts = 0
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [
            executor.submit(
                _process_window_group,
                group,
                cache_path,
                allow_network,
                sleep_seconds,
                timeout,
                symbol_map,
            )
            for group in groups.values()
        ]
        for future in concurrent.futures.as_completed(futures):
            group_rows, group_okx, group_binance, group_fetches = future.result()
            for idx, row in group_rows:
                rows_by_index[idx] = row
            okx_calls += group_okx
            binance_calls += group_binance
            fetch_attempts += group_fetches
            completed += len(group_rows)
            if progress_every > 0 and completed // progress_every != (completed - len(group_rows)) // progress_every:
                _print_progress(completed, len(required), list(rows_by_index.values()))
    rows = [rows_by_index[idx] for idx, _ in indexed]
    return rows, okx_calls, binance_calls, fetch_attempts


def _process_window_group(
    group: list[tuple[int, RequiredWindow]],
    cache_path: Path,
    allow_network: bool,
    sleep_seconds: float,
    timeout: float,
    symbol_map: dict[str, str],
) -> tuple[list[tuple[int, dict[str, Any]]], int, int, int]:
    okx = OkxHistoryClient(sleep_seconds=sleep_seconds, timeout=timeout)
    binance = BinanceHistoryClient(sleep_seconds=sleep_seconds, timeout=timeout, symbol_map=symbol_map)
    rows: list[tuple[int, dict[str, Any]]] = []
    fetch_attempts = 0
    for idx, window in group:
        row, did_fetch = _process_one_window(
            window,
            cache_path=cache_path,
            allow_network=allow_network,
            okx=okx,
            binance=binance,
        )
        fetch_attempts += did_fetch
        rows.append((idx, row))
    return rows, okx.calls, binance.calls, fetch_attempts


def _process_one_window(
    window: RequiredWindow,
    *,
    cache_path: Path,
    allow_network: bool,
    okx: OkxHistoryClient,
    binance: BinanceHistoryClient,
) -> tuple[dict[str, Any], int]:
    status, evidence = _window_cache_status(cache_path, window)
    source = "cache"
    error = ""
    did_fetch = 0
    if status != "available" and allow_network:
        did_fetch = 1
        fetch = okx.fetch(window.symbol, window.bar, window.start_ms, window.end_ms)
        fetch_error = fetch.error
        fallback_error = ""
        if fetch.rows:
            path = _write_cache_file(cache_path, window, fetch.rows, fetch.source)
            source = fetch.source
            status, evidence = _window_cache_status(cache_path, window)
            evidence = f"{path};{evidence}"
        if status != "available":
            fallback = binance.fetch(window.symbol, window.bar, window.start_ms, window.end_ms)
            fallback_error = fallback.error
            if fallback.rows:
                path = _write_cache_file(cache_path, window, fallback.rows, fallback.source)
                source = fallback.source
                status, evidence = _window_cache_status(cache_path, window)
                evidence = f"{path};{evidence}"
            elif not fetch.rows:
                error = fetch.error or fallback.error or "empty_exchange_response"
                evidence = _join_evidence(fetch.error, fallback.error)
        if status != "available" and not error:
            error = fetch.error or _missing_reason_from_cache_evidence(evidence)
        if status != "available":
            status = _terminal_status_from_exchange_errors(fetch_error, fallback_error, evidence)
    elif status != "available" and not allow_network:
        error = _missing_reason_from_cache_evidence(evidence)
    return (
        {
            "symbol": window.symbol,
            "bar": window.bar,
            "start_ms": window.start_ms,
            "end_ms": window.end_ms,
            "trade_count": len(window.trade_ids),
            "trade_ids": "|".join(window.trade_ids[:20]),
            "coverage_status": status,
            "source": source,
            "unavailable_reason": "" if status == "available" else error or "cache_window_missing",
            "evidence": evidence,
        },
        did_fetch,
    )


def _join_evidence(*parts: str) -> str:
    values = [part for part in parts if part]
    return ";".join(values) if values else "empty_exchange_response"


def _terminal_status_from_exchange_errors(okx_error: str, binance_error: str, evidence: str) -> str:
    text = f"{okx_error};{binance_error};{evidence}".lower()
    instrument_markers = (
        "instrument id does not exist",
        "invalid symbol",
        "no such symbol",
        "does not exist",
        "-1121",
        "51001",
    )
    if any(marker in text for marker in instrument_markers):
        return "instrument_unavailable"
    if "delist" in text or "delivery" in text:
        return "delisted_history_unavailable"
    if "too early" in text or "start time" in text or "listing" in text:
        return "listing_boundary"
    return "exchange_unavailable"


def _missing_reason_from_cache_evidence(evidence: str) -> str:
    if evidence == "no_cache_files" or evidence.startswith("covered=0/"):
        return "cache_window_missing"
    if evidence.startswith("covered="):
        return "partial_cache_below_threshold"
    return "cache_window_missing"


def _read_digest_required_windows(path: Path, *, bars: list[str], missing_only: bool = True) -> list[RequiredWindow]:
    allowed_bars = set(bars)
    terminal_statuses = {
        "instrument_unavailable",
        "delisted_history_unavailable",
        "listing_boundary",
        "exchange_unavailable",
    }
    windows: list[RequiredWindow] = []
    with path.open(encoding="utf-8") as handle:
        for idx, row in enumerate(csv.DictReader(handle), start=1):
            symbol = str(row.get("symbol") or "").strip()
            bar = str(row.get("timeframe") or row.get("bar") or "").strip()
            if not symbol or not bar or (allowed_bars and bar not in allowed_bars):
                continue
            status = str(row.get("coverage_status") or "").strip()
            if missing_only and status == "available":
                continue
            if missing_only and status in terminal_statuses:
                continue
            try:
                start_ms = int(float(row.get("start_ms") or 0))
                end_ms = int(float(row.get("end_ms") or 0))
            except (TypeError, ValueError):
                continue
            if start_ms <= 0 or end_ms <= 0 or end_ms < start_ms:
                continue
            trade_id = str(row.get("trade_id") or idx)
            windows.append(
                RequiredWindow(
                    symbol=symbol,
                    bar=bar,
                    start_ms=_floor_to_bar(start_ms, bar),
                    end_ms=_ceil_to_bar(end_ms, bar),
                    trade_ids=(trade_id,),
                )
            )
    return _merge_windows(windows)


def _floor_to_bar(ts: int, bar: str) -> int:
    size = BAR_MS[bar]
    return (ts // size) * size


def _ceil_to_bar(ts: int, bar: str) -> int:
    size = BAR_MS[bar]
    return ((ts + size - 1) // size) * size


def _print_progress(done: int, total: int, rows: list[dict[str, Any]]) -> None:
    available = sum(1 for row in rows if row["coverage_status"] == "available")
    print(f"[kline_backfill] {done}/{total} windows checked, available={available}", file=sys.stderr, flush=True)


def build_required_windows(trades: list[dict[str, Any]], *, bars: list[str]) -> list[RequiredWindow]:
    raw: list[RequiredWindow] = []
    for trade in trades:
        entry_ms = int(trade["entry_ts"])
        exit_ms = int(trade.get("exit_ts") or entry_ms)
        anchor_end = max(entry_ms, exit_ms)
        for bar in bars:
            window = WINDOWS[bar]
            if bar == "1m":
                end_ms = min(entry_ms + window["after_ms"], anchor_end + 2 * 60 * 60_000)
            else:
                end_ms = anchor_end + window["after_ms"]
            raw.append(
                RequiredWindow(
                    symbol=trade["symbol"],
                    bar=bar,
                    start_ms=_floor_to_bar(entry_ms - window["before_ms"], bar),
                    end_ms=_ceil_to_bar(end_ms, bar),
                    trade_ids=(str(trade.get("trade_id") or ""),),
                )
            )
    return _merge_windows(raw)


class OkxHistoryClient:
    def __init__(self, *, sleep_seconds: float, timeout: float):
        self.sleep_seconds = sleep_seconds
        self.timeout = timeout
        self.calls = 0

    def fetch(self, symbol: str, bar: str, start_ms: int, end_ms: int) -> FetchResult:
        rows: dict[int, dict[str, Any]] = {}
        cursor = end_ms + BAR_MS[bar]
        last_error = ""
        while cursor > start_ms:
            params = parse.urlencode({"instId": symbol, "bar": bar, "limit": "300", "after": str(cursor)})
            url = f"https://www.okx.com/api/v5/market/history-candles?{params}"
            payload, error = _get_json(url, timeout=self.timeout)
            self.calls += 1
            time.sleep(self.sleep_seconds)
            if error:
                last_error = error
                break
            if payload.get("code") != "0":
                last_error = f"okx:{payload.get('code')}:{payload.get('msg')}"
                break
            batch = payload.get("data") or []
            if not batch:
                break
            min_ts = cursor
            for item in batch:
                ts = int(item[0])
                min_ts = min(min_ts, ts)
                if start_ms <= ts <= end_ms:
                    rows[ts] = {
                        "ts": ts,
                        "open": item[1],
                        "high": item[2],
                        "low": item[3],
                        "close": item[4],
                        "vol": item[5],
                        "vol_ccy": item[6] if len(item) > 6 else "",
                        "vol_quote": item[7] if len(item) > 7 else "",
                        "confirm": item[8] if len(item) > 8 else "1",
                    }
            if min_ts >= cursor:
                break
            cursor = min_ts
            if min_ts <= start_ms:
                break
        return FetchResult("okx", [rows[ts] for ts in sorted(rows)], last_error)


class BinanceHistoryClient:
    def __init__(
        self,
        *,
        sleep_seconds: float,
        timeout: float,
        symbol_map: dict[str, str] | None = None,
        json_getter: Any | None = None,
    ):
        self.sleep_seconds = sleep_seconds
        self.timeout = timeout
        self.symbol_map = dict(symbol_map or {})
        self.json_getter = json_getter
        self.calls = 0

    def fetch(self, symbol: str, bar: str, start_ms: int, end_ms: int) -> FetchResult:
        rows: dict[int, dict[str, Any]] = {}
        cursor = start_ms
        last_error = ""
        while cursor <= end_ms:
            params = parse.urlencode(
                {
                    "symbol": self._binance_symbol(symbol),
                    "interval": BINANCE_INTERVALS[bar],
                    "startTime": str(cursor),
                    "endTime": str(end_ms),
                    "limit": "1500",
                }
            )
            url = f"https://fapi.binance.com/fapi/v1/klines?{params}"
            payload, error = self._get_json(url)
            self.calls += 1
            time.sleep(self.sleep_seconds)
            if error:
                last_error = error
                break
            if isinstance(payload, dict) and payload.get("code"):
                last_error = f"binance:{payload.get('code')}:{payload.get('msg')}"
                break
            if not payload:
                break
            max_ts = cursor
            for item in payload:
                ts = int(item[0])
                max_ts = max(max_ts, ts)
                if start_ms <= ts <= end_ms:
                    rows[ts] = {
                        "ts": ts,
                        "open": item[1],
                        "high": item[2],
                        "low": item[3],
                        "close": item[4],
                        "vol": item[5],
                        "vol_ccy": "",
                        "vol_quote": item[7] if len(item) > 7 else "",
                        "confirm": "1",
                    }
            next_cursor = max_ts + BAR_MS[bar]
            if next_cursor <= cursor:
                break
            cursor = next_cursor
        return FetchResult("binance", [rows[ts] for ts in sorted(rows)], last_error)

    def _binance_symbol(self, symbol: str) -> str:
        return self.symbol_map.get(symbol, _binance_symbol(symbol))

    def _get_json(self, url: str) -> tuple[Any, str]:
        if self.json_getter is not None:
            return self.json_getter(url, timeout=self.timeout)
        return _get_json(url, timeout=self.timeout)


def _read_trades(path: Path, *, time_offset_hours: int) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    trades: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        symbol = str(row.get("symbol") or "").strip()
        entry = str(row.get("entry_time") or "").strip()
        if not symbol or not entry:
            continue
        exit_time = str(row.get("exit_time") or entry).strip() or entry
        item = dict(row)
        item["trade_id"] = str(row.get("trade_id") or idx)
        item["symbol"] = symbol
        item["entry_ts"] = _parse_time_ms(entry, time_offset_hours=time_offset_hours)
        item["exit_ts"] = _parse_time_ms(exit_time, time_offset_hours=time_offset_hours)
        trades.append(item)
    return trades


def _parse_time_ms(value: str, *, time_offset_hours: int) -> int:
    raw = value.strip()
    if raw.isdigit():
        number = int(raw)
        return number if number > 10_000_000_000 else number * 1000
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc) + timedelta(hours=time_offset_hours)
    return int(dt.timestamp() * 1000)


def _merge_windows(windows: list[RequiredWindow]) -> list[RequiredWindow]:
    grouped: dict[tuple[str, str], list[RequiredWindow]] = {}
    for window in windows:
        grouped.setdefault((window.symbol, window.bar), []).append(window)
    merged: list[RequiredWindow] = []
    for (symbol, bar), rows in grouped.items():
        sorted_rows = sorted(rows, key=lambda row: (row.start_ms, row.end_ms))
        current: RequiredWindow | None = None
        max_gap = BAR_MS[bar] * 2
        max_span = _max_merged_span_ms(bar)
        for row in sorted_rows:
            if current is None:
                current = row
                continue
            new_start = current.start_ms
            new_end = max(current.end_ms, row.end_ms)
            if row.start_ms <= current.end_ms + max_gap and new_end - new_start <= max_span:
                current = RequiredWindow(
                    symbol=symbol,
                    bar=bar,
                    start_ms=new_start,
                    end_ms=new_end,
                    trade_ids=(*current.trade_ids, *row.trade_ids),
                )
            else:
                merged.append(current)
                current = row
        if current is not None:
            merged.append(current)
    return sorted(merged, key=lambda row: (_bar_sort_key(row.bar), row.symbol, row.start_ms))


def _bar_sort_key(bar: str) -> tuple[int, str]:
    order = {"1D": 0, "4H": 1, "1H": 2, "15m": 3, "5m": 4, "1m": 5}
    return (order.get(bar, 99), bar)


def _max_merged_span_ms(bar: str) -> int:
    if bar == "1m":
        return 7 * BAR_MS["1D"]
    if bar == "5m":
        return 21 * BAR_MS["1D"]
    if bar == "15m":
        return 60 * BAR_MS["1D"]
    return 420 * BAR_MS["1D"]


def _window_cache_status(cache_dir: Path, window: RequiredWindow) -> tuple[str, str]:
    files = sorted((cache_dir / window.bar).glob(f"{window.symbol}_*.csv"))
    if not files:
        return "missing", "no_cache_files"
    expected = max(1, int((window.end_ms - window.start_ms) / BAR_MS[window.bar]) + 1)
    covered_ts: set[int] = set()
    all_relevant_ts: set[int] = set()
    evidence_files: list[str] = []
    for path in files:
        first, last, count, timestamps = _read_cache_stats(path)
        if count <= 0:
            continue
        if last < window.start_ms or first > window.end_ms:
            continue
        evidence_files.append(str(path))
        covered_ts.update(ts for ts in timestamps if window.start_ms <= ts <= window.end_ms)
        all_relevant_ts.update(timestamps)
    ratio = len(covered_ts) / expected
    edge_covered = (
        bool(all_relevant_ts)
        and any(ts <= window.start_ms for ts in all_relevant_ts)
        and any(ts >= window.end_ms for ts in all_relevant_ts)
    )
    evidence = f"covered={len(covered_ts)}/{expected};edge_covered={str(edge_covered).lower()};files={'|'.join(evidence_files[:5])}"
    if ratio >= 0.85 and edge_covered:
        return "available", evidence
    return "missing", evidence


def _read_cache_stats(path: Path) -> tuple[int, int, int, list[int]]:
    timestamps: list[int] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                ts = int(float(row.get("ts") or row.get("timestamp") or 0))
                if ts > 0:
                    timestamps.append(ts)
    except (OSError, ValueError):
        return 0, 0, 0, []
    if not timestamps:
        return 0, 0, 0, []
    return min(timestamps), max(timestamps), len(timestamps), timestamps


def _write_cache_file(cache_dir: Path, window: RequiredWindow, rows: list[dict[str, Any]], source: str) -> Path:
    path = cache_dir / window.bar
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / f"{window.symbol}_merged.csv"
    merged: dict[int, dict[str, Any]] = {}
    for existing_path in sorted(path.glob(f"{window.symbol}_*.csv")):
        try:
            with existing_path.open(encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    ts = int(float(row.get("ts") or row.get("timestamp") or 0))
                    if ts > 0:
                        merged[ts] = {column: row.get(column, "") for column in CSV_COLUMNS}
        except (OSError, ValueError):
            continue
    for row in rows:
        ts = int(float(row.get("ts") or 0))
        if ts > 0:
            merged[ts] = {column: row.get(column, "") for column in CSV_COLUMNS}
    with file_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for ts in sorted(merged):
            writer.writerow(merged[ts])
    return file_path


def _get_json(url: str, *, timeout: float) -> tuple[Any, str]:
    try:
        req = request.Request(url, headers={"User-Agent": "langlang-trader/0.1", "Accept": "application/json"})
        with request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8")), ""
    except (URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        return {}, repr(exc)


def _binance_symbol(symbol: str) -> str:
    if symbol.endswith("-USDT-SWAP"):
        return symbol.replace("-USDT-SWAP", "USDT").replace("-", "")
    return symbol.replace("-", "")


def _fetch_binance_symbol_map(*, timeout: float) -> dict[str, str]:
    payload, _ = _get_json("https://fapi.binance.com/fapi/v1/exchangeInfo", timeout=timeout)
    if not isinstance(payload, dict):
        return {}
    mapping: dict[str, str] = {}
    for row in payload.get("symbols", []):
        if row.get("quoteAsset") != "USDT":
            continue
        if row.get("contractType") != "PERPETUAL":
            continue
        if row.get("status") != "TRADING":
            continue
        exchange_symbol = str(row.get("symbol") or "")
        if not exchange_symbol.endswith("USDT"):
            continue
        base = exchange_symbol[:-4]
        canonical_base = base
        for multiplier in ("1000000", "100000", "10000", "1000"):
            if canonical_base.startswith(multiplier):
                canonical_base = canonical_base[len(multiplier) :]
                break
        if canonical_base:
            mapping.setdefault(f"{canonical_base}-USDT-SWAP", exchange_symbol)
    return mapping


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _coverage_report(rows: list[dict[str, Any]], trades: list[dict[str, Any]], bars: list[str]) -> str:
    total = len(rows)
    available = sum(1 for row in rows if row["coverage_status"] == "available")
    unavailable = total - available
    by_bar: dict[str, dict[str, int]] = {}
    by_reason: dict[str, int] = {}
    for row in rows:
        bar_stats = by_bar.setdefault(row["bar"], {"total": 0, "available": 0})
        bar_stats["total"] += 1
        if row["coverage_status"] == "available":
            bar_stats["available"] += 1
        reason = row["unavailable_reason"] or "available"
        by_reason[reason] = by_reason.get(reason, 0) + 1
    lines = [
        "# LangLang v1.3 Market Data Coverage",
        "",
        f"- trades: {len(trades)}",
        f"- symbols: {len({trade['symbol'] for trade in trades})}",
        f"- bars: {','.join(bars)}",
        f"- required_windows: {total}",
        f"- available_windows: {available}",
        f"- unavailable_windows: {unavailable}",
        f"- coverage_status: {'complete_feature_coverage' if unavailable == 0 else 'partial_feature_coverage'}",
        "- unknown_windows: 0",
        "",
        "## By Bar",
        "",
    ]
    for bar in bars:
        stats = by_bar.get(bar, {"total": 0, "available": 0})
        lines.append(f"- {bar}: {stats['available']}/{stats['total']}")
    lines.extend(["", "## Reasons", ""])
    lines.extend(f"- {reason}: {count}" for reason, count in sorted(by_reason.items()))
    return "\n".join(lines) + "\n"


def _coverage_gap_report(rows: list[dict[str, Any]]) -> str:
    total = len(rows)
    available = sum(1 for row in rows if row["coverage_status"] == "available")
    terminal_statuses = {
        "instrument_unavailable",
        "delisted_history_unavailable",
        "listing_boundary",
        "exchange_unavailable",
    }
    terminal = sum(1 for row in rows if row["coverage_status"] in terminal_statuses)
    unresolved = total - available - terminal
    by_status: dict[str, int] = {}
    by_reason: dict[str, int] = {}
    by_bar: dict[str, int] = {}
    for row in rows:
        by_status[row["coverage_status"]] = by_status.get(row["coverage_status"], 0) + 1
        reason = row.get("unavailable_reason") or "available"
        by_reason[reason] = by_reason.get(reason, 0) + 1
        if row["coverage_status"] != "available":
            by_bar[row["bar"]] = by_bar.get(row["bar"], 0) + 1
    lines = [
        "# LangLang K-line Gap Report",
        "",
        f"- total_windows: {total}",
        f"- available_windows: {available}",
        f"- terminal_unavailable_windows: {terminal}",
        f"- unresolved_windows: {unresolved}",
        f"- unknown_windows: 0",
        "",
        "## Status",
        "",
    ]
    lines.extend(f"- {status}: {count}" for status, count in sorted(by_status.items()))
    lines.extend(["", "## Missing By Bar", ""])
    lines.extend(f"- {bar}: {count}" for bar, count in sorted(by_bar.items(), key=lambda item: _bar_sort_key(item[0])))
    lines.extend(["", "## Reasons", ""])
    lines.extend(f"- {reason}: {count}" for reason, count in sorted(by_reason.items()))
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())

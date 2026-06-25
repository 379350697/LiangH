from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable

from langlang_trader.kline_backfill import BAR_MS, CSV_COLUMNS, RequiredWindow, _write_cache_file
from langlang_trader.market_data import MarketData
from langlang_trader.models import Candle


class RollingKlineCacheMarketData:
    def __init__(
        self,
        cache_dir: str | Path,
        upstream: MarketData,
        *,
        now_ms: Callable[[], int] | None = None,
        freshness_multiplier: int = 2,
        fetch_buffer_bars: int = 2,
    ):
        self.cache_dir = Path(cache_dir)
        self.upstream = upstream
        self.now_ms = now_ms or (lambda: int(time.time() * 1000))
        self.freshness_multiplier = max(1, int(freshness_multiplier))
        self.fetch_buffer_bars = max(0, int(fetch_buffer_bars))
        self.stats = {
            "cache_hit": 0,
            "cache_miss": 0,
            "cache_stale": 0,
            "cache_stale_served": 0,
            "tail_fetch": 0,
            "tail_fetch_empty": 0,
            "tail_fetch_error": 0,
        }

    def get_candles(self, symbol: str, bar: str = "1D", limit: int = 120) -> list[Candle]:
        cached = _read_cached_candles(self.cache_dir, symbol, bar)
        if cached and len(cached) >= limit and self._is_fresh(cached[-1], bar):
            self.stats["cache_hit"] += 1
            return cached[-limit:]

        if cached:
            self.stats["cache_stale"] += 1
        else:
            self.stats["cache_miss"] += 1

        fetch_limit = self._tail_fetch_limit(cached, bar, limit)
        try:
            fetched = self.upstream.get_candles(symbol, bar=bar, limit=fetch_limit)
            self.stats["tail_fetch"] += 1
        except Exception:
            self.stats["tail_fetch_error"] += 1
            if cached:
                self.stats["cache_stale_served"] += 1
                return cached[-limit:]
            raise

        if not fetched:
            self.stats["tail_fetch_empty"] += 1
            if cached:
                self.stats["cache_stale_served"] += 1
                return cached[-limit:]
            raise RuntimeError(f"empty market data response for {symbol} {bar}")

        merged = _merge_candles([*cached, *fetched])
        _write_candles(self.cache_dir, symbol, bar, fetched)
        return merged[-limit:]

    def latest_price(self, symbol: str) -> float:
        return self.upstream.latest_price(symbol)

    def get_ticker(self, symbol: str):
        return self.upstream.get_ticker(symbol)

    def get_order_book(self, symbol: str, depth: int = 20):
        return self.upstream.get_order_book(symbol, depth=depth)

    def get_market_metrics(self, symbol: str) -> dict[str, Any]:
        return self.upstream.get_market_metrics(symbol)

    def _is_fresh(self, latest: Candle, bar: str) -> bool:
        bar_ms = BAR_MS.get(bar)
        if not bar_ms:
            return False
        return latest.ts >= self.now_ms() - self.freshness_multiplier * bar_ms

    def _tail_fetch_limit(self, cached: list[Candle], bar: str, requested_limit: int) -> int:
        if not cached:
            return requested_limit
        bar_ms = BAR_MS.get(bar)
        if not bar_ms:
            return requested_limit
        missing_by_count = max(0, requested_limit - len(cached))
        missing_by_time = max(0, math.ceil((self.now_ms() - cached[-1].ts) / bar_ms))
        return max(1, min(requested_limit, max(missing_by_count, missing_by_time) + self.fetch_buffer_bars))


class MarketSnapshotCache:
    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)

    def write_snapshot(
        self,
        *,
        run_id: str,
        phase: str,
        symbol: str,
        bar: str,
        last_ts: int,
        features: dict[str, Any],
    ) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / f"{_safe_name(run_id)}.jsonl"
        row = {
            "run_id": run_id,
            "phase": phase,
            "symbol": symbol,
            "bar": bar,
            "last_ts": last_ts,
            "features": _json_safe(features),
            "created_at_ms": int(time.time() * 1000),
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        return path


class MarketMetricsCache:
    def __init__(
        self,
        cache_dir: str | Path,
        *,
        ttl_seconds: int = 300,
        now_ms: Callable[[], int] | None = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.ttl_ms = max(1, int(ttl_seconds)) * 1000
        self.now_ms = now_ms or (lambda: int(time.time() * 1000))

    def get(self, symbol: str) -> dict[str, Any] | None:
        path = self._path(symbol)
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return None
        created_at_ms = int(row.get("created_at_ms") or 0)
        if created_at_ms <= 0 or self.now_ms() - created_at_ms > self.ttl_ms:
            return None
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            return None
        return metrics

    def put(self, symbol: str, metrics: dict[str, Any]) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._path(symbol)
        row = {
            "symbol": symbol,
            "created_at_ms": self.now_ms(),
            "metrics": _json_safe(metrics),
        }
        path.write_text(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def _path(self, symbol: str) -> Path:
        return self.cache_dir / f"{_safe_name(symbol)}.json"


def _read_cached_candles(cache_dir: Path, symbol: str, bar: str) -> list[Candle]:
    rows: dict[int, Candle] = {}
    for path in sorted((cache_dir / bar).glob(f"{symbol}_*.csv")):
        try:
            with path.open(encoding="utf-8") as handle:
                for raw in csv.DictReader(handle):
                    candle = _candle_from_row(symbol, bar, raw)
                    if candle is not None:
                        rows[candle.ts] = candle
        except (OSError, ValueError):
            continue
    return [rows[ts] for ts in sorted(rows)]


def _write_candles(cache_dir: Path, symbol: str, bar: str, candles: list[Candle]) -> None:
    if not candles:
        return
    rows = [
        {
            "ts": candle.ts,
            "open": candle.open,
            "high": candle.high,
            "low": candle.low,
            "close": candle.close,
            "vol": candle.volume,
            "vol_ccy": candle.vol_ccy if candle.vol_ccy is not None else "",
            "vol_quote": candle.vol_quote if candle.vol_quote is not None else "",
            "confirm": "1",
        }
        for candle in candles
    ]
    _write_cache_file(
        cache_dir,
        RequiredWindow(symbol=symbol, bar=bar, start_ms=min(candle.ts for candle in candles), end_ms=max(candle.ts for candle in candles), trade_ids=("runtime",)),
        rows,
        source="runtime",
    )


def _candle_from_row(symbol: str, bar: str, row: dict[str, Any]) -> Candle | None:
    ts = int(float(row.get("ts") or row.get("timestamp") or 0))
    if ts <= 0:
        return None
    return Candle(
        symbol=symbol,
        bar=bar,
        ts=ts,
        open=float(row.get("open") or 0),
        high=float(row.get("high") or 0),
        low=float(row.get("low") or 0),
        close=float(row.get("close") or 0),
        volume=float(row.get("vol") or row.get("volume") or 0),
        vol_ccy=_optional_float(row.get("vol_ccy")),
        vol_quote=_optional_float(row.get("vol_quote")),
        source="kline_cache",
    )


def _merge_candles(candles: list[Candle]) -> list[Candle]:
    rows: dict[int, Candle] = {}
    for candle in candles:
        rows[candle.ts] = candle
    return [rows[ts] for ts in sorted(rows)]


def _optional_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    return float(value)


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)

from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
import gzip
import json
import math
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable

from langlang_trader.kline_backfill import BAR_MS, CSV_COLUMNS, RequiredWindow, _write_cache_file
from langlang_trader.market_data import MarketData
from langlang_trader.models import Candle


class PublicMarketDataCooldownError(RuntimeError):
    pass


class PublicMarketDataGuard:
    def __init__(
        self,
        upstream: MarketData,
        *,
        provider_name: str = "public",
        now_ms: Callable[[], int] | None = None,
        cooldown_base_seconds: int = 60,
        cooldown_max_seconds: int = 300,
    ):
        self.upstream = upstream
        self.provider_name = provider_name
        self.now_ms = now_ms or (lambda: int(time.time() * 1000))
        self.cooldown_base_ms = max(1, int(cooldown_base_seconds)) * 1000
        self.cooldown_max_ms = max(self.cooldown_base_ms, int(cooldown_max_seconds) * 1000)
        self._states: dict[tuple[Any, ...], dict[str, Any]] = {}
        self.stats = {
            "public_market_cooldown_opened": 0,
            "public_market_cooldown_skipped": 0,
            "public_market_recovered": 0,
        }

    def get_candles(self, symbol: str, bar: str = "1D", limit: int = 120) -> list[Candle]:
        key = self._key("candles", symbol, bar)

        def call():
            rows = self.upstream.get_candles(symbol, bar=bar, limit=limit)
            if not rows:
                raise RuntimeError(f"empty market data response for {symbol} {bar}")
            return rows

        return self._call(key, call)

    def latest_price(self, symbol: str) -> float:
        return self._call(self._key("latest_price", symbol), lambda: self.upstream.latest_price(symbol))

    def get_ticker(self, symbol: str):
        return self._call(self._key("ticker", symbol), lambda: self.upstream.get_ticker(symbol))

    def get_order_book(self, symbol: str, depth: int = 20):
        return self._call(self._key("order_book", symbol, depth), lambda: self.upstream.get_order_book(symbol, depth=depth))

    def get_market_metrics(self, symbol: str) -> dict[str, Any]:
        return self._call(self._key("market_metrics", symbol), lambda: self.upstream.get_market_metrics(symbol))

    def _call(self, key: tuple[Any, ...], call: Callable[[], Any]) -> Any:
        now = self.now_ms()
        state = self._states.get(key)
        if state and now < int(state.get("cooldown_until_ms") or 0):
            self.stats["public_market_cooldown_skipped"] += 1
            raise PublicMarketDataCooldownError(
                "public_market_cooldown_skipped "
                f"provider={self.provider_name} key={key} consecutive_failures={state.get('consecutive_failures')} "
                f"cooldown_until_ms={state.get('cooldown_until_ms')} last_error={state.get('last_error')}"
            )
        try:
            result = call()
        except Exception as exc:
            failures = int(state.get("consecutive_failures", 0) if state else 0) + 1
            cooldown_ms = self._cooldown_ms(failures)
            self._states[key] = {
                "consecutive_failures": failures,
                "cooldown_until_ms": now + cooldown_ms,
                "last_error": repr(exc),
            }
            self.stats["public_market_cooldown_opened"] += 1
            raise RuntimeError(
                "public_market_cooldown_opened "
                f"provider={self.provider_name} key={key} consecutive_failures={failures} "
                f"cooldown_until_ms={now + cooldown_ms} last_error={exc!r}"
            ) from exc
        if state and int(state.get("consecutive_failures") or 0) > 0:
            self.stats["public_market_recovered"] += 1
        self._states.pop(key, None)
        return result

    def _cooldown_ms(self, failures: int) -> int:
        if failures <= 1:
            return min(self.cooldown_base_ms, self.cooldown_max_ms)
        return min(self.cooldown_max_ms, self.cooldown_base_ms * 3)

    def _key(self, method: str, symbol: str, *parts: Any) -> tuple[Any, ...]:
        return (self.provider_name, method, symbol, *parts)


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


SUMMARY_FEATURE_KEYS = {
    "requested_side",
    "selection_board",
    "selection_bias",
    "selection_score",
    "long_selection_score",
    "short_selection_score",
    "selection_reason_codes",
    "filter_codes",
    "ret_3d",
    "ret_7d",
    "ret_20d",
    "ret_60d",
    "pos_20d",
    "vol_ratio_20d",
    "liquidity_score",
    "funding_rate_last",
    "funding_rate_status",
    "oi_change_3d",
    "open_interest_status",
    "open_interest_usd",
    "open_interest_change_24h",
    "market_cap_status",
    "market_cap_usd",
    "book_depth_usdt_1pct",
    "book_depth_usdt_2pct",
    "spread_bps",
    "turnover_usdt",
    "turnover_rank",
    "turnover_rank_top_n",
    "strong_pattern_tag",
    "risk_pattern_tag",
    "strong_pattern_score",
    "risk_pattern_score",
    "pattern_reason_codes",
    "strong_pattern_reason_codes",
    "risk_pattern_reason_codes",
    "leader_platform_start_score",
    "golden_pit_reclaim_score",
    "small_divergence_absorb_score",
    "second_wave_start_score",
    "spoon_bottom_confirmed_score",
    "five_wave_late_risk_score",
    "false_breakout_risk_score",
    "leader_platform_start_reason_codes",
    "golden_pit_reclaim_reason_codes",
    "small_divergence_absorb_reason_codes",
    "second_wave_start_reason_codes",
    "spoon_bottom_confirmed_reason_codes",
    "five_wave_late_risk_reason_codes",
    "false_breakout_risk_reason_codes",
    "wyckoff_phase_tag",
    "wyckoff_long_setup_tag",
    "wyckoff_short_setup_tag",
    "wyckoff_exit_tag",
    "wyckoff_long_score",
    "wyckoff_short_score",
    "wyckoff_risk_score",
    "wyckoff_exit_score",
    "wyckoff_reason_codes",
    "wyckoff_long_reason_codes",
    "wyckoff_short_reason_codes",
    "wyckoff_risk_reason_codes",
    "wyckoff_exit_reason_codes",
    "market_data_cache_status",
    "market_metrics_cache_status",
    "data_quality_flags",
}


class MarketSnapshotCache:
    def __init__(
        self,
        cache_dir: str | Path,
        *,
        now_ms: Callable[[], int] | None = None,
        retention_days: int = 30,
    ):
        self.cache_dir = Path(cache_dir)
        self.now_ms = now_ms or (lambda: int(time.time() * 1000))
        self.retention_days = max(1, int(retention_days))
        self._compressed_for_date: str | None = None

    def write_snapshot(
        self,
        *,
        run_id: str,
        phase: str,
        symbol: str,
        bar: str,
        last_ts: int,
        features: dict[str, Any],
        full: bool = False,
        update_latest: bool = True,
    ) -> dict[str, Path]:
        created_at_ms = self.now_ms()
        day_key = _day_key(created_at_ms)
        self._compress_old_files(day_key)
        row_base = {
            "run_id": run_id,
            "phase": phase,
            "symbol": symbol,
            "bar": bar,
            "last_ts": last_ts,
            "created_at_ms": created_at_ms,
        }
        summary_row = {
            **row_base,
            "features": _json_safe(_summary_features(features)),
        }
        paths: dict[str, Path] = {}
        summary_path = self._append_row("summary", day_key, summary_row)
        paths["summary"] = summary_path
        if update_latest:
            paths["latest"] = self._write_latest(symbol, summary_row)
        if full:
            full_row = {
                **row_base,
                "features": _json_safe(features),
            }
            paths["full"] = self._append_row("full", day_key, full_row)
        return paths

    def _append_row(self, kind: str, day_key: str, row: dict[str, Any]) -> Path:
        path = self.cache_dir / kind / f"{day_key}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        return path

    def _write_latest(self, symbol: str, row: dict[str, Any]) -> Path:
        path = self.cache_dir / "latest" / f"{_safe_name(symbol)}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def _compress_old_files(self, current_day: str) -> None:
        if self._compressed_for_date == current_day:
            return
        self._compressed_for_date = current_day
        cutoff = datetime.strptime(current_day, "%Y-%m-%d").replace(tzinfo=timezone.utc) - timedelta(days=self.retention_days)
        for kind in ("summary", "full"):
            directory = self.cache_dir / kind
            if not directory.exists():
                continue
            for path in directory.glob("*.jsonl"):
                if path.name == f"{current_day}.jsonl":
                    continue
                _gzip_file(path)
            for path in directory.glob("*.jsonl.gz"):
                day = path.name[:10]
                try:
                    file_day = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
                if file_day < cutoff:
                    path.unlink(missing_ok=True)


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
        row = {
            "symbol": symbol,
            "created_at_ms": self.now_ms(),
            "metrics": _json_safe(metrics),
        }
        path = self._path(symbol)
        path.parent.mkdir(parents=True, exist_ok=True)
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


def _day_key(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def _summary_features(features: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, value in features.items():
        if key in SUMMARY_FEATURE_KEYS or _is_diagnostic_feature(key):
            summary[key] = value
    return summary


def _is_diagnostic_feature(key: str) -> bool:
    return (
        key.endswith("_error")
        or key.endswith("_errors")
        or key.endswith("_status")
        or "partial_bar_error" in key
        or "data_quality" in key
    )


def _gzip_file(path: Path) -> None:
    gz_path = path.with_suffix(path.suffix + ".gz")
    if gz_path.exists():
        path.unlink(missing_ok=True)
        return
    with path.open("rb") as src, gzip.open(gz_path, "wb") as dst:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)
    path.unlink(missing_ok=True)


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)

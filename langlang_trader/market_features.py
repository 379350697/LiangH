from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any
from urllib import parse, request
from urllib.error import URLError

from langlang_trader.features import FeatureSnapshot, _atr, _ema, _macd, _pct_change, _range_position, _rsi
from langlang_trader.models import Candle, utc_now_iso


KNOWN_FEATURE_STATUSES = {
    "available",
    "estimated",
    "estimated_turnover",
    "indicator_warmup",
    "listing_boundary",
    "exchange_unavailable",
    "provider_limited",
    "not_supported",
}


@dataclass(frozen=True)
class DerivativeMetric:
    symbol: str
    ts: int
    metric: str
    value: float
    source: str
    source_scope: str = "instrument"


class BinanceSymbolMapper:
    def __init__(self, base_to_exchange_symbol: dict[str, str] | None = None):
        self.base_to_exchange_symbol = dict(base_to_exchange_symbol or {})

    @classmethod
    def from_exchange_info(cls, payload: dict[str, Any]) -> "BinanceSymbolMapper":
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
                mapping.setdefault(canonical_base, exchange_symbol)
        return cls(mapping)

    def to_exchange_symbol(self, canonical_symbol: str) -> str:
        base = canonical_symbol
        if base.endswith("-USDT-SWAP"):
            base = base[: -len("-USDT-SWAP")]
        elif base.endswith("USDT"):
            base = base[:-4]
        base = base.replace("-", "")
        return self.base_to_exchange_symbol.get(base, f"{base}USDT")


class BinanceDerivativeMetricsClient:
    def __init__(
        self,
        *,
        symbol_mapper: BinanceSymbolMapper | None = None,
        base_url: str = "https://fapi.binance.com",
        json_getter: Any | None = None,
        sleep_seconds: float = 0.05,
        timeout: float = 15.0,
    ):
        self.symbol_mapper = symbol_mapper or BinanceSymbolMapper()
        self.base_url = base_url.rstrip("/")
        self.json_getter = json_getter
        self.sleep_seconds = sleep_seconds
        self.timeout = timeout

    def fetch_funding_rates(self, symbol: str, start_ms: int, end_ms: int, *, limit: int = 1000) -> list[DerivativeMetric]:
        exchange_symbol = self.symbol_mapper.to_exchange_symbol(symbol)
        cursor = start_ms
        rows: dict[int, DerivativeMetric] = {}
        while cursor <= end_ms:
            params = parse.urlencode(
                {
                    "symbol": exchange_symbol,
                    "startTime": str(cursor),
                    "endTime": str(end_ms),
                    "limit": str(limit),
                }
            )
            payload = self._get_json(f"{self.base_url}/fapi/v1/fundingRate?{params}")
            if isinstance(payload, dict) and payload.get("code"):
                break
            batch = payload or []
            if not batch:
                break
            max_ts = cursor
            for row in batch:
                if row.get("fundingTime") is None or row.get("fundingRate") is None:
                    continue
                ts = int(row["fundingTime"])
                max_ts = max(max_ts, ts)
                rows[ts] = DerivativeMetric(
                    symbol=symbol,
                    ts=ts,
                    metric="funding_rate",
                    value=float(row["fundingRate"]),
                    source="binance",
                )
            if len(batch) < limit or max_ts <= cursor:
                break
            cursor = max_ts + 1
        return [rows[ts] for ts in sorted(rows)]

    def fetch_open_interest(
        self,
        symbol: str,
        start_ms: int,
        end_ms: int,
        *,
        period: str = "1d",
        limit: int = 500,
    ) -> list[DerivativeMetric]:
        exchange_symbol = self.symbol_mapper.to_exchange_symbol(symbol)
        cursor = start_ms
        rows: dict[int, DerivativeMetric] = {}
        while cursor <= end_ms:
            params = parse.urlencode(
                {
                    "symbol": exchange_symbol,
                    "period": period,
                    "startTime": str(cursor),
                    "endTime": str(end_ms),
                    "limit": str(limit),
                }
            )
            payload = self._get_json(f"{self.base_url}/futures/data/openInterestHist?{params}")
            if isinstance(payload, dict) and payload.get("code"):
                break
            batch = payload or []
            if not batch:
                break
            max_ts = cursor
            for row in batch:
                ts = row.get("timestamp")
                value = row.get("sumOpenInterestValue") or row.get("sumOpenInterest")
                if ts is None or value is None:
                    continue
                parsed_ts = int(ts)
                max_ts = max(max_ts, parsed_ts)
                rows[parsed_ts] = DerivativeMetric(
                    symbol=symbol,
                    ts=parsed_ts,
                    metric="open_interest_usd",
                    value=float(value),
                    source="binance",
                )
            if len(batch) < limit or max_ts <= cursor:
                break
            cursor = max_ts + 1
        return [rows[ts] for ts in sorted(rows)]

    def _get_json(self, url: str) -> Any:
        if self.json_getter is not None:
            return self.json_getter(url)
        time.sleep(self.sleep_seconds)
        return _http_json(url, timeout=self.timeout)


class OkxDerivativeMetricsClient:
    def __init__(
        self,
        *,
        base_url: str = "https://www.okx.com",
        json_getter: Any | None = None,
        sleep_seconds: float = 0.05,
        timeout: float = 15.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.json_getter = json_getter
        self.sleep_seconds = sleep_seconds
        self.timeout = timeout

    def fetch_funding_rates(self, symbol: str, start_ms: int, end_ms: int, *, limit: int = 100) -> list[DerivativeMetric]:
        params = parse.urlencode(
            {
                "instId": symbol,
                "before": str(end_ms),
                "after": str(start_ms),
                "limit": str(limit),
            }
        )
        payload = self._get_json(f"{self.base_url}/api/v5/public/funding-rate-history?{params}")
        if payload.get("code") != "0":
            return []
        rows: list[DerivativeMetric] = []
        for row in payload.get("data") or []:
            ts = row.get("fundingTime")
            value = row.get("fundingRate")
            if ts is None or value is None:
                continue
            rows.append(
                DerivativeMetric(
                    symbol=symbol,
                    ts=int(ts),
                    metric="funding_rate",
                    value=float(value),
                    source="okx",
                )
            )
        return sorted(rows, key=lambda item: item.ts)

    def _get_json(self, url: str) -> dict[str, Any]:
        if self.json_getter is not None:
            return self.json_getter(url)
        time.sleep(self.sleep_seconds)
        payload = _http_json(url, timeout=self.timeout)
        return payload if isinstance(payload, dict) else {}


class HistoricalMarketFeatureBuilder:
    """Materialize historical market features without modifying raw K-line caches."""

    def build_technical_rows(self, candles_by_symbol_bar: dict[str, dict[str, list[Candle]]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for symbol, bars in sorted(candles_by_symbol_bar.items()):
            for timeframe, candles in sorted(bars.items()):
                ordered = sorted(candles, key=lambda row: row.ts)
                if not ordered:
                    continue
                rows.extend(_technical_rows_for_ordered(symbol, timeframe, ordered))
        return rows

    def build_derivative_rows(
        self,
        candles_by_symbol_bar: dict[str, dict[str, list[Candle]]],
        *,
        funding_metrics: list[DerivativeMetric] | None = None,
        open_interest_metrics: list[DerivativeMetric] | None = None,
    ) -> list[dict[str, Any]]:
        funding_by_symbol = _metrics_by_symbol(funding_metrics or [], "funding_rate")
        oi_by_symbol = _metrics_by_symbol(open_interest_metrics or [], "open_interest_usd")
        rows: list[dict[str, Any]] = []
        for symbol, bars in sorted(candles_by_symbol_bar.items()):
            candles = sorted(bars.get("1D", []), key=lambda row: row.ts)
            if not candles:
                continue
            oi_series: list[tuple[int, float | None]] = []
            for candle in candles:
                funding_window = _metrics_between(
                    funding_by_symbol.get(symbol, []),
                    _previous_ts(candles, candle.ts),
                    candle.ts,
                )
                funding_latest = _latest_metric(funding_by_symbol.get(symbol, []), candle.ts)
                oi_latest = _latest_metric(oi_by_symbol.get(symbol, []), candle.ts)
                oi_value = oi_latest.value if oi_latest else None
                oi_series.append((candle.ts, oi_value))
                row = {
                    "symbol": symbol,
                    "timeframe": "1D",
                    "ts": candle.ts,
                    "source": _source_pair(funding_latest, oi_latest),
                    "data_status": "available" if funding_latest or oi_latest else "exchange_unavailable",
                    "funding_rate_last": funding_latest.value if funding_latest else "",
                    "funding_rate_avg": fmean([metric.value for metric in funding_window]) if funding_window else "",
                    "funding_rate_sum": sum(metric.value for metric in funding_window) if funding_window else "",
                    "funding_rate_status": "available" if funding_latest else "exchange_unavailable",
                    "open_interest_usd": oi_value if oi_value is not None else "",
                    "open_interest_status": "available" if oi_latest else "exchange_unavailable",
                    "oi_change_1d": _oi_change(oi_series, 1),
                    "oi_change_3d": _oi_change(oi_series, 3),
                    "oi_change_7d": _oi_change(oi_series, 7),
                    "price_oi_divergence": _price_oi_divergence(candles, candle.ts, oi_series),
                }
                rows.append(row)
        return rows

    def build_external_market_rows(
        self,
        candles_by_symbol_bar: dict[str, dict[str, list[Candle]]],
        *,
        market_cap_rows: dict[tuple[str, int], dict[str, Any]] | None = None,
        listing_times: dict[str, int] | None = None,
    ) -> list[dict[str, Any]]:
        market_cap_rows = market_cap_rows or {}
        listing_times = listing_times or {}
        rows: list[dict[str, Any]] = []
        for symbol, bars in sorted(candles_by_symbol_bar.items()):
            for candle in sorted(bars.get("1D", []), key=lambda row: row.ts):
                provided = market_cap_rows.get((symbol, candle.ts))
                listing_ts = listing_times.get(symbol)
                if provided:
                    market_cap_status = "available"
                    data_status = "available"
                    market_cap = provided.get("market_cap_usd", "")
                    supply = provided.get("circulating_supply", "")
                else:
                    market_cap_status = "provider_limited"
                    data_status = "provider_limited"
                    market_cap = ""
                    supply = ""
                rows.append(
                    {
                        "symbol": symbol,
                        "timeframe": "1D",
                        "ts": candle.ts,
                        "source": provided.get("source", "external_provider_disabled") if provided else "external_provider_disabled",
                        "data_status": data_status,
                        "market_cap_usd": market_cap,
                        "circulating_supply": supply,
                        "market_cap_status": market_cap_status,
                        "listing_age_days": ((candle.ts - listing_ts) / 86_400_000) if listing_ts else "",
                        "listing_age_status": "available" if listing_ts else "provider_limited",
                    }
                )
        return rows

    def build_feature_coverage(self, rows_by_name: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
        coverage: list[dict[str, Any]] = []
        for table_name, rows in sorted(rows_by_name.items()):
            statuses = [str(row.get("data_status") or "") for row in rows]
            unknown = [status for status in statuses if status not in KNOWN_FEATURE_STATUSES]
            coverage.append(
                {
                    "table": table_name,
                    "rows": len(rows),
                    "available_rows": sum(1 for status in statuses if status == "available"),
                    "unavailable_rows": sum(1 for status in statuses if status != "available"),
                    "unknown_rows": len(unknown),
                    "coverage_status": "complete_status_coverage" if not unknown else "unknown_status_present",
                    "known_statuses": "|".join(sorted(set(statuses))),
                }
            )
        return coverage


def collect_historical_derivative_metrics(
    candles_by_symbol_bar: dict[str, dict[str, list[Candle]]],
    *,
    binance_client: BinanceDerivativeMetricsClient | Any | None = None,
    okx_client: OkxDerivativeMetricsClient | Any | None = None,
    progress_every: int = 0,
    progress_stream: Any | None = None,
) -> tuple[list[DerivativeMetric], list[DerivativeMetric], list[dict[str, Any]]]:
    binance_client = binance_client or BinanceDerivativeMetricsClient()
    okx_client = okx_client or OkxDerivativeMetricsClient()
    funding_rows: list[DerivativeMetric] = []
    oi_rows: list[DerivativeMetric] = []
    coverage_rows: list[dict[str, Any]] = []
    items = sorted(candles_by_symbol_bar.items())
    stream = progress_stream if progress_stream is not None else sys.stderr
    for idx, (symbol, bars) in enumerate(items, start=1):
        candles = sorted(bars.get("1D", []), key=lambda candle: candle.ts)
        if not candles:
            continue
        start_ms = candles[0].ts
        end_ms = candles[-1].ts + 86_400_000
        binance_funding = _safe_fetch_metrics(binance_client.fetch_funding_rates, symbol, start_ms, end_ms)
        okx_funding = _safe_fetch_metrics(okx_client.fetch_funding_rates, symbol, start_ms, end_ms)
        binance_oi = _safe_fetch_metrics(binance_client.fetch_open_interest, symbol, start_ms, end_ms)
        symbol_funding = sorted([*binance_funding, *okx_funding], key=lambda metric: (metric.ts, metric.source))
        funding_rows.extend(symbol_funding)
        oi_rows.extend(binance_oi)
        coverage_rows.append(
            {
                "symbol": symbol,
                "metric": "funding_rate",
                "source": _coverage_sources(symbol_funding),
                "start_ms": start_ms,
                "end_ms": end_ms,
                "rows": len(symbol_funding),
                "data_status": "available" if symbol_funding else "exchange_unavailable",
                "evidence": "binance_fundingRate|okx_funding-rate-history",
            }
        )
        coverage_rows.append(
            {
                "symbol": symbol,
                "metric": "open_interest_usd",
                "source": _coverage_sources(binance_oi),
                "start_ms": start_ms,
                "end_ms": end_ms,
                "rows": len(binance_oi),
                "data_status": "available" if binance_oi else "exchange_unavailable",
                "evidence": "binance_openInterestHist",
            }
        )
        if progress_every > 0 and idx % progress_every == 0:
            print(
                f"[market_features] derivative metrics {idx}/{len(items)} symbols processed",
                file=stream,
                flush=True,
            )
    return funding_rows, sorted(oi_rows, key=lambda metric: (metric.symbol, metric.ts)), coverage_rows


def _safe_fetch_metrics(fetcher: Any, symbol: str, start_ms: int, end_ms: int) -> list[DerivativeMetric]:
    try:
        return list(fetcher(symbol, start_ms, end_ms))
    except (TypeError, ValueError, OSError, URLError):
        return []


def _coverage_sources(metrics: list[DerivativeMetric]) -> str:
    if not metrics:
        return "exchange_unavailable"
    return "|".join(dict.fromkeys(metric.source for metric in metrics))


class MarketFeatureJoiner:
    def __init__(self, rows_by_key: dict[tuple[str, str, int], dict[str, Any]] | list[dict[str, Any]]):
        if isinstance(rows_by_key, dict):
            self.rows_by_key = rows_by_key
        else:
            self.rows_by_key = {
                (str(row["symbol"]), str(row["timeframe"]), int(row["ts"])): dict(row)
                for row in rows_by_key
                if row.get("symbol") and row.get("timeframe") and row.get("ts") is not None
            }

    def join(self, snapshot: FeatureSnapshot, *, timeframe: str = "1D") -> FeatureSnapshot:
        row = self.latest_row(snapshot.symbol, timeframe, snapshot.last_ts)
        if not row:
            return snapshot
        merged = dict(snapshot.features)
        for key, value in row.items():
            if key in {"symbol", "timeframe", "ts"}:
                continue
            merged[key] = value
        return FeatureSnapshot(
            symbol=snapshot.symbol,
            bar=snapshot.bar,
            last_ts=snapshot.last_ts,
            features=merged,
            created_at=snapshot.created_at or utc_now_iso(),
        )

    def latest_row(self, symbol: str, timeframe: str, ts: int) -> dict[str, Any] | None:
        candidates = [
            (row_ts, row)
            for (row_symbol, row_timeframe, row_ts), row in self.rows_by_key.items()
            if row_symbol == symbol and row_timeframe == timeframe and row_ts <= ts
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda item: item[0])[1]


def write_market_feature_artifacts(
    *,
    out_dir: str | Path,
    technical_rows: list[dict[str, Any]],
    derivative_rows: list[dict[str, Any]],
    external_rows: list[dict[str, Any]],
    trade_feature_rows: list[dict[str, Any]] | None = None,
    derivative_coverage_rows: list[dict[str, Any]] | None = None,
) -> dict[str, str]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    builder = HistoricalMarketFeatureBuilder()
    coverage_rows = builder.build_feature_coverage(
        {
            "technical_features": technical_rows,
            "derivatives_features": derivative_rows,
            "external_market_features": external_rows,
        }
    )
    paths = {
        "technical_features_multi_tf": str(out_path / "technical_features_multi_tf.csv"),
        "derivatives_features_1d": str(out_path / "derivatives_features_1d.csv"),
        "external_market_features_1d": str(out_path / "external_market_features_1d.csv"),
        "feature_coverage": str(out_path / "feature_coverage.csv"),
        "feature_coverage_json": str(out_path / "feature_coverage.json"),
        "trade_feature_matrix": str(out_path / "trade_feature_matrix.csv"),
        "derivative_market_data_coverage": str(out_path / "derivative_market_data_coverage.csv"),
    }
    technical_1d = [row for row in technical_rows if row.get("timeframe") == "1D"]
    paths["technical_features_1d"] = str(out_path / "technical_features_1d.csv")
    _write_rows(Path(paths["technical_features_1d"]), technical_1d)
    _write_rows(Path(paths["technical_features_multi_tf"]), technical_rows)
    _write_rows(Path(paths["derivatives_features_1d"]), derivative_rows)
    _write_rows(Path(paths["external_market_features_1d"]), external_rows)
    _write_rows(Path(paths["feature_coverage"]), coverage_rows)
    Path(paths["feature_coverage_json"]).write_text(json.dumps(coverage_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_rows(Path(paths["trade_feature_matrix"]), trade_feature_rows or [])
    _write_rows(Path(paths["derivative_market_data_coverage"]), derivative_coverage_rows or [])
    return paths


def _technical_rows_for_ordered(symbol: str, timeframe: str, ordered: list[Candle]) -> list[dict[str, Any]]:
    closes = [row.close for row in ordered]
    ema_12_series = _ema_series(closes, 12)
    ema_26_series = _ema_series(closes, 26)
    dif_series = [fast - slow for fast, slow in zip(ema_12_series, ema_26_series)]
    dea_series = _ema_series(dif_series, 9)
    true_ranges = _true_range_series(ordered)
    deltas = [0.0] + [closes[idx] - closes[idx - 1] for idx in range(1, len(closes))]
    last_ts = ordered[-1].ts
    rows: list[dict[str, Any]] = []
    for idx, candle in enumerate(ordered):
        rows.append(
            _technical_row_from_series(
                symbol=symbol,
                timeframe=timeframe,
                candle=candle,
                ordered=ordered,
                closes=closes,
                idx=idx,
                last_ts=last_ts,
                ema_12=ema_12_series[idx],
                ema_26=ema_26_series[idx],
                macd_dif=dif_series[idx],
                macd_dea=dea_series[idx],
                atr_14=_mean_slice(true_ranges, idx, 14),
                rsi_14=_rsi_from_deltas(deltas, idx, 14),
            )
        )
    return rows


def _technical_row_from_series(
    *,
    symbol: str,
    timeframe: str,
    candle: Candle,
    ordered: list[Candle],
    closes: list[float],
    idx: int,
    last_ts: int,
    ema_12: float,
    ema_26: float,
    macd_dif: float,
    macd_dea: float,
    atr_14: float,
    rsi_14: float,
) -> dict[str, Any]:
    start_20 = max(0, idx - 19)
    start_60 = max(0, idx - 59)
    highs_20 = [row.high for row in ordered[start_20 : idx + 1]]
    lows_20 = [row.low for row in ordered[start_20 : idx + 1]]
    highs_60 = [row.high for row in ordered[start_60 : idx + 1]]
    volumes_20 = [row.volume for row in ordered[start_20 : idx + 1]]
    macd_hist = macd_dif - macd_dea
    quote_volume = getattr(candle, "vol_quote", None)
    if quote_volume is not None and quote_volume > 0:
        turnover = quote_volume
        turnover_status = "available"
    else:
        turnover = candle.close * candle.volume
        turnover_status = "estimated_turnover"
    avg_volume_20 = fmean(volumes_20) if volumes_20 else 0.0
    high_20 = max(highs_20) if highs_20 else candle.high
    low_20 = min(lows_20) if lows_20 else candle.low
    high_60 = max(highs_60) if highs_60 else candle.high
    row = {
        "symbol": symbol,
        "timeframe": timeframe,
        "ts": candle.ts,
        "last_ts": last_ts,
        "source": getattr(candle, "source", "") or "kline_cache",
        "data_status": "available" if idx + 1 >= _minimum_history(timeframe) else "indicator_warmup",
        "open": candle.open,
        "high": candle.high,
        "low": candle.low,
        "close": candle.close,
        "volume": candle.volume,
        "quote_volume": quote_volume if quote_volume is not None else "",
        "turnover_usdt": turnover,
        "turnover_status": turnover_status,
        "ret_3": _indexed_return(closes, idx, 3),
        "ret_7": _indexed_return(closes, idx, 7),
        "ret_20": _indexed_return(closes, idx, 20),
        "ret_60": _indexed_return(closes, idx, 60),
        "pos_20": _range_position(candle.close, highs_20, lows_20),
        "pullback_from_20_high": (candle.close / high_20) - 1 if high_20 else 0.0,
        "high_20": high_20,
        "low_20": low_20,
        "high_60": high_60,
        "ma_5": _mean_closes(closes, idx, 5),
        "ma_20": _mean_closes(closes, idx, 20),
        "ma_60": _mean_closes(closes, idx, 60),
        "ema_12": ema_12,
        "ema_26": ema_26,
        "macd_dif": macd_dif,
        "macd_dea": macd_dea,
        "macd_hist": macd_hist,
        "atr_14": atr_14,
        "rsi_14": rsi_14,
        "vol_ratio_20": candle.volume / avg_volume_20 if avg_volume_20 > 0 else 0.0,
    }
    return row


def _indexed_return(closes: list[float], idx: int, window: int) -> float:
    if idx < window:
        return 0.0
    return _pct_change(closes[idx - window], closes[idx])


def _ema_series(values: list[float], period: int) -> list[float]:
    if not values:
        return []
    if period <= 1:
        return list(values)
    alpha = 2 / (period + 1)
    result = [values[0]]
    for value in values[1:]:
        result.append(value * alpha + result[-1] * (1 - alpha))
    return result


def _true_range_series(rows: list[Candle]) -> list[float]:
    result: list[float] = []
    previous_close: float | None = None
    for row in rows:
        if previous_close is None:
            result.append(row.high - row.low)
        else:
            result.append(max(row.high - row.low, abs(row.high - previous_close), abs(row.low - previous_close)))
        previous_close = row.close
    return result


def _mean_slice(values: list[float], idx: int, window: int) -> float:
    start = max(0, idx - window + 1)
    window_values = values[start : idx + 1]
    return fmean(window_values) if window_values else 0.0


def _mean_closes(closes: list[float], idx: int, window: int) -> float:
    return _mean_slice(closes, idx, window)


def _rsi_from_deltas(deltas: list[float], idx: int, period: int) -> float:
    if idx <= 0:
        return 50.0
    start = max(1, idx - period + 1)
    window = deltas[start : idx + 1]
    gains = [delta for delta in window if delta > 0]
    losses = [-delta for delta in window if delta < 0]
    avg_gain = fmean(gains) if gains else 0.0
    avg_loss = fmean(losses) if losses else 0.0
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _minimum_history(timeframe: str) -> int:
    if timeframe == "1D":
        return 61
    return 20


def _metrics_by_symbol(metrics: list[DerivativeMetric], metric_name: str) -> dict[str, list[DerivativeMetric]]:
    grouped: dict[str, list[DerivativeMetric]] = {}
    for metric in metrics:
        if metric.metric == metric_name:
            grouped.setdefault(metric.symbol, []).append(metric)
    return {symbol: sorted(rows, key=lambda item: item.ts) for symbol, rows in grouped.items()}


def _latest_metric(metrics: list[DerivativeMetric], ts: int) -> DerivativeMetric | None:
    latest: DerivativeMetric | None = None
    for metric in metrics:
        if metric.ts <= ts:
            latest = metric
        else:
            break
    return latest


def _metrics_between(metrics: list[DerivativeMetric], start_ts: int, end_ts: int) -> list[DerivativeMetric]:
    return [metric for metric in metrics if start_ts < metric.ts <= end_ts]


def _previous_ts(candles: list[Candle], ts: int) -> int:
    previous = ts - 86_400_000
    for candle in candles:
        if candle.ts < ts:
            previous = candle.ts
        else:
            break
    return previous


def _source_pair(left: DerivativeMetric | None, right: DerivativeMetric | None) -> str:
    sources = [metric.source for metric in (left, right) if metric is not None]
    return "|".join(dict.fromkeys(sources)) if sources else "derivatives_unavailable"


def _oi_change(oi_series: list[tuple[int, float | None]], lookback: int) -> float | str:
    if len(oi_series) <= lookback:
        return ""
    current = oi_series[-1][1]
    previous = oi_series[-lookback - 1][1]
    if current in {None, 0} or previous in {None, 0}:
        return ""
    return (float(current) / float(previous)) - 1.0


def _price_oi_divergence(candles: list[Candle], ts: int, oi_series: list[tuple[int, float | None]]) -> str:
    idx = next((i for i, candle in enumerate(candles) if candle.ts == ts), None)
    if idx is None or idx < 3 or len(oi_series) < 4:
        return ""
    price_change = (candles[idx].close / candles[idx - 3].close) - 1 if candles[idx - 3].close else 0.0
    oi_change = _oi_change(oi_series, 3)
    if oi_change == "":
        return ""
    oi_float = float(oi_change)
    if price_change > 0 and oi_float > 0:
        return "price_up_oi_up"
    if price_change < 0 and oi_float > 0:
        return "price_down_oi_up"
    if price_change < 0 and oi_float < 0:
        return "price_down_oi_down"
    if price_change > 0 and oi_float < 0:
        return "price_up_oi_down"
    return "neutral"


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _http_json(url: str, *, timeout: float) -> Any:
    try:
        req = request.Request(url, headers={"User-Agent": "langlang-trader/0.1", "Accept": "application/json"})
        with request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (URLError, TimeoutError, json.JSONDecodeError, OSError):
        return {}

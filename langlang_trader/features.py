from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Any

from langlang_trader.models import Candle, utc_now_iso
from langlang_trader.pattern_recognition import PatternConsensusScorer, StrongPatternDetector


@dataclass(frozen=True)
class FeatureSnapshot:
    symbol: str
    bar: str
    last_ts: int
    features: dict[str, Any]
    created_at: str


class DailyFeatureBuilder:
    def build(self, symbol: str, candles: list[Candle]) -> FeatureSnapshot | None:
        rows = sorted(candles, key=lambda candle: candle.ts)
        if len(rows) < 61:
            return None

        latest = rows[-1]
        closes = [row.close for row in rows]
        lows = [row.low for row in rows]
        highs = [row.high for row in rows]
        volumes = [row.volume for row in rows]
        ret_3d = _pct_change(closes[-4], closes[-1])
        ret_7d = _pct_change(closes[-8], closes[-1])
        ret_20d = _pct_change(closes[-21], closes[-1])
        ret_60d = _pct_change(closes[-61], closes[-1])
        high_20d = max(highs[-20:])
        low_20d = min(lows[-20:])
        high_60d = max(highs[-60:])
        pullback_from_20d_high = (latest.close / high_20d) - 1 if high_20d else 0.0
        range_width = high_20d - low_20d
        pos_20d = (latest.close - low_20d) / range_width if range_width > 0 else 0.5
        ma_5 = fmean(closes[-5:])
        ma_20 = fmean(closes[-20:])
        ma_60 = fmean(closes[-60:])
        ema_12 = _ema(closes, 12)
        ema_26 = _ema(closes, 26)
        macd_dif, macd_dea, macd_hist = _macd(closes)
        atr_14 = _atr(rows, 14)
        rsi_14 = _rsi(closes, 14)
        avg_vol_20d = fmean(volumes[-20:]) if volumes[-20:] else 0.0
        vol_ratio_20d = latest.volume / avg_vol_20d if avg_vol_20d > 0 else 0.0

        pattern_features = StrongPatternDetector().detect(rows)

        return FeatureSnapshot(
            symbol=symbol,
            bar=latest.bar,
            last_ts=latest.ts,
            created_at=utc_now_iso(),
            features={
                "ret_3d": ret_3d,
                "ret_7d": ret_7d,
                "ret_20d": ret_20d,
                "ret_60d": ret_60d,
                "pos_20d": pos_20d,
                "pullback_from_20d_high": pullback_from_20d_high,
                "ma_5": ma_5,
                "ma_20": ma_20,
                "ma_60": ma_60,
                "ema_12": ema_12,
                "ema_26": ema_26,
                "macd_dif": macd_dif,
                "macd_dea": macd_dea,
                "macd_hist": macd_hist,
                "atr_14": atr_14,
                "rsi_14": rsi_14,
                "high_20d": high_20d,
                "low_20d": low_20d,
                "high_60d": high_60d,
                "latest_close": latest.close,
                "latest_volume": latest.volume,
                "vol_ratio_20d": vol_ratio_20d,
                **pattern_features,
            },
        )


class MultiTimeframeFeatureBuilder:
    def build(self, symbol: str, candles_by_bar: dict[str, list[Candle]]) -> FeatureSnapshot | None:
        daily = DailyFeatureBuilder().build(symbol, candles_by_bar.get("1D", []))
        if daily is None:
            return None

        features = dict(daily.features)
        for prefix, bar, windows in (
            ("h1", "1H", (6, 24, 48)),
            ("m15", "15m", (8, 32, 64)),
            ("m5", "5m", (6, 24, 48)),
            ("m1", "1m", (15, 60, 120)),
        ):
            features.update(_bar_features(prefix, candles_by_bar.get(bar, []), windows))
        features.update(PatternConsensusScorer().score(candles_by_bar))

        return FeatureSnapshot(
            symbol=symbol,
            bar="multi",
            last_ts=daily.last_ts,
            created_at=utc_now_iso(),
            features=features,
        )


def _pct_change(start: float, end: float) -> float:
    if start == 0:
        return 0.0
    return (end / start) - 1.0


def _bar_features(prefix: str, candles: list[Candle], windows: tuple[int, int, int]) -> dict[str, Any]:
    rows = sorted(candles, key=lambda candle: candle.ts)
    if not rows:
        return {
            f"{prefix}_ret_{windows[0]}": 0.0,
            f"{prefix}_ret_{windows[1]}": 0.0,
            f"{prefix}_pos_{windows[1]}": 0.5,
            f"{prefix}_pullback_from_high": 0.0,
            f"{prefix}_ma_fast": 0.0,
            f"{prefix}_ma_slow": 0.0,
            f"{prefix}_ema_fast": 0.0,
            f"{prefix}_ema_slow": 0.0,
            f"{prefix}_macd_dif": 0.0,
            f"{prefix}_macd_dea": 0.0,
            f"{prefix}_macd_hist": 0.0,
            f"{prefix}_atr_14": 0.0,
            f"{prefix}_rsi_14": 0.0,
        }

    closes = [row.close for row in rows]
    highs = [row.high for row in rows]
    lows = [row.low for row in rows]
    short, medium, slow = windows
    latest = rows[-1]
    ret_short = _window_return(closes, short)
    ret_medium = _window_return(closes, medium)
    pos_medium = _range_position(latest.close, highs[-medium:], lows[-medium:])
    high_medium = max(highs[-medium:]) if highs else latest.high
    pullback = (latest.close / high_medium) - 1 if high_medium else 0.0
    ma_fast = fmean(closes[-min(short, len(closes)):])
    ma_slow = fmean(closes[-min(slow, len(closes)):])
    ema_fast = _ema(closes, min(12, max(2, len(closes))))
    ema_slow = _ema(closes, min(26, max(2, len(closes))))
    macd_dif, macd_dea, macd_hist = _macd(closes)
    atr_14 = _atr(rows, 14)
    rsi_14 = _rsi(closes, 14)
    return {
        f"{prefix}_ret_{short}": ret_short,
        f"{prefix}_ret_{medium}": ret_medium,
        f"{prefix}_pos_{medium}": pos_medium,
        f"{prefix}_pullback_from_high": pullback,
        f"{prefix}_ma_fast": ma_fast,
        f"{prefix}_ma_slow": ma_slow,
        f"{prefix}_ema_fast": ema_fast,
        f"{prefix}_ema_slow": ema_slow,
        f"{prefix}_macd_dif": macd_dif,
        f"{prefix}_macd_dea": macd_dea,
        f"{prefix}_macd_hist": macd_hist,
        f"{prefix}_atr_14": atr_14,
        f"{prefix}_rsi_14": rsi_14,
    }


def _window_return(closes: list[float], window: int) -> float:
    if len(closes) <= window:
        return 0.0
    return _pct_change(closes[-window - 1], closes[-1])


def _range_position(close: float, highs: list[float], lows: list[float]) -> float:
    if not highs or not lows:
        return 0.5
    high = max(highs)
    low = min(lows)
    width = high - low
    return (close - low) / width if width > 0 else 0.5


def _ema(values: list[float], period: int) -> float:
    if not values:
        return 0.0
    if period <= 1:
        return values[-1]
    alpha = 2 / (period + 1)
    current = values[0]
    for value in values[1:]:
        current = value * alpha + current * (1 - alpha)
    return current


def _ema_series(values: list[float], period: int) -> list[float]:
    if not values:
        return []
    alpha = 2 / (period + 1)
    series = [values[0]]
    for value in values[1:]:
        series.append(value * alpha + series[-1] * (1 - alpha))
    return series


def _macd(closes: list[float]) -> tuple[float, float, float]:
    if not closes:
        return 0.0, 0.0, 0.0
    fast = _ema_series(closes, 12)
    slow = _ema_series(closes, 26)
    dif_series = [fast_value - slow_value for fast_value, slow_value in zip(fast, slow)]
    dea = _ema(dif_series, 9)
    dif = dif_series[-1]
    return dif, dea, dif - dea


def _atr(rows: list[Candle], period: int) -> float:
    if not rows:
        return 0.0
    true_ranges: list[float] = []
    previous_close: float | None = None
    for row in rows:
        if previous_close is None:
            true_range = row.high - row.low
        else:
            true_range = max(row.high - row.low, abs(row.high - previous_close), abs(row.low - previous_close))
        true_ranges.append(true_range)
        previous_close = row.close
    window = true_ranges[-min(period, len(true_ranges)) :]
    return fmean(window) if window else 0.0


def _rsi(closes: list[float], period: int) -> float:
    if len(closes) < 2:
        return 50.0
    deltas = [closes[idx] - closes[idx - 1] for idx in range(1, len(closes))]
    window = deltas[-min(period, len(deltas)) :]
    gains = [delta for delta in window if delta > 0]
    losses = [-delta for delta in window if delta < 0]
    avg_gain = fmean(gains) if gains else 0.0
    avg_loss = fmean(losses) if losses else 0.0
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

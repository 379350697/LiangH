from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean, pstdev
from typing import Any

from langlang_trader.models import Candle


WYCKOFF_EVENT_KEYS = (
    "spring_reclaim",
    "sos_breakout",
    "lps_retest",
    "reaccumulation_breakout",
    "upthrust_reversal",
    "utad_risk",
    "sow_breakdown",
    "lpsy_retest",
    "no_demand_breakout",
    "effort_result_divergence",
)

LONG_SETUP_TAGS = {"spring_reclaim", "sos_breakout", "lps_retest", "reaccumulation_breakout"}
SHORT_SETUP_TAGS = {"upthrust_reversal", "utad_risk", "sow_breakdown", "lpsy_retest"}
RISK_TAGS = {"upthrust_reversal", "utad_risk", "sow_breakdown", "lpsy_retest", "no_demand_breakout", "effort_result_divergence"}
EXIT_TAGS = {"upthrust_reversal", "utad_risk", "sow_breakdown", "lpsy_retest", "effort_result_divergence"}
RECENT_WYCKOFF_WINDOW = 120


@dataclass(frozen=True)
class WyckoffRange:
    high: float
    low: float
    mid: float
    width_pct: float
    avg_volume: float
    quality_score: float
    volume_quality_score: float
    prior_return: float


class WyckoffRangeAnalyzer:
    def __init__(self, *, range_window: int = 30, event_window: int = 4):
        self.range_window = max(12, range_window)
        self.event_window = max(2, event_window)

    def analyze(self, candles: list[Candle]) -> WyckoffRange | None:
        rows = _sorted_rows(candles)[-RECENT_WYCKOFF_WINDOW:]
        if len(rows) < self.range_window // 2:
            return None
        base_end = max(0, len(rows) - self.event_window)
        base_start = max(0, base_end - self.range_window)
        base = rows[base_start:base_end] or rows[:-1]
        if len(base) < 8:
            return None
        highs = [row.high for row in base]
        lows = [row.low for row in base]
        closes = [row.close for row in base]
        volumes = [row.volume for row in base]
        high = max(highs)
        low = min(lows)
        mid = (high + low) / 2.0
        if mid <= 0:
            return None
        width_pct = (high - low) / mid
        close_stdev = pstdev(closes) / fmean(closes) if len(closes) >= 2 and fmean(closes) else width_pct
        width_quality = 1.0 - min(abs(width_pct - 0.11) / 0.18, 1.0)
        stability_quality = 1.0 - min(close_stdev / max(width_pct, 0.01), 1.0) * 0.35
        quality = _clamp(0.20 + 0.55 * width_quality + 0.25 * stability_quality)
        avg_volume = fmean(volumes) if volumes else 0.0
        recent_volume = fmean([row.volume for row in rows[-self.event_window:]]) if rows[-self.event_window:] else avg_volume
        volume_quality = _clamp(recent_volume / avg_volume / 2.0) if avg_volume > 0 else 0.0
        prior_idx = max(0, base_start - self.range_window)
        prior_price = rows[prior_idx].close if rows else 0.0
        prior_return = (base[0].close / prior_price - 1.0) if prior_price > 0 else 0.0
        return WyckoffRange(
            high=high,
            low=low,
            mid=mid,
            width_pct=width_pct,
            avg_volume=avg_volume,
            quality_score=quality,
            volume_quality_score=volume_quality,
            prior_return=prior_return,
        )


class WyckoffEventDetector:
    def __init__(self, *, range_analyzer: WyckoffRangeAnalyzer | None = None):
        self.range_analyzer = range_analyzer or WyckoffRangeAnalyzer()

    def detect(self, candles: list[Candle]) -> dict[str, Any]:
        rows = _sorted_rows(candles)[-RECENT_WYCKOFF_WINDOW:]
        features = _default_features()
        wyckoff_range = self.range_analyzer.analyze(rows)
        if wyckoff_range is None or len(rows) < 10:
            return features

        scores: dict[str, float] = {}
        reasons: dict[str, list[str]] = {event: [] for event in WYCKOFF_EVENT_KEYS}
        scores["spring_reclaim"] = _spring_reclaim_score(rows, wyckoff_range, reasons["spring_reclaim"])
        scores["sos_breakout"] = _sos_breakout_score(rows, wyckoff_range, reasons["sos_breakout"])
        scores["lps_retest"] = _lps_retest_score(rows, wyckoff_range, reasons["lps_retest"])
        scores["reaccumulation_breakout"] = _reaccumulation_breakout_score(
            rows,
            wyckoff_range,
            reasons["reaccumulation_breakout"],
            scores["sos_breakout"],
        )
        scores["upthrust_reversal"] = _upthrust_reversal_score(rows, wyckoff_range, reasons["upthrust_reversal"])
        scores["utad_risk"] = _utad_risk_score(rows, wyckoff_range, reasons["utad_risk"])
        scores["sow_breakdown"] = _sow_breakdown_score(rows, wyckoff_range, reasons["sow_breakdown"])
        scores["lpsy_retest"] = _lpsy_retest_score(rows, wyckoff_range, reasons["lpsy_retest"])
        scores["no_demand_breakout"] = _no_demand_breakout_score(rows, wyckoff_range, reasons["no_demand_breakout"])
        scores["effort_result_divergence"] = _effort_result_score(rows, wyckoff_range, reasons["effort_result_divergence"])

        for event, score in scores.items():
            features[f"wyckoff_{event}_score"] = score
            features[f"wyckoff_{event}_reason_codes"] = _dedupe(reasons[event])
        features["wyckoff_range_quality_score"] = wyckoff_range.quality_score
        features["wyckoff_volume_quality_score"] = wyckoff_range.volume_quality_score
        features["wyckoff_effort_result_score"] = scores["effort_result_divergence"]
        return _refresh_summary_fields(features)


class WyckoffConsensusScorer:
    def __init__(self, *, detector: WyckoffEventDetector | None = None):
        self.detector = detector or WyckoffEventDetector()

    def score(self, candles_by_bar: dict[str, list[Candle]]) -> dict[str, Any]:
        daily = self.detector.detect(candles_by_bar.get("1D", []))
        features = dict(daily)
        for prefix, bar in (("h4", "4H"), ("h1", "1H"), ("m15", "15m"), ("m5", "5m")):
            detected = self.detector.detect(candles_by_bar.get(bar, []))
            for key in (
                "wyckoff_long_score",
                "wyckoff_short_score",
                "wyckoff_risk_score",
                "wyckoff_exit_score",
                "wyckoff_range_quality_score",
                "wyckoff_volume_quality_score",
                "wyckoff_effort_result_score",
            ):
                features[f"{prefix}_{key}"] = detected.get(key, 0.0)
            for key in ("wyckoff_long_setup_tag", "wyckoff_short_setup_tag", "wyckoff_exit_tag", "wyckoff_phase_tag"):
                features[f"{prefix}_{key}"] = detected.get(key, "")
            if detected.get("wyckoff_long_score", 0.0) >= 0.50 and features.get("wyckoff_long_score", 0.0) >= 0.50:
                features["wyckoff_long_score"] = _clamp(features["wyckoff_long_score"] + 0.08)
                features["wyckoff_long_reason_codes"] = _dedupe(
                    [*features.get("wyckoff_long_reason_codes", []), *detected.get("wyckoff_long_reason_codes", [])]
                )
            if detected.get("wyckoff_short_score", 0.0) >= 0.50 and features.get("wyckoff_short_score", 0.0) >= 0.50:
                features["wyckoff_short_score"] = _clamp(features["wyckoff_short_score"] + 0.08)
                features["wyckoff_short_reason_codes"] = _dedupe(
                    [*features.get("wyckoff_short_reason_codes", []), *detected.get("wyckoff_short_reason_codes", [])]
                )
            if detected.get("wyckoff_risk_score", 0.0) >= 0.55 and features.get("wyckoff_risk_score", 0.0) >= 0.55:
                features["wyckoff_risk_score"] = _clamp(features["wyckoff_risk_score"] + 0.08)
                features["wyckoff_risk_reason_codes"] = _dedupe(
                    [*features.get("wyckoff_risk_reason_codes", []), *detected.get("wyckoff_risk_reason_codes", [])]
                )
        return _refresh_summary_fields(features)


def _default_features() -> dict[str, Any]:
    features: dict[str, Any] = {
        "wyckoff_phase_tag": "none",
        "wyckoff_long_setup_tag": "",
        "wyckoff_short_setup_tag": "",
        "wyckoff_exit_tag": "",
        "wyckoff_long_score": 0.0,
        "wyckoff_short_score": 0.0,
        "wyckoff_risk_score": 0.0,
        "wyckoff_exit_score": 0.0,
        "wyckoff_reason_codes": [],
        "wyckoff_long_reason_codes": [],
        "wyckoff_short_reason_codes": [],
        "wyckoff_risk_reason_codes": [],
        "wyckoff_exit_reason_codes": [],
        "wyckoff_range_quality_score": 0.0,
        "wyckoff_volume_quality_score": 0.0,
        "wyckoff_effort_result_score": 0.0,
    }
    for event in WYCKOFF_EVENT_KEYS:
        features[f"wyckoff_{event}_score"] = 0.0
        features[f"wyckoff_{event}_reason_codes"] = []
    return features


def _refresh_summary_fields(features: dict[str, Any]) -> dict[str, Any]:
    long_event, long_score = _best_event(features, LONG_SETUP_TAGS)
    short_event, short_score = _best_event(features, SHORT_SETUP_TAGS)
    risk_event, risk_score = _best_event(features, RISK_TAGS)
    exit_event, exit_score = _best_event(features, EXIT_TAGS)
    features["wyckoff_long_score"] = long_score
    features["wyckoff_short_score"] = short_score
    features["wyckoff_risk_score"] = risk_score
    features["wyckoff_exit_score"] = exit_score
    features["wyckoff_long_setup_tag"] = long_event if long_score >= 0.68 else ""
    features["wyckoff_short_setup_tag"] = short_event if short_score >= 0.70 else ""
    features["wyckoff_exit_tag"] = exit_event if exit_score >= 0.70 else ""
    features["wyckoff_long_reason_codes"] = _event_reasons(features, features["wyckoff_long_setup_tag"])
    features["wyckoff_short_reason_codes"] = _event_reasons(features, features["wyckoff_short_setup_tag"])
    features["wyckoff_risk_reason_codes"] = _event_reasons(features, risk_event) if risk_score >= 0.65 else []
    features["wyckoff_exit_reason_codes"] = _event_reasons(features, features["wyckoff_exit_tag"])
    features["wyckoff_reason_codes"] = _dedupe(
        [
            *features["wyckoff_long_reason_codes"],
            *features["wyckoff_short_reason_codes"],
            *features["wyckoff_risk_reason_codes"],
            *features["wyckoff_exit_reason_codes"],
        ]
    )
    features["wyckoff_phase_tag"] = _phase_tag(features)
    return features


def _best_event(features: dict[str, Any], events: set[str]) -> tuple[str, float]:
    return max(((event, _float(features.get(f"wyckoff_{event}_score"))) for event in events), key=lambda item: item[1])


def _event_reasons(features: dict[str, Any], event: str) -> list[str]:
    return _dedupe(features.get(f"wyckoff_{event}_reason_codes", [])) if event else []


def _phase_tag(features: dict[str, Any]) -> str:
    short_tag = str(features.get("wyckoff_short_setup_tag", ""))
    long_tag = str(features.get("wyckoff_long_setup_tag", ""))
    risk_score = _float(features.get("wyckoff_risk_score"))
    if short_tag in {"sow_breakdown", "lpsy_retest"}:
        return "markdown"
    if short_tag in {"utad_risk", "upthrust_reversal"}:
        return "distribution"
    if risk_score >= 0.70:
        return "distribution"
    if long_tag == "reaccumulation_breakout":
        return "reaccumulation"
    if long_tag:
        return "accumulation"
    return "none"


def _spring_reclaim_score(rows: list[Candle], wyckoff_range: WyckoffRange, reasons: list[str]) -> float:
    recent = rows[-4:]
    latest = rows[-1]
    spring_low = min(row.low for row in recent)
    reclaimed = latest.close >= wyckoff_range.low * 1.02 and latest.close >= wyckoff_range.mid * 0.98
    swept = spring_low <= wyckoff_range.low * 0.97
    volume_confirmed = _recent_volume(rows, 3) >= wyckoff_range.avg_volume * 1.35 if wyckoff_range.avg_volume else False
    score = 0.0
    if swept:
        score += 0.30
        reasons.append("wyckoff_spring_swept_range_low")
    if reclaimed:
        score += 0.28
        reasons.append("wyckoff_spring_reclaim")
    if volume_confirmed:
        score += 0.16
        reasons.append("wyckoff_spring_volume_expansion")
    score += 0.14 * wyckoff_range.quality_score
    return _clamp(score)


def _sos_breakout_score(rows: list[Candle], wyckoff_range: WyckoffRange, reasons: list[str]) -> float:
    latest = rows[-1]
    broke = latest.close >= wyckoff_range.high * 1.05 or max(row.close for row in rows[-3:]) >= wyckoff_range.high * 1.07
    volume_confirmed = _recent_volume(rows, 3) >= wyckoff_range.avg_volume * 1.25 if wyckoff_range.avg_volume else False
    holds_range = min(row.low for row in rows[-3:]) >= wyckoff_range.mid
    score = 0.0
    if broke:
        score += 0.34
        reasons.append("wyckoff_sos_breakout")
    if volume_confirmed:
        score += 0.17
        reasons.append("wyckoff_sos_volume_expansion")
    if holds_range:
        score += 0.12
        reasons.append("wyckoff_sos_holds_upper_range")
    score += 0.15 * wyckoff_range.quality_score
    return _clamp(score)


def _lps_retest_score(rows: list[Candle], wyckoff_range: WyckoffRange, reasons: list[str]) -> float:
    recent = rows[-5:]
    prior_breakout = max(row.close for row in recent[:-1]) >= wyckoff_range.high * 1.05 if len(recent) >= 2 else False
    retest_low = min(row.low for row in recent[-3:])
    holds_breakout_zone = wyckoff_range.high * 0.98 <= retest_low <= wyckoff_range.high * 1.08
    latest_recovers = rows[-1].close >= wyckoff_range.high * 1.02
    volume_quiet = rows[-2].volume <= wyckoff_range.avg_volume * 1.35 if len(rows) >= 2 and wyckoff_range.avg_volume else False
    score = 0.0
    if prior_breakout:
        score += 0.22
        reasons.append("wyckoff_lps_prior_sos")
    if holds_breakout_zone:
        score += 0.25
        reasons.append("wyckoff_lps_retest")
    if latest_recovers:
        score += 0.16
        reasons.append("wyckoff_lps_reclaim")
    if volume_quiet:
        score += 0.08
        reasons.append("wyckoff_lps_supply_quiet")
    score += 0.10 * wyckoff_range.quality_score
    return _clamp(score)


def _reaccumulation_breakout_score(rows: list[Candle], wyckoff_range: WyckoffRange, reasons: list[str], sos_score: float) -> float:
    if wyckoff_range.prior_return < 0.18 or sos_score < 0.55:
        return 0.0
    reasons.append("wyckoff_reaccumulation_breakout")
    return _clamp(sos_score + 0.08)


def _upthrust_reversal_score(rows: list[Candle], wyckoff_range: WyckoffRange, reasons: list[str]) -> float:
    recent = rows[-4:]
    thrust_high = max(row.high for row in recent)
    fell_back = rows[-1].close <= wyckoff_range.high * 1.01
    swept_high = thrust_high >= wyckoff_range.high * 1.07
    volume_expanded = _recent_volume(rows, 3) >= wyckoff_range.avg_volume * 1.35 if wyckoff_range.avg_volume else False
    score = 0.0
    if swept_high:
        score += 0.30
        reasons.append("wyckoff_upthrust_swept_range_high")
    if fell_back:
        score += 0.26
        reasons.append("wyckoff_upthrust_reversal")
    if volume_expanded:
        score += 0.14
        reasons.append("wyckoff_upthrust_volume_expansion")
    score += 0.10 * wyckoff_range.quality_score
    return _clamp(score)


def _utad_risk_score(rows: list[Candle], wyckoff_range: WyckoffRange, reasons: list[str]) -> float:
    recent = rows[-5:]
    above_range = max(row.high for row in recent) >= wyckoff_range.high * 1.10
    closes_back_inside = rows[-1].close <= wyckoff_range.high and rows[-1].close <= wyckoff_range.mid * 1.03
    heavy_supply = _recent_volume(rows, 3) >= wyckoff_range.avg_volume * 1.55 if wyckoff_range.avg_volume else False
    score = 0.0
    if above_range:
        score += 0.32
        reasons.append("wyckoff_utad_swept_high")
    if closes_back_inside:
        score += 0.27
        reasons.append("wyckoff_utad_risk")
    if heavy_supply:
        score += 0.16
        reasons.append("wyckoff_utad_supply_volume")
    score += 0.12 * wyckoff_range.quality_score
    return _clamp(score)


def _sow_breakdown_score(rows: list[Candle], wyckoff_range: WyckoffRange, reasons: list[str]) -> float:
    latest = rows[-1]
    broke = latest.close <= wyckoff_range.low * 0.95 or min(row.close for row in rows[-3:]) <= wyckoff_range.low * 0.93
    volume_expanded = _recent_volume(rows, 3) >= wyckoff_range.avg_volume * 1.35 if wyckoff_range.avg_volume else False
    weak_close = latest.close <= wyckoff_range.mid * 0.96
    score = 0.0
    if broke:
        score += 0.34
        reasons.append("wyckoff_sow_breakdown")
    if volume_expanded:
        score += 0.16
        reasons.append("wyckoff_sow_volume_expansion")
    if weak_close:
        score += 0.12
        reasons.append("wyckoff_sow_weak_close")
    score += 0.12 * wyckoff_range.quality_score
    return _clamp(score)


def _lpsy_retest_score(rows: list[Candle], wyckoff_range: WyckoffRange, reasons: list[str]) -> float:
    recent = rows[-5:]
    prior_breakdown = min(row.close for row in recent[:-1]) <= wyckoff_range.low * 0.95 if len(recent) >= 2 else False
    failed_retest = max(row.high for row in recent[-3:]) <= wyckoff_range.low * 1.03 and rows[-1].close <= wyckoff_range.low * 0.98
    volume_fades_on_retest = rows[-2].volume <= wyckoff_range.avg_volume * 1.45 if len(rows) >= 2 and wyckoff_range.avg_volume else False
    score = 0.0
    if prior_breakdown:
        score += 0.24
        reasons.append("wyckoff_lpsy_prior_sow")
    if failed_retest:
        score += 0.27
        reasons.append("wyckoff_lpsy_retest")
    if volume_fades_on_retest:
        score += 0.09
        reasons.append("wyckoff_lpsy_demand_weak")
    score += 0.10 * wyckoff_range.quality_score
    return _clamp(score)


def _no_demand_breakout_score(rows: list[Candle], wyckoff_range: WyckoffRange, reasons: list[str]) -> float:
    latest = rows[-1]
    breakout = latest.close >= wyckoff_range.high * 1.04
    weak_volume = latest.volume <= wyckoff_range.avg_volume * 0.90 if wyckoff_range.avg_volume else False
    narrow_body = _body_pct(latest) <= 0.30
    score = 0.0
    if breakout and weak_volume:
        score += 0.42
        reasons.append("wyckoff_no_demand_breakout")
    if breakout and narrow_body:
        score += 0.17
        reasons.append("wyckoff_no_demand_narrow_result")
    score += 0.08 * wyckoff_range.quality_score
    return _clamp(score)


def _effort_result_score(rows: list[Candle], wyckoff_range: WyckoffRange, reasons: list[str]) -> float:
    recent = rows[-3:]
    if not recent or wyckoff_range.avg_volume <= 0:
        return 0.0
    high_effort = fmean([row.volume for row in recent]) >= wyckoff_range.avg_volume * 1.75
    narrow_result = fmean([_body_pct(row) for row in recent]) <= 0.32
    price_progress = abs(rows[-1].close / rows[-4].close - 1.0) if len(rows) >= 4 and rows[-4].close else 0.0
    poor_progress = price_progress <= max(0.045, wyckoff_range.width_pct * 0.35)
    score = 0.0
    if high_effort:
        score += 0.26
        reasons.append("wyckoff_effort_high_volume")
    if narrow_result:
        score += 0.26
        reasons.append("wyckoff_effort_result_divergence")
    if poor_progress:
        score += 0.16
        reasons.append("wyckoff_poor_price_progress")
    return _clamp(score)


def _sorted_rows(candles: list[Candle]) -> list[Candle]:
    return sorted(candles, key=lambda candle: candle.ts)


def _recent_volume(rows: list[Candle], window: int) -> float:
    recent = rows[-min(window, len(rows)) :]
    return fmean([row.volume for row in recent]) if recent else 0.0


def _body_pct(row: Candle) -> float:
    width = max(row.high - row.low, 0.0)
    if width <= 0:
        return 1.0
    return abs(row.close - row.open) / width


def _float(value: Any, default: float = 0.0) -> float:
    if value in {None, ""}:
        return default
    return float(value)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result

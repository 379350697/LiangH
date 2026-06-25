from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Any

from langlang_trader.models import Candle


POSITIVE_PATTERN_KEYS = (
    "leader_platform_start_score",
    "golden_pit_reclaim_score",
    "small_divergence_absorb_score",
    "second_wave_start_score",
    "spoon_bottom_confirmed_score",
)

RISK_PATTERN_KEYS = (
    "five_wave_late_risk_score",
    "false_breakout_risk_score",
)

RECENT_PATTERN_WINDOW = 90


@dataclass(frozen=True)
class Pivot:
    idx: int
    kind: str
    price: float


@dataclass(frozen=True)
class PivotStructure:
    pivots: list[Pivot]
    wave_push_count: int
    pivot_quality_score: float


class PivotStructureAnalyzer:
    def __init__(self, *, local_window: int = 1, min_leg_pct: float = 0.03):
        self.local_window = max(1, local_window)
        self.min_leg_pct = max(0.0, min_leg_pct)

    def analyze(self, candles: list[Candle]) -> PivotStructure:
        rows = _sorted_rows(candles)
        closes = [row.close for row in rows]
        if len(closes) < 3:
            return PivotStructure(pivots=[], wave_push_count=0, pivot_quality_score=0.0)
        pivots = _turning_points(closes, self.local_window)
        pivots = _filter_small_legs(pivots, self.min_leg_pct)
        push_count = _wave_push_count_from_pivots(pivots)
        quality = _clamp(len(pivots) / 8.0) * (0.55 + min(push_count, 5) * 0.09)
        return PivotStructure(pivots=pivots, wave_push_count=push_count, pivot_quality_score=_clamp(quality))


class StrongPatternDetector:
    def __init__(self, *, pivot_analyzer: PivotStructureAnalyzer | None = None):
        self.pivot_analyzer = pivot_analyzer or PivotStructureAnalyzer()

    def detect(self, candles: list[Candle]) -> dict[str, Any]:
        rows = _sorted_rows(candles)[-RECENT_PATTERN_WINDOW:]
        closes = [row.close for row in rows]
        volumes = [row.volume for row in rows]
        structure = self.pivot_analyzer.analyze(rows)
        features = _default_features()
        features["wave_push_count"] = structure.wave_push_count
        features["pivot_quality_score"] = structure.pivot_quality_score
        if len(rows) < 8:
            return features

        reason_buckets: dict[str, list[str]] = {key: [] for key in (*POSITIVE_PATTERN_KEYS, *RISK_PATTERN_KEYS)}
        scores = {
            "leader_platform_start_score": _leader_platform_start_score(closes, volumes, reason_buckets["leader_platform_start_score"]),
            "golden_pit_reclaim_score": _golden_pit_reclaim_score(closes, volumes, reason_buckets["golden_pit_reclaim_score"]),
            "small_divergence_absorb_score": _small_divergence_absorb_score(closes, reason_buckets["small_divergence_absorb_score"]),
            "second_wave_start_score": _second_wave_start_score(closes, reason_buckets["second_wave_start_score"]),
            "spoon_bottom_confirmed_score": _spoon_bottom_confirmed_score(closes, reason_buckets["spoon_bottom_confirmed_score"]),
            "five_wave_late_risk_score": _five_wave_late_risk_score(closes, structure, reason_buckets["five_wave_late_risk_score"]),
            "false_breakout_risk_score": _false_breakout_risk_score(closes, volumes, reason_buckets["false_breakout_risk_score"]),
        }
        features.update(scores)
        for key, reasons in reason_buckets.items():
            features[_reason_field_for_score(key)] = _dedupe(reasons)
        features["small_divergence_count"] = _small_divergence_count(closes)

        positive_key, positive_score = max(
            ((key, scores[key]) for key in POSITIVE_PATTERN_KEYS),
            key=lambda item: item[1],
        )
        risk_key, risk_score = max(
            ((key, scores[key]) for key in RISK_PATTERN_KEYS),
            key=lambda item: item[1],
        )
        features["strong_pattern_score"] = positive_score
        features["risk_pattern_score"] = risk_score
        if positive_score >= 0.65:
            features["strong_pattern_tag"] = positive_key.removesuffix("_score")
        else:
            features["strong_pattern_tag"] = ""
        if risk_score >= 0.65:
            features["risk_pattern_tag"] = risk_key.removesuffix("_score")
        else:
            features["risk_pattern_tag"] = ""
        _refresh_reason_fields(features)
        return features


class PatternConsensusScorer:
    def __init__(self, *, detector: StrongPatternDetector | None = None):
        self.detector = detector or StrongPatternDetector()

    def score(self, candles_by_bar: dict[str, list[Candle]]) -> dict[str, Any]:
        daily = self.detector.detect(candles_by_bar.get("1D", []))
        if not daily:
            return _default_features()
        features = dict(daily)
        for prefix, bar in (("h4", "4H"), ("h1", "1H"), ("m15", "15m"), ("m5", "5m")):
            detected = self.detector.detect(candles_by_bar.get(bar, []))
            for key in (*POSITIVE_PATTERN_KEYS, *RISK_PATTERN_KEYS):
                features[f"{prefix}_{key}"] = detected.get(key, 0.0)
                if detected.get(key, 0.0) >= 0.65 and features.get(key, 0.0) >= 0.55:
                    reason_field = _reason_field_for_score(key)
                    features[reason_field] = _dedupe(
                        [*features.get(reason_field, []), *detected.get(reason_field, [])]
                    )
        if (
            features.get("golden_pit_reclaim_score", 0.0) >= 0.55
            and max(features.get("h1_golden_pit_reclaim_score", 0.0), features.get("m15_golden_pit_reclaim_score", 0.0)) >= 0.45
        ):
            features["golden_pit_reclaim_score"] = _clamp(features["golden_pit_reclaim_score"] + 0.10)
            features["golden_pit_reclaim_reason_codes"] = _dedupe(
                [*features.get("golden_pit_reclaim_reason_codes", []), "golden_pit_intraday_reclaim_confirmed"]
            )
        if (
            features.get("small_divergence_absorb_score", 0.0) >= 0.55
            and max(features.get("m15_small_divergence_absorb_score", 0.0), features.get("m5_small_divergence_absorb_score", 0.0)) >= 0.45
        ):
            features["small_divergence_absorb_score"] = _clamp(features["small_divergence_absorb_score"] + 0.08)
            features["small_divergence_absorb_reason_codes"] = _dedupe(
                [*features.get("small_divergence_absorb_reason_codes", []), "small_divergence_intraday_absorb_confirmed"]
            )
        return _refresh_summary_fields(features)


def _default_features() -> dict[str, Any]:
    features: dict[str, Any] = {
        "strong_pattern_tag": "",
        "risk_pattern_tag": "",
        "strong_pattern_score": 0.0,
        "risk_pattern_score": 0.0,
        "pattern_reason_codes": [],
        "strong_pattern_reason_codes": [],
        "risk_pattern_reason_codes": [],
        "wave_push_count": 0,
        "small_divergence_count": 0,
        "pivot_quality_score": 0.0,
    }
    for key in (*POSITIVE_PATTERN_KEYS, *RISK_PATTERN_KEYS):
        features[key] = 0.0
        features[_reason_field_for_score(key)] = []
    return features


def _refresh_summary_fields(features: dict[str, Any]) -> dict[str, Any]:
    positive_key, positive_score = max(
        ((key, _float(features.get(key))) for key in POSITIVE_PATTERN_KEYS),
        key=lambda item: item[1],
    )
    risk_key, risk_score = max(
        ((key, _float(features.get(key))) for key in RISK_PATTERN_KEYS),
        key=lambda item: item[1],
    )
    features["strong_pattern_score"] = positive_score
    features["risk_pattern_score"] = risk_score
    features["strong_pattern_tag"] = positive_key.removesuffix("_score") if positive_score >= 0.65 else ""
    features["risk_pattern_tag"] = risk_key.removesuffix("_score") if risk_score >= 0.65 else ""
    _refresh_reason_fields(features)
    return features


def _reason_field_for_score(score_key: str) -> str:
    return f"{score_key.removesuffix('_score')}_reason_codes"


def _refresh_reason_fields(features: dict[str, Any]) -> None:
    strong_tag = str(features.get("strong_pattern_tag", ""))
    risk_tag = str(features.get("risk_pattern_tag", ""))
    strong_reasons = _dedupe(features.get(f"{strong_tag}_reason_codes", [])) if strong_tag else []
    risk_reasons = _dedupe(features.get(f"{risk_tag}_reason_codes", [])) if risk_tag else []
    features["strong_pattern_reason_codes"] = strong_reasons
    features["risk_pattern_reason_codes"] = risk_reasons
    features["pattern_reason_codes"] = _dedupe([*strong_reasons, *risk_reasons])


def _leader_platform_start_score(closes: list[float], volumes: list[float], reasons: list[str]) -> float:
    if len(closes) < 24:
        return 0.0
    box = closes[-18:-3]
    if len(box) < 8:
        return 0.0
    box_mean = fmean(box)
    if box_mean <= 0:
        return 0.0
    prior_start = closes[max(0, len(closes) - 55)]
    prior_ret = box[0] / prior_start - 1 if prior_start else 0.0
    box_width = (max(box) - min(box)) / box_mean
    pre_box = closes[-35:-18] if len(closes) >= 35 else closes[:-18]
    pre_width = ((max(pre_box) - min(pre_box)) / fmean(pre_box)) if len(pre_box) >= 4 and fmean(pre_box) else box_width * 2
    contraction = box_width <= 0.08 and box_width <= pre_width * 0.75
    breakout = closes[-1] >= max(box) * 1.06 and closes[-2] >= max(box) * 1.02
    recent_volume = fmean(volumes[-3:]) if len(volumes) >= 3 else 0.0
    box_volume = fmean(volumes[-18:-3]) if len(volumes) >= 18 else recent_volume
    volume_alive = box_volume > 0 and recent_volume >= box_volume * 0.75
    score = 0.0
    if prior_ret >= 0.25:
        score += 0.28
        reasons.append("leader_platform_prior_strength")
    if contraction:
        score += 0.26
        reasons.append("platform_volatility_contracting")
    if breakout:
        score += 0.34
        reasons.append("leader_platform_start")
    if volume_alive:
        score += 0.12
        reasons.append("platform_volume_not_dead")
    return _clamp(score)


def _golden_pit_reclaim_score(closes: list[float], volumes: list[float], reasons: list[str]) -> float:
    if len(closes) < 18:
        return 0.0
    recent = closes[-14:]
    pit_offset = min(range(len(recent)), key=recent.__getitem__)
    pit_idx = len(closes) - len(recent) + pit_offset
    if pit_idx < 8 or pit_idx >= len(closes) - 1:
        return 0.0
    pre = closes[max(0, pit_idx - 12):pit_idx]
    if len(pre) < 5:
        return 0.0
    pre_low = min(pre)
    pre_high = max(pre)
    pit_low = closes[pit_idx]
    pit_depth = (pre_low - pit_low) / pre_low if pre_low else 0.0
    bars_since_pit = len(closes) - pit_idx - 1
    reclaim = closes[-1] >= pre_low * 1.02 and max(closes[pit_idx + 1:]) >= pre_low * 1.01
    fast = bars_since_pit <= 6
    prior_ret = pre[-1] / closes[max(0, pit_idx - 30)] - 1 if closes[max(0, pit_idx - 30)] else 0.0
    avg_vol = fmean(volumes[max(0, pit_idx - 12):pit_idx]) if volumes[max(0, pit_idx - 12):pit_idx] else 0.0
    reclaim_vol = fmean(volumes[pit_idx:min(len(volumes), pit_idx + 3)]) if volumes[pit_idx:min(len(volumes), pit_idx + 3)] else 0.0
    score = 0.0
    if prior_ret >= 0.18 or pre_high / pre_low - 1 <= 0.10:
        score += 0.18
        reasons.append("golden_pit_strong_context")
    if pit_depth >= 0.12:
        score += 0.28
        reasons.append("golden_pit_washout")
    if reclaim:
        score += 0.30
        reasons.append("golden_pit_fast_reclaim")
    if fast:
        score += 0.12
    if avg_vol > 0 and reclaim_vol >= avg_vol * 1.15:
        score += 0.12
        reasons.append("golden_pit_reclaim_volume")
    return _clamp(score)


def _small_divergence_absorb_score(closes: list[float], reasons: list[str]) -> float:
    if len(closes) < 18:
        return 0.0
    prior = closes[:-8] if len(closes) > 12 else closes[: max(1, len(closes) // 2)]
    trend = closes[-8] / closes[max(0, len(closes) - 35)] - 1 if closes[max(0, len(closes) - 35)] else 0.0
    recent_high = max(closes[-10:-2])
    recent_low = min(closes[-8:])
    pullback = (recent_high - recent_low) / recent_high if recent_high else 0.0
    reclaim = closes[-1] >= recent_low * 1.035 and closes[-1] >= closes[-3]
    structural_low = recent_low >= min(prior[-20:]) * 1.08 if len(prior) >= 4 else True
    score = 0.0
    if trend >= 0.25:
        score += 0.24
        reasons.append("small_divergence_main_wave_context")
    if 0.035 <= pullback <= 0.16:
        score += 0.28
        reasons.append("small_divergence_pullback_depth")
    if structural_low:
        score += 0.18
        reasons.append("small_divergence_not_breaking_structure")
    if reclaim:
        score += 0.25
        reasons.append("small_divergence_absorbed")
    return _clamp(score)


def _second_wave_start_score(closes: list[float], reasons: list[str]) -> float:
    if len(closes) < 20:
        return 0.0
    tail = closes[-18:]
    peak_idx = max(range(max(1, len(tail) // 2)), key=lambda idx: tail[idx])
    trough_idx = peak_idx + min(range(len(tail[peak_idx:])), key=lambda idx: tail[peak_idx + idx])
    if trough_idx <= peak_idx or trough_idx >= len(tail) - 2:
        return 0.0
    peak = tail[peak_idx]
    trough = tail[trough_idx]
    drop = (peak - trough) / peak if peak else 0.0
    post = tail[trough_idx + 1:]
    recovery = (closes[-1] - trough) / max(peak - trough, 1e-9)
    bottom_lift = len(post) >= 3 and min(post[-3:]) > trough * 1.035
    prior_trend = peak / closes[max(0, len(closes) - 45)] - 1 if closes[max(0, len(closes) - 45)] else 0.0
    score = 0.0
    if prior_trend >= 0.25:
        score += 0.18
    if drop >= 0.14:
        score += 0.28
        reasons.append("large_divergence_pullback")
    if bottom_lift:
        score += 0.24
        reasons.append("large_divergence_bottom_lift")
    if recovery >= 0.55:
        score += 0.25
        reasons.append("second_wave_reclaim")
    return _clamp(score)


def _spoon_bottom_confirmed_score(closes: list[float], reasons: list[str]) -> float:
    if len(closes) < 18:
        return 0.0
    window = closes[-45:]
    min_idx = min(range(len(window)), key=window.__getitem__)
    if min_idx < 4 or min_idx > len(window) - 5:
        return 0.0
    left_start = window[0]
    bottom = window[min_idx]
    left_drop = (left_start - bottom) / left_start if left_start else 0.0
    right_gain = window[-1] / bottom - 1 if bottom else 0.0
    if right_gain > 0.55:
        return 0.0
    right_lows = [min(window[idx:idx + 3]) for idx in range(min_idx, len(window) - 2, 3)]
    higher_lows = len(right_lows) >= 2 and right_lows[-1] > right_lows[0] * 1.08
    ma_fast = fmean(window[-5:])
    ma_slow = fmean(window[-min(15, len(window)):])
    neckline = max(window[:min_idx])
    confirmed = window[-1] >= neckline * 1.02 or window[-1] >= max(window[min_idx:-3]) * 1.06
    score = 0.0
    if left_drop >= 0.16:
        score += 0.18
        reasons.append("spoon_bottom_prior_decline")
    if right_gain >= 0.25:
        score += 0.24
        reasons.append("spoon_bottom_right_side_lift")
    if higher_lows:
        score += 0.20
        reasons.append("spoon_bottom_higher_lows")
    if ma_fast > ma_slow:
        score += 0.14
        reasons.append("spoon_bottom_ma_turn_up")
    if confirmed:
        score += 0.24
        reasons.append("spoon_bottom_confirmed")
    return _clamp(score)


def _five_wave_late_risk_score(closes: list[float], structure: PivotStructure, reasons: list[str]) -> float:
    if len(closes) < 10:
        return 0.0
    push_count = max(structure.wave_push_count, _simple_push_count(closes))
    total_ret = closes[-1] / closes[0] - 1 if closes[0] else 0.0
    high_pos = closes[-1] >= max(closes) * 0.96
    push_returns = _push_returns(structure.pivots)
    weakening = len(push_returns) >= 3 and push_returns[-1] <= fmean(push_returns[:-1]) * 0.75
    score = 0.0
    if push_count >= 5:
        score += 0.38
        reasons.append("five_wave_late_risk")
    if total_ret >= 0.45:
        score += 0.17
        reasons.append("five_wave_extended_ret")
    if high_pos:
        score += 0.14
        reasons.append("five_wave_high_position")
    if weakening or _small_divergence_count(closes) >= 2:
        score += 0.18
        reasons.append("five_wave_momentum_decay")
    if structure.pivot_quality_score >= 0.45:
        score += 0.08
    return _clamp(score)


def _false_breakout_risk_score(closes: list[float], volumes: list[float], reasons: list[str]) -> float:
    if len(closes) < 12:
        return 0.0
    lookback = min(8, len(closes) - 4)
    recent = closes[-lookback:]
    breakout_rel_idx = max(range(len(recent)), key=recent.__getitem__)
    breakout_idx = len(closes) - lookback + breakout_rel_idx
    if breakout_idx < 6 or breakout_idx >= len(closes) - 1:
        return 0.0
    pre = closes[max(0, breakout_idx - 12):breakout_idx]
    if len(pre) < 5:
        return 0.0
    pre_high = max(pre)
    pre_width = (pre_high - min(pre)) / fmean(pre) if fmean(pre) else 0.0
    broke_out = closes[breakout_idx] >= pre_high * 1.08
    fell_back = closes[-1] <= pre_high * 0.99
    avg_pre_vol = fmean(volumes[max(0, breakout_idx - 12):breakout_idx]) if volumes[max(0, breakout_idx - 12):breakout_idx] else 0.0
    breakout_vol = volumes[breakout_idx] if breakout_idx < len(volumes) else 0.0
    weak_volume = avg_pre_vol > 0 and breakout_vol <= avg_pre_vol * 1.10
    score = 0.0
    if pre_width <= 0.10:
        score += 0.15
        reasons.append("false_breakout_after_contraction")
    if broke_out:
        score += 0.25
    if fell_back:
        score += 0.28
        reasons.append("false_breakout_fell_back_into_box")
    if weak_volume:
        score += 0.12
        reasons.append("false_breakout_weak_volume")
    return _clamp(score)


def _small_divergence_count(closes: list[float]) -> int:
    if len(closes) < 8:
        return 0
    count = 0
    for idx in range(3, len(closes) - 2):
        left_high = max(closes[max(0, idx - 3):idx])
        local_low = min(closes[idx:idx + 3])
        if left_high > 0 and 0.035 <= (left_high - local_low) / left_high <= 0.16 and closes[min(len(closes) - 1, idx + 2)] >= local_low * 1.02:
            count += 1
    return min(count, 5)


def _sorted_rows(candles: list[Candle]) -> list[Candle]:
    return sorted(candles, key=lambda row: row.ts)


def _turning_points(closes: list[float], window: int) -> list[Pivot]:
    pivots: list[Pivot] = []
    for idx in range(window, len(closes) - window):
        left = closes[idx - window:idx]
        right = closes[idx + 1:idx + 1 + window]
        value = closes[idx]
        if value > max(left) and value >= max(right):
            pivots.append(Pivot(idx=idx, kind="high", price=value))
        elif value < min(left) and value <= min(right):
            pivots.append(Pivot(idx=idx, kind="low", price=value))
    return pivots


def _filter_small_legs(pivots: list[Pivot], min_leg_pct: float) -> list[Pivot]:
    filtered: list[Pivot] = []
    for pivot in pivots:
        if not filtered:
            filtered.append(pivot)
            continue
        previous = filtered[-1]
        if previous.kind == pivot.kind:
            if (pivot.kind == "high" and pivot.price > previous.price) or (pivot.kind == "low" and pivot.price < previous.price):
                filtered[-1] = pivot
            continue
        move = abs(pivot.price / previous.price - 1) if previous.price else 0.0
        if move >= min_leg_pct:
            filtered.append(pivot)
    return filtered


def _wave_push_count_from_pivots(pivots: list[Pivot]) -> int:
    count = 0
    previous_low: Pivot | None = None
    for pivot in pivots:
        if pivot.kind == "low":
            previous_low = pivot
        elif previous_low is not None and pivot.price > previous_low.price * 1.03:
            count += 1
    return count


def _simple_push_count(closes: list[float]) -> int:
    pivots = _filter_small_legs(_turning_points(closes, 1), 0.03)
    return _wave_push_count_from_pivots(pivots)


def _push_returns(pivots: list[Pivot]) -> list[float]:
    returns: list[float] = []
    previous_low: Pivot | None = None
    for pivot in pivots:
        if pivot.kind == "low":
            previous_low = pivot
        elif previous_low is not None and previous_low.price:
            returns.append(pivot.price / previous_low.price - 1)
    return returns


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value in {None, ""}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))

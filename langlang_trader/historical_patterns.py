from __future__ import annotations

from dataclasses import dataclass
import csv
import heapq
import json
from pathlib import Path
from typing import Any

from langlang_trader.distill_v1 import TradeLabeler


FEATURE_KEYS = ("ret_20d", "ret_60d", "pos_20d", "pullback_from_20d_high", "h1_ret_24", "m15_ret_8")


@dataclass(frozen=True)
class PatternMatch:
    score: float
    examples: list[dict[str, Any]]
    big_loss_overlap_count: int


def build_historical_patterns(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labels_by_trade = TradeLabeler().label_trades(trades)
    patterns: list[dict[str, Any]] = []
    for idx, trade in enumerate(trades):
        trade_id = str(trade.get("trade_id") or idx)
        labels = list(labels_by_trade.get(trade_id, []))
        if _is_main_wave_long(trade):
            labels.append("main_wave_long")
        if _is_waterfall_short(trade):
            labels.append("waterfall_short")
        if _float(trade.get("selection_bias_mismatch")) or trade.get("selection_bias_mismatch") is True:
            labels.append("selection_bias_mismatch")
        if _float(trade.get("stop_loss_cluster_24h")) >= 2:
            labels.append("stop_cluster")
        patterns.append(
            {
                "trade_id": trade_id,
                "symbol": trade.get("symbol", ""),
                "side": str(trade.get("side", "")).lower(),
                "regime": str(trade.get("regime", "")),
                "setup": str(trade.get("setup", "")),
                "labels": sorted(set(labels)),
                "pnl_usdt": _float(trade.get("pnl_usdt")),
                "return_rate": _float(trade.get("return_rate")),
                "hold_minutes": _float(trade.get("hold_minutes")),
                "features": {key: _float(trade.get(key)) for key in FEATURE_KEYS if _has_value(trade.get(key))},
            }
        )
    return patterns


def write_historical_patterns(path: str | Path, patterns: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["trade_id", "symbol", "side", "regime", "setup", "labels", "pnl_usdt", "return_rate", "hold_minutes", "features_json"]
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for pattern in patterns:
            writer.writerow(
                {
                    "trade_id": pattern.get("trade_id", ""),
                    "symbol": pattern.get("symbol", ""),
                    "side": pattern.get("side", ""),
                    "regime": pattern.get("regime", ""),
                    "setup": pattern.get("setup", ""),
                    "labels": "|".join(pattern.get("labels", [])),
                    "pnl_usdt": pattern.get("pnl_usdt", 0.0),
                    "return_rate": pattern.get("return_rate", 0.0),
                    "hold_minutes": pattern.get("hold_minutes", 0.0),
                    "features_json": json.dumps(pattern.get("features", {}), ensure_ascii=False, sort_keys=True),
                }
            )


def read_historical_patterns(path: str | Path) -> list[dict[str, Any]]:
    src = Path(path)
    if not src.exists():
        return []
    patterns: list[dict[str, Any]] = []
    with src.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            patterns.append(
                {
                    "trade_id": row.get("trade_id", ""),
                    "symbol": row.get("symbol", ""),
                    "side": row.get("side", ""),
                    "regime": row.get("regime", ""),
                    "setup": row.get("setup", ""),
                    "labels": [item for item in str(row.get("labels", "")).split("|") if item],
                    "pnl_usdt": _float(row.get("pnl_usdt")),
                    "return_rate": _float(row.get("return_rate")),
                    "hold_minutes": _float(row.get("hold_minutes")),
                    "features": _json_dict(row.get("features_json")),
                }
            )
    return patterns


class HistoricalPatternMatcher:
    def __init__(self, patterns: list[dict[str, Any]], *, max_examples: int = 3, max_candidates_per_side: int = 200):
        self.patterns = patterns
        self.max_examples = max_examples
        self.patterns_by_side: dict[str, list[dict[str, Any]]] = {}
        self.cache: dict[tuple[Any, ...], PatternMatch] = {}
        for pattern in patterns:
            self.patterns_by_side.setdefault(str(pattern.get("side", "")).lower(), []).append(pattern)
        for side, rows in list(self.patterns_by_side.items()):
            self.patterns_by_side[side] = sorted(rows, key=_candidate_priority, reverse=True)[:max_candidates_per_side]

    def match(self, *, side: str, regime: str, setup: str, features: dict[str, Any]) -> PatternMatch:
        side = str(side).lower()
        cache_key = _cache_key(side, regime, setup, features)
        if cache_key in self.cache:
            return self.cache[cache_key]
        candidates = self.patterns_by_side.get(side, [])
        scored = []
        for pattern in candidates:
            score = _pattern_score(pattern, regime, setup, features)
            if score > 0:
                scored.append((score, str(pattern.get("trade_id")), pattern))
        best = heapq.nlargest(self.max_examples, scored, key=lambda item: (item[0], item[1]))
        examples = [_example(pattern, score) for score, _trade_id, pattern in best]
        big_loss_overlap = sum(
            1 for score, _trade_id, pattern in best if "big_loss" in pattern.get("labels", []) and score >= 0.55
        )
        positive_scores = [
            score
            for score, _trade_id, pattern in best
            if "big_loss" not in pattern.get("labels", [])
        ]
        score = max(positive_scores) if positive_scores else (best[0][0] if best else 0.0)
        if big_loss_overlap:
            score = max(0.0, score - 0.20 * big_loss_overlap)
        result = PatternMatch(score=score, examples=examples, big_loss_overlap_count=big_loss_overlap)
        self.cache[cache_key] = result
        return result


def _pattern_score(pattern: dict[str, Any], regime: str, setup: str, features: dict[str, Any]) -> float:
    score = 0.15
    if pattern.get("regime") and pattern.get("regime") == regime:
        score += 0.25
    if pattern.get("setup") and pattern.get("setup") == setup:
        score += 0.25
    distances = []
    pattern_features = pattern.get("features", {})
    for key in FEATURE_KEYS:
        if key not in pattern_features or key not in features:
            continue
        distances.append(abs(_float(features.get(key)) - _float(pattern_features.get(key))))
    if distances:
        avg_distance = sum(distances) / len(distances)
        score += max(0.0, 0.30 - min(0.30, avg_distance))
    if "big_win" in pattern.get("labels", []) or "right_tail" in pattern.get("labels", []):
        score += 0.10
    if "big_loss" in pattern.get("labels", []):
        score -= 0.08
    return max(0.0, min(1.0, score))


def _example(pattern: dict[str, Any], score: float) -> dict[str, Any]:
    return {
        "trade_id": pattern.get("trade_id"),
        "symbol": pattern.get("symbol"),
        "side": pattern.get("side"),
        "regime": pattern.get("regime"),
        "setup": pattern.get("setup"),
        "labels": pattern.get("labels", []),
        "pnl_usdt": pattern.get("pnl_usdt", 0.0),
        "score": round(score, 6),
    }


def _json_dict(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _cache_key(side: str, regime: str, setup: str, features: dict[str, Any]) -> tuple[Any, ...]:
    rounded = tuple(round(_float(features.get(key)), 4) for key in FEATURE_KEYS)
    return (side, regime, setup, *rounded)


def _is_main_wave_long(trade: dict[str, Any]) -> bool:
    return (
        str(trade.get("side", "")).lower() == "long"
        and _float(trade.get("ret_20d")) >= 0.20
        and _float(trade.get("pos_20d"), 0.5) >= 0.45
    )


def _candidate_priority(pattern: dict[str, Any]) -> tuple[float, float]:
    labels = set(pattern.get("labels", []))
    label_score = 0.0
    if "right_tail" in labels:
        label_score += 5.0
    if "big_win" in labels:
        label_score += 4.0
    if "main_wave_long" in labels or "waterfall_short" in labels:
        label_score += 2.0
    if "big_loss" in labels:
        label_score += 1.5
    if "fast_failure" in labels or "chase_failure" in labels:
        label_score += 1.0
    return label_score, abs(_float(pattern.get("pnl_usdt")))


def _is_waterfall_short(trade: dict[str, Any]) -> bool:
    return (
        str(trade.get("side", "")).lower() == "short"
        and _float(trade.get("ret_20d")) <= -0.12
        and _float(trade.get("pos_20d"), 0.5) <= 0.35
    )


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value in {None, ""}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _has_value(value: Any) -> bool:
    return value is not None and value != ""

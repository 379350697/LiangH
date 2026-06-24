from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any
from zoneinfo import ZoneInfo

from langlang_trader.config import SymbolSelectionConfig
from langlang_trader.features import DailyFeatureBuilder, FeatureSnapshot
from langlang_trader.models import Candle


LOCAL_TZ = ZoneInfo("Asia/Shanghai")


@dataclass(frozen=True)
class SymbolSelectionResult:
    symbol: str
    selected: bool
    selection_rank: int
    selection_score: float
    selection_bias: str
    reason_codes: list[str]
    features: dict[str, Any]
    selection_mode: str = "mixed"
    market_env: dict[str, Any] = field(default_factory=dict)
    filter_codes: list[str] = field(default_factory=list)
    data_status: str = "available"
    unavailable_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SymbolSelector:
    """Cross-sectional selector that explains why a contract deserves attention."""

    def __init__(self, config: SymbolSelectionConfig | None = None):
        self.config = config or SymbolSelectionConfig()

    def rank(
        self,
        snapshots: dict[str, FeatureSnapshot],
        *,
        top_n: int | None = None,
        min_score: float | None = None,
    ) -> list[SymbolSelectionResult]:
        usable = {symbol: snapshot for symbol, snapshot in snapshots.items() if snapshot is not None}
        if not usable:
            return []

        top_n = self.config.top_n if top_n is None else top_n
        min_score = self.config.min_score if min_score is None else min_score
        ret_20d_pctiles = _percentiles({symbol: _float(snapshot.features.get("ret_20d")) for symbol, snapshot in usable.items()})
        ret_60d_pctiles = _percentiles({symbol: _float(snapshot.features.get("ret_60d")) for symbol, snapshot in usable.items()})
        vol_pctiles = _percentiles({symbol: _float(snapshot.features.get("vol_ratio_20d"), 1.0) for symbol, snapshot in usable.items()})

        raw_results: list[SymbolSelectionResult] = []
        for symbol, snapshot in usable.items():
            features = snapshot.features
            ret_20d = _float(features.get("ret_20d"))
            ret_60d = _float(features.get("ret_60d"))
            pos_20d = _clamp(_float(features.get("pos_20d"), 0.5))
            pullback = _float(features.get("pullback_from_20d_high"))
            vol_ratio = _float(features.get("vol_ratio_20d"), 1.0)
            ret_20_pct = ret_20d_pctiles[symbol]
            ret_60_pct = ret_60d_pctiles[symbol]
            vol_pct = vol_pctiles[symbol]
            near_high_score = _clamp(pos_20d)
            breakdown_score = _clamp(1.0 - pos_20d)
            long_score = (
                0.36 * ret_20_pct
                + 0.22 * ret_60_pct
                + 0.20 * pos_20d
                + 0.12 * vol_pct
                + 0.10 * near_high_score
            )
            short_score = (
                0.36 * (1.0 - ret_20_pct)
                + 0.22 * (1.0 - ret_60_pct)
                + 0.20 * (1.0 - pos_20d)
                + 0.12 * vol_pct
                + 0.10 * breakdown_score
            )
            bias = "long" if long_score >= short_score else "short"
            score = long_score if bias == "long" else short_score
            selection_features = {
                "selection_score": score,
                "selection_bias": bias,
                "long_selection_score": long_score,
                "short_selection_score": short_score,
                "ret_20d_pctile": ret_20_pct,
                "ret_60d_pctile": ret_60_pct,
                "pos_20d_pctile": pos_20d,
                "vol_ratio_20d_pctile": vol_pct,
                "selection_ret_20d": ret_20d,
                "selection_ret_60d": ret_60d,
                "selection_pos_20d": pos_20d,
                "selection_pullback_from_20d_high": pullback,
                "selection_vol_ratio_20d": vol_ratio,
            }
            reason_codes = _reason_codes(
                bias=bias,
                ret_20_pct=ret_20_pct,
                ret_60_pct=ret_60_pct,
                pos_20d=pos_20d,
                vol_pct=vol_pct,
                vol_ratio=vol_ratio,
                pullback=pullback,
                features=features,
            )
            raw_results.append(
                SymbolSelectionResult(
                    symbol=symbol,
                    selected=False,
                    selection_rank=0,
                    selection_score=score,
                    selection_bias=bias,
                    reason_codes=reason_codes,
                    features=selection_features,
                )
            )

        ranked = sorted(raw_results, key=lambda result: (-result.selection_score, result.symbol))
        final: list[SymbolSelectionResult] = []
        for idx, result in enumerate(ranked, start=1):
            selected = result.selection_score >= min_score and (top_n <= 0 or idx <= top_n)
            result_features = dict(result.features)
            result_features["selection_rank"] = idx
            final.append(
                SymbolSelectionResult(
                    symbol=result.symbol,
                    selected=selected,
                    selection_rank=idx,
                    selection_score=result.selection_score,
                    selection_bias=result.selection_bias,
                    reason_codes=result.reason_codes,
                    features=result_features,
                )
            )
        return final


class SelectionEngine:
    """v1.2 all-market selector with separate long and short boards."""

    def __init__(self, config: SymbolSelectionConfig | None = None):
        self.config = config or SymbolSelectionConfig(style="dual_board")

    def rank_all_market(
        self,
        snapshots: dict[str, FeatureSnapshot],
        *,
        reference_symbols: list[str] | tuple[str, ...] = ("BTC-USDT-SWAP", "ETH-USDT-SWAP"),
    ) -> dict[str, list[SymbolSelectionResult]]:
        usable = {symbol: snapshot for symbol, snapshot in snapshots.items() if snapshot is not None}
        reference_set = set(reference_symbols)
        candidates = {symbol: snapshot for symbol, snapshot in usable.items() if symbol not in reference_set}
        if not candidates:
            return {"long_main_wave": [], "short_waterfall": []}

        btc_features = usable.get("BTC-USDT-SWAP").features if usable.get("BTC-USDT-SWAP") else {}
        eth_features = usable.get("ETH-USDT-SWAP").features if usable.get("ETH-USDT-SWAP") else {}
        market_env = {
            "btc_ret_20d": _float(btc_features.get("ret_20d")),
            "btc_ret_60d": _float(btc_features.get("ret_60d")),
            "eth_ret_20d": _float(eth_features.get("ret_20d")),
            "eth_ret_60d": _float(eth_features.get("ret_60d")),
        }

        ret_3_pctiles = _percentiles({symbol: _float(snapshot.features.get("ret_3d")) for symbol, snapshot in candidates.items()})
        ret_7_pctiles = _percentiles({symbol: _float(snapshot.features.get("ret_7d")) for symbol, snapshot in candidates.items()})
        ret_20_pctiles = _percentiles({symbol: _float(snapshot.features.get("ret_20d")) for symbol, snapshot in candidates.items()})
        ret_60_pctiles = _percentiles({symbol: _float(snapshot.features.get("ret_60d")) for symbol, snapshot in candidates.items()})
        vol_pctiles = _percentiles({symbol: _float(snapshot.features.get("vol_ratio_20d"), 1.0) for symbol, snapshot in candidates.items()})
        liquidity_scores = {symbol: _liquidity_score(snapshot.features) for symbol, snapshot in candidates.items()}
        oi_change_values = {
            symbol: _float(snapshot.features.get("oi_change_3d"))
            for symbol, snapshot in candidates.items()
        }
        oi_change_pctiles = _percentiles(oi_change_values)
        rel_btc_values = {
            symbol: _float(snapshot.features.get("ret_20d")) - market_env["btc_ret_20d"]
            for symbol, snapshot in candidates.items()
        }
        rel_eth_values = {
            symbol: _float(snapshot.features.get("ret_20d")) - market_env["eth_ret_20d"]
            for symbol, snapshot in candidates.items()
        }
        rel_btc_pctiles = _percentiles(rel_btc_values)
        rel_eth_pctiles = _percentiles(rel_eth_values)
        selection_profile = self.config.scoring_profile or "enhanced"
        native_profile = _is_native_selection_profile(selection_profile)

        long_rows: list[SymbolSelectionResult] = []
        short_rows: list[SymbolSelectionResult] = []
        for symbol, snapshot in candidates.items():
            features = snapshot.features
            ret_3d = _float(features.get("ret_3d"))
            ret_7d = _float(features.get("ret_7d"))
            ret_20d = _float(features.get("ret_20d"))
            ret_60d = _float(features.get("ret_60d"))
            pos_20d = _clamp(_float(features.get("pos_20d"), 0.5))
            pullback = _float(features.get("pullback_from_20d_high"))
            vol_ratio = _float(features.get("vol_ratio_20d"), 1.0)
            rel_btc = rel_btc_values[symbol]
            rel_eth = rel_eth_values[symbol]
            liquidity_score = liquidity_scores[symbol]
            oi_change_3d = oi_change_values[symbol]
            funding_rate_last = _float(features.get("funding_rate_last"))
            pullback_quality = _pullback_quality(pullback, pos_20d)
            breakout_quality = 1.0 if pos_20d >= 0.82 else (0.65 if pos_20d >= 0.65 else 0.35)
            breakdown_quality = 1.0 if pos_20d <= 0.18 else (0.65 if pos_20d <= 0.35 else 0.35)
            filter_codes = _long_filter_codes(
                ret_20d=ret_20d,
                ret_60d=ret_60d,
                pos_20d=pos_20d,
                pullback=pullback,
                vol_ratio=vol_ratio,
            )
            if not native_profile:
                filter_codes.extend(_auxiliary_filter_codes(features))
            penalty = _long_filter_penalty(filter_codes)
            if native_profile:
                long_score = _clamp(
                    0.16 * ret_3_pctiles[symbol]
                    + 0.20 * ret_7_pctiles[symbol]
                    + 0.22 * ret_20_pctiles[symbol]
                    + 0.14 * ret_60_pctiles[symbol]
                    + 0.16 * rel_btc_pctiles[symbol]
                    + 0.06 * rel_eth_pctiles[symbol]
                    + 0.04 * breakout_quality
                    + 0.02 * pullback_quality
                    - penalty
                )
                short_score = _clamp(
                    0.18 * (1.0 - ret_3_pctiles[symbol])
                    + 0.20 * (1.0 - ret_7_pctiles[symbol])
                    + 0.22 * (1.0 - ret_20_pctiles[symbol])
                    + 0.14 * (1.0 - ret_60_pctiles[symbol])
                    + 0.18 * (1.0 - rel_btc_pctiles[symbol])
                    + 0.06 * (1.0 - rel_eth_pctiles[symbol])
                    + 0.02 * breakdown_quality
                )
            else:
                long_score = _clamp(
                    0.14 * ret_3_pctiles[symbol]
                    + 0.18 * ret_7_pctiles[symbol]
                    + 0.20 * ret_20_pctiles[symbol]
                    + 0.12 * ret_60_pctiles[symbol]
                    + 0.14 * rel_btc_pctiles[symbol]
                    + 0.06 * rel_eth_pctiles[symbol]
                    + 0.08 * breakout_quality
                    + 0.04 * pullback_quality
                    + 0.04 * vol_pctiles[symbol]
                    + 0.04 * liquidity_score
                    + 0.05 * oi_change_pctiles[symbol]
                    - penalty
                )
                short_score = _clamp(
                    0.16 * (1.0 - ret_3_pctiles[symbol])
                    + 0.18 * (1.0 - ret_7_pctiles[symbol])
                    + 0.20 * (1.0 - ret_20_pctiles[symbol])
                    + 0.12 * (1.0 - ret_60_pctiles[symbol])
                    + 0.16 * (1.0 - rel_btc_pctiles[symbol])
                    + 0.06 * (1.0 - rel_eth_pctiles[symbol])
                    + 0.08 * breakdown_quality
                    + 0.04 * vol_pctiles[symbol]
                    + 0.03 * liquidity_score
                    + 0.03 * oi_change_pctiles[symbol]
                )
            common_features = {
                "selection_ret_3d": ret_3d,
                "selection_ret_7d": ret_7d,
                "selection_ret_20d": ret_20d,
                "selection_ret_60d": ret_60d,
                "selection_pos_20d": pos_20d,
                "selection_pullback_from_20d_high": pullback,
                "selection_vol_ratio_20d": vol_ratio,
                "relative_to_btc_20d": rel_btc,
                "relative_to_eth_20d": rel_eth,
                "ret_3d_pctile": ret_3_pctiles[symbol],
                "ret_7d_pctile": ret_7_pctiles[symbol],
                "ret_20d_pctile": ret_20_pctiles[symbol],
                "ret_60d_pctile": ret_60_pctiles[symbol],
                "vol_ratio_20d_pctile": vol_pctiles[symbol],
                "relative_to_btc_20d_pctile": rel_btc_pctiles[symbol],
                "relative_to_eth_20d_pctile": rel_eth_pctiles[symbol],
                "liquidity_score": liquidity_score,
                "turnover_rank": features.get("turnover_rank", ""),
                "turnover_rank_top_n": features.get("turnover_rank_top_n", ""),
                "turnover_usdt": features.get("turnover_usdt", ""),
                "funding_rate_last": funding_rate_last,
                "oi_change_3d": oi_change_3d,
                "oi_change_3d_pctile": oi_change_pctiles[symbol],
                "long_selection_score": long_score,
                "short_selection_score": short_score,
            }
            upside_space = _float(features.get("upside_space_pct"), 0.0)
            long_reasons = _long_reason_codes(
                ret_3d=ret_3d,
                ret_7d=ret_7d,
                ret_20d=ret_20d,
                rel_btc=rel_btc,
                pos_20d=pos_20d,
                pullback=pullback,
                vol_ratio=vol_ratio,
            )
            long_reasons = _with_v1_3_long_selection_codes(
                long_reasons,
                features=features,
                ret_20d=ret_20d,
                ret_60d=ret_60d,
                rel_btc=rel_btc,
                rel_eth=rel_eth,
                pos_20d=pos_20d,
                pullback=pullback,
                vol_ratio=vol_ratio,
                upside_space=upside_space,
            )
            if not native_profile:
                long_reasons = _with_auxiliary_long_selection_codes(
                    long_reasons,
                    features=features,
                    liquidity_score=liquidity_score,
                    oi_change_3d=oi_change_3d,
                    funding_rate_last=funding_rate_last,
                )
            long_tag = _long_selection_tag(long_reasons)
            long_filter_codes = list(filter_codes)
            if long_tag == "catch_up_short_hold":
                long_filter_codes.append("not_leader_catch_up")
            if long_tag == "leader_altcoin":
                long_score = _clamp(long_score + 0.04)
            elif long_tag == "catch_up_short_hold":
                long_score = _clamp(long_score - 0.04)
            short_reasons = _short_reason_codes(
                ret_3d=ret_3d,
                ret_7d=ret_7d,
                ret_20d=ret_20d,
                rel_btc=rel_btc,
                pos_20d=pos_20d,
                vol_ratio=vol_ratio,
                features=features,
            )
            short_reasons = _with_v1_3_short_selection_codes(short_reasons, features=features)
            if not native_profile:
                short_reasons = _with_auxiliary_short_selection_codes(
                    short_reasons,
                    features=features,
                    liquidity_score=liquidity_score,
                    oi_change_3d=oi_change_3d,
                    funding_rate_last=funding_rate_last,
                )
            profile_long = _profile_score_adjustment(
                selection_profile,
                side="long",
                features=features,
                ret_20_pct=ret_20_pctiles[symbol],
                ret_60_pct=ret_60_pctiles[symbol],
                rel_btc_pct=rel_btc_pctiles[symbol],
                vol_pct=vol_pctiles[symbol],
                liquidity_score=liquidity_score,
                oi_pct=oi_change_pctiles[symbol],
                pullback_quality=pullback_quality,
                breakout_quality=breakout_quality,
                breakdown_quality=breakdown_quality,
                pos_20d=pos_20d,
                pullback=pullback,
                funding_rate_last=funding_rate_last,
                long_tag=long_tag,
                filter_codes=long_filter_codes,
            )
            profile_short = _profile_score_adjustment(
                selection_profile,
                side="short",
                features=features,
                ret_20_pct=ret_20_pctiles[symbol],
                ret_60_pct=ret_60_pctiles[symbol],
                rel_btc_pct=rel_btc_pctiles[symbol],
                vol_pct=vol_pctiles[symbol],
                liquidity_score=liquidity_score,
                oi_pct=oi_change_pctiles[symbol],
                pullback_quality=pullback_quality,
                breakout_quality=breakout_quality,
                breakdown_quality=breakdown_quality,
                pos_20d=pos_20d,
                pullback=pullback,
                funding_rate_last=funding_rate_last,
                long_tag=long_tag,
                filter_codes=long_filter_codes,
            )
            long_score = _clamp(long_score + profile_long["delta"])
            short_score = _clamp(short_score + profile_short["delta"])
            long_reasons = _dedupe([*long_reasons, *profile_long["reason_codes"]])
            short_reasons = _dedupe([*short_reasons, *profile_short["reason_codes"]])
            long_filter_codes = _dedupe([*long_filter_codes, *profile_long["filter_codes"]])
            short_filter_codes = profile_short["filter_codes"]
            common_features["long_selection_score"] = long_score
            common_features["short_selection_score"] = short_score
            common_features["selection_profile"] = selection_profile
            common_features["selection_profile_delta_long"] = profile_long["delta"]
            common_features["selection_profile_delta_short"] = profile_short["delta"]
            long_rows.append(
                SymbolSelectionResult(
                    symbol=symbol,
                    selected=False,
                    selection_rank=0,
                    selection_score=long_score,
                    selection_bias="long",
                    reason_codes=long_reasons,
                    features={
                        **common_features,
                        "selection_mode": "long_main_wave",
                        "selection_score": long_score,
                        "symbol_selection_tag": long_tag,
                        "selection_reason_codes": long_reasons,
                    },
                    selection_mode="long_main_wave",
                    market_env=market_env,
                    filter_codes=long_filter_codes,
                )
            )
            short_rows.append(
                SymbolSelectionResult(
                    symbol=symbol,
                    selected=False,
                    selection_rank=0,
                    selection_score=short_score,
                    selection_bias="short",
                    reason_codes=short_reasons,
                    features={
                        **common_features,
                        "selection_mode": "short_waterfall",
                        "selection_score": short_score,
                        "symbol_selection_tag": "short_waterfall",
                        "selection_reason_codes": short_reasons,
                    },
                    selection_mode="short_waterfall",
                    market_env=market_env,
                    filter_codes=short_filter_codes,
                )
            )

        return {
            "long_main_wave": _finalize_board(
                long_rows,
                top_n=self.config.long_top_n,
                min_score=self.config.min_long_score,
                required_structure="long_main_wave",
            ),
            "short_waterfall": _finalize_board(
                short_rows,
                top_n=self.config.short_top_n,
                min_score=self.config.min_short_score,
                required_structure="short_waterfall",
            ),
        }


class HistoricalSymbolSelectionAnalyzer:
    def __init__(
        self,
        kline_cache_dir: str | Path,
        *,
        selector: SymbolSelector | None = None,
        config: SymbolSelectionConfig | None = None,
    ):
        self.kline_cache_dir = Path(kline_cache_dir)
        self.config = config or SymbolSelectionConfig(enabled=True, top_n=20)
        self.selector = selector or SymbolSelector(self.config)
        self.feature_builder = DailyFeatureBuilder()

    def run(self, trades_csv: str | Path, out_dir: str | Path) -> dict[str, Any]:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        candles_by_symbol = _load_daily_cache(self.kline_cache_dir)
        trades = _load_trades(Path(trades_csv))
        snapshots_cache: dict[int, dict[str, FeatureSnapshot]] = {}
        ranking_cache: dict[int, dict[str, SymbolSelectionResult]] = {}
        history_by_symbol: dict[str, list[dict[str, Any]]] = {}
        rows: list[dict[str, Any]] = []

        for trade in sorted(trades, key=lambda row: row["entry_ts"]):
            entry_ts = int(trade["entry_ts"])
            day_key = _day_floor(entry_ts)
            if day_key not in snapshots_cache:
                snapshots_cache[day_key] = self._build_snapshots(candles_by_symbol, entry_ts)
                ranking_cache[day_key] = {
                    result.symbol: result
                    for result in self.selector.rank(
                        snapshots_cache[day_key],
                        top_n=self.config.top_n,
                        min_score=self.config.min_score,
                    )
                }
            ranking = ranking_cache[day_key]
            result = ranking.get(trade["symbol"])
            prior = _prior_trade_context(history_by_symbol.get(trade["symbol"], []), entry_ts)
            row = _selection_row(trade, result, ranking, prior)
            rows.append(row)
            history_by_symbol.setdefault(trade["symbol"], []).append(trade)

        features_path = out_path / "symbol_selection_features.csv"
        summary_path = out_path / "symbol_selection_summary.csv"
        report_path = out_path / "symbol_selection_report.md"
        _write_rows(features_path, rows)
        summary_rows = _summary_rows(rows)
        _write_rows(summary_path, summary_rows)
        report_path.write_text(_build_report(rows, summary_rows, candles_by_symbol), encoding="utf-8")
        return {
            "trades": len(rows),
            "symbols": len({row["symbol"] for row in rows}),
            "cached_symbols": len(candles_by_symbol),
            "features_path": str(features_path),
            "summary_path": str(summary_path),
            "report_path": str(report_path),
        }

    def _build_snapshots(
        self,
        candles_by_symbol: dict[str, list[Candle]],
        entry_ts: int,
    ) -> dict[str, FeatureSnapshot]:
        snapshots: dict[str, FeatureSnapshot] = {}
        completed_daily_cutoff = _day_floor(entry_ts)
        for symbol, rows in candles_by_symbol.items():
            prefix = [row for row in rows if row.ts < completed_daily_cutoff]
            if len(prefix) < self.config.min_daily_bars:
                continue
            snapshot = self.feature_builder.build(symbol, prefix)
            if snapshot is not None:
                snapshots[symbol] = snapshot
        return snapshots


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze LangLang historical symbol selection")
    parser.add_argument("--trades", required=True)
    parser.add_argument("--kline-cache", default="output/langlang_distill/kline_cache")
    parser.add_argument("--out", required=True)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--min-score", type=float, default=0.0)
    args = parser.parse_args(argv)

    config = SymbolSelectionConfig(enabled=True, top_n=args.top_n, min_score=args.min_score)
    result = HistoricalSymbolSelectionAnalyzer(args.kline_cache, config=config).run(args.trades, args.out)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


def _reason_codes(
    *,
    bias: str,
    ret_20_pct: float,
    ret_60_pct: float,
    pos_20d: float,
    vol_pct: float,
    vol_ratio: float,
    pullback: float,
    features: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    relative_to_btc = features.get("relative_to_btc_20d")
    if bias == "long":
        if ret_20_pct >= 0.75:
            reasons.append("relative_strength_top_quartile")
        if ret_60_pct >= 0.75:
            reasons.append("medium_trend_leader")
        if pos_20d >= 0.75:
            reasons.append("near_range_high_or_breakout")
        if -0.16 <= pullback <= -0.015:
            reasons.append("strong_pullback_candidate")
        if _float(relative_to_btc) > 0:
            reasons.append("outperforming_btc_20d")
    else:
        if ret_20_pct <= 0.25:
            reasons.append("relative_weakness_bottom_quartile")
        if ret_60_pct <= 0.25:
            reasons.append("medium_trend_laggard")
        if pos_20d <= 0.25:
            reasons.append("low_range_breakdown")
        if _float(relative_to_btc) < 0:
            reasons.append("underperforming_btc_20d")
    if vol_pct >= 0.65 and vol_ratio >= 1.05:
        reasons.append("volume_expansion")
    if not reasons:
        reasons.append("active_but_no_dominant_cross_section_reason")
    return reasons


def _is_native_selection_profile(profile: str) -> bool:
    normalized = (profile or "").lower()
    return normalized == "native" or normalized.startswith("langlang_01")


def _profile_score_adjustment(
    profile: str,
    *,
    side: str,
    features: dict[str, Any],
    ret_20_pct: float,
    ret_60_pct: float,
    rel_btc_pct: float,
    vol_pct: float,
    liquidity_score: float,
    oi_pct: float,
    pullback_quality: float,
    breakout_quality: float,
    breakdown_quality: float,
    pos_20d: float,
    pullback: float,
    funding_rate_last: float,
    long_tag: str,
    filter_codes: list[str],
) -> dict[str, Any]:
    normalized = (profile or "enhanced").lower()
    delta = 0.0
    reasons: list[str] = []
    filters: list[str] = []
    upside_space = _float(features.get("upside_space_pct"))

    if side == "long":
        if normalized in {"langlang_01_select", "langlang_plus_01_select"}:
            delta += 0.05 * rel_btc_pct + 0.04 * ret_20_pct + 0.03 * ret_60_pct
            if long_tag == "leader_altcoin":
                delta += 0.04
                reasons.append("profile_select_leader_priority")
            else:
                delta -= 0.08
                filters.append("profile_select_not_leader")
            if upside_space > 0:
                delta += 0.04 if upside_space >= 0.25 else -0.04
        elif normalized in {"langlang_01_entry", "langlang_plus_01_entry"}:
            delta += 0.08 * pullback_quality + 0.06 * breakout_quality
            if -0.20 <= pullback <= -0.015:
                reasons.append("profile_entry_retest_priority")
            else:
                delta -= 0.04
                filters.append("profile_entry_retest_weak")
        elif normalized in {"langlang_01_exit", "langlang_plus_01_exit"}:
            delta += 0.05 * ret_60_pct + 0.04 * rel_btc_pct
            if long_tag == "leader_altcoin":
                delta += 0.03
                reasons.append("profile_exit_runner_candidate")
            if upside_space > 0:
                delta += 0.03 if upside_space >= 0.30 else -0.02
        elif normalized in {"langlang_01_risk", "langlang_plus_01_loss"}:
            if any(code in filter_codes for code in {"high_position_no_structure", "chase_overheat", "first_10x_too_high"}):
                delta -= 0.12
                filters.append("profile_risk_high_position_cut")
            if pos_20d >= 0.94 and pullback > -0.02:
                delta -= 0.06
                filters.append("profile_risk_no_retest_cut")

        if normalized.startswith("langlang_plus"):
            delta += 0.04 * liquidity_score + 0.04 * vol_pct + 0.03 * oi_pct
            if funding_rate_last >= 0.012:
                delta -= 0.12
                filters.append("profile_plus_funding_heat_cut")
            elif 0 <= funding_rate_last <= 0.004:
                reasons.append("profile_plus_funding_ok")
            if normalized == "langlang_plus_01_loss":
                if liquidity_score < 0.35:
                    delta -= 0.10
                    filters.append("profile_loss_low_liquidity_cut")
                if oi_pct < 0.25:
                    delta -= 0.05
                    filters.append("profile_loss_oi_not_confirmed")

    elif side == "short":
        if normalized in {"langlang_01_select", "langlang_plus_01_select"}:
            delta += 0.04 * (1.0 - rel_btc_pct) + 0.04 * (1.0 - ret_20_pct)
            if breakdown_quality >= 0.65:
                reasons.append("profile_select_waterfall_priority")
        elif normalized in {"langlang_01_entry", "langlang_plus_01_entry"}:
            delta += 0.08 * breakdown_quality
            if breakdown_quality < 0.65:
                delta -= 0.04
                filters.append("profile_entry_breakdown_weak")
        elif normalized in {"langlang_01_risk", "langlang_plus_01_loss"}:
            delta += 0.05 * breakdown_quality
            if funding_rate_last <= -0.006:
                delta -= 0.06
                filters.append("profile_loss_short_crowding_cut")
            if liquidity_score < 0.35:
                delta -= 0.08
                filters.append("profile_loss_low_liquidity_cut")
        elif normalized in {"langlang_01_exit", "langlang_plus_01_exit"}:
            delta += 0.03 * (1.0 - ret_60_pct) + 0.03 * breakdown_quality

        if normalized.startswith("langlang_plus"):
            delta += 0.03 * liquidity_score + 0.03 * vol_pct

    return {
        "delta": _clamp(delta, -0.30, 0.30),
        "reason_codes": reasons,
        "filter_codes": filters,
    }


def _finalize_board(
    rows: list[SymbolSelectionResult],
    *,
    top_n: int,
    min_score: float,
    required_structure: str | None = None,
) -> list[SymbolSelectionResult]:
    ranked = sorted(rows, key=lambda result: (-result.selection_score, result.symbol))
    final: list[SymbolSelectionResult] = []
    for rank, result in enumerate(ranked, start=1):
        structure_ok = _has_required_structure(result, required_structure)
        selected = top_n > 0 and result.selection_score >= min_score and rank <= top_n and structure_ok
        features = dict(result.features)
        features["selection_rank"] = rank
        filter_codes = list(result.filter_codes)
        if not structure_ok and required_structure == "long_main_wave":
            filter_codes.append("incomplete_long_main_wave_structure")
        elif not structure_ok and required_structure == "short_waterfall":
            filter_codes.append("incomplete_short_waterfall_structure")
        final.append(
            SymbolSelectionResult(
                symbol=result.symbol,
                selected=selected,
                selection_rank=rank,
                selection_score=result.selection_score,
                selection_bias=result.selection_bias,
                reason_codes=result.reason_codes,
                features=features,
                selection_mode=result.selection_mode,
                market_env=result.market_env,
                filter_codes=filter_codes,
                data_status=result.data_status,
                unavailable_reason=result.unavailable_reason,
            )
        )
    return final


def _has_required_structure(result: SymbolSelectionResult, required_structure: str | None) -> bool:
    if required_structure is None:
        return True
    reasons = set(result.reason_codes)
    if required_structure == "long_main_wave":
        if "main_wave_acceleration" in reasons:
            return True
        if "short_term_reclaim" in reasons and "breakout_retest_quality" in reasons:
            return "relative_to_btc_strength" in reasons or "volume_expansion" in reasons
        if "near_high_or_breakout" in reasons and "relative_to_btc_strength" in reasons:
            return "short_term_reclaim" in reasons or "volume_expansion" in reasons
        return False
    if required_structure == "short_waterfall":
        if "waterfall_breakdown" in reasons:
            return True
        return (
            "downside_acceleration" in reasons
            and "relative_to_btc_weakness" in reasons
            and "ma_stack_down" in reasons
        )
    return True


def _pullback_quality(pullback: float, pos_20d: float) -> float:
    if -0.12 <= pullback <= -0.015:
        return 1.0
    if -0.20 <= pullback < -0.12:
        return 0.55
    if pullback > -0.015 and pos_20d < 0.94:
        return 0.65
    return 0.25


def _long_filter_codes(
    *,
    ret_20d: float,
    ret_60d: float,
    pos_20d: float,
    pullback: float,
    vol_ratio: float,
) -> list[str]:
    filters: list[str] = []
    if vol_ratio < 0.20:
        filters.append("low_liquidity")
    if ret_20d >= 1.20 and pos_20d >= 0.95 and pullback > -0.01:
        filters.append("chase_overheat")
    if ret_60d >= 2.50 and ret_20d >= 1.00 and pos_20d >= 0.95 and pullback > -0.015:
        filters.append("first_10x_too_high")
    if pos_20d >= 0.96 and pullback > -0.015:
        filters.append("high_position_no_structure")
    return filters


def _long_filter_penalty(filter_codes: list[str]) -> float:
    penalty = 0.0
    if "low_liquidity" in filter_codes:
        penalty += 0.18
    if "liquidity_rank_filtered" in filter_codes:
        penalty += 0.24
    if "funding_overheated" in filter_codes:
        penalty += 0.48
    if "high_position_no_structure" in filter_codes:
        penalty += 0.16
    if "chase_overheat" in filter_codes:
        penalty += 0.28
    if "first_10x_too_high" in filter_codes:
        penalty += 0.22
    return penalty


def _auxiliary_filter_codes(features: dict[str, Any]) -> list[str]:
    filters: list[str] = []
    rank = _float(features.get("turnover_rank"))
    top_n = _float(features.get("turnover_rank_top_n"), 200.0)
    if rank > 0 and top_n > 0 and rank > top_n:
        filters.append("liquidity_rank_filtered")
    funding_rate = _float(features.get("funding_rate_last"))
    if funding_rate >= 0.015:
        filters.append("funding_overheated")
    return filters


def _liquidity_score(features: dict[str, Any]) -> float:
    rank = _float(features.get("turnover_rank"))
    top_n = _float(features.get("turnover_rank_top_n"), 200.0)
    if rank > 0 and top_n > 0:
        return _clamp(1.0 - ((rank - 1.0) / top_n))
    turnover = _float(features.get("turnover_usdt"))
    if turnover <= 0:
        return 0.35
    if turnover >= 100_000_000:
        return 1.0
    if turnover >= 50_000_000:
        return 0.85
    if turnover >= 10_000_000:
        return 0.65
    if turnover >= 3_000_000:
        return 0.40
    return 0.20


def _with_auxiliary_long_selection_codes(
    reasons: list[str],
    *,
    features: dict[str, Any],
    liquidity_score: float,
    oi_change_3d: float,
    funding_rate_last: float,
) -> list[str]:
    result = list(reasons)
    rank = _float(features.get("turnover_rank"))
    top_n = _float(features.get("turnover_rank_top_n"), 200.0)
    if (rank > 0 and top_n > 0 and rank <= top_n) or liquidity_score >= 0.75:
        result.append("liquid_top200_turnover")
    if oi_change_3d >= 0.08:
        result.append("oi_expansion_confirmation")
    if 0 <= funding_rate_last <= 0.003:
        result.append("funding_not_overheated")
    if funding_rate_last >= 0.015:
        result.append("funding_overheated_crowding")
    if _float(features.get("listing_age_days")) and _float(features.get("listing_age_days")) <= 180:
        result.append("new_listing_attention")
    return _dedupe(result)


def _with_auxiliary_short_selection_codes(
    reasons: list[str],
    *,
    features: dict[str, Any],
    liquidity_score: float,
    oi_change_3d: float,
    funding_rate_last: float,
) -> list[str]:
    result = list(reasons)
    rank = _float(features.get("turnover_rank"))
    top_n = _float(features.get("turnover_rank_top_n"), 200.0)
    if (rank > 0 and top_n > 0 and rank <= top_n) or liquidity_score >= 0.75:
        result.append("liquid_top200_turnover")
    if oi_change_3d < -0.05:
        result.append("oi_contraction_waterfall")
    elif oi_change_3d > 0.10:
        result.append("short_crowding_or_forced_move")
    if funding_rate_last <= -0.003:
        result.append("negative_funding_short_heat")
    return _dedupe(result)


def _long_reason_codes(
    *,
    ret_3d: float,
    ret_7d: float,
    ret_20d: float,
    rel_btc: float,
    pos_20d: float,
    pullback: float,
    vol_ratio: float,
) -> list[str]:
    reasons: list[str] = []
    if ret_7d > 0.08 and ret_20d > 0.20:
        reasons.append("main_wave_acceleration")
    if ret_3d > 0 and ret_7d > 0:
        reasons.append("short_term_reclaim")
    if rel_btc > 0:
        reasons.append("relative_to_btc_strength")
    if pos_20d >= 0.78:
        reasons.append("near_high_or_breakout")
    if -0.16 <= pullback <= -0.015:
        reasons.append("breakout_retest_quality")
    if vol_ratio >= 1.20:
        reasons.append("volume_expansion")
    if not reasons:
        reasons.append("long_watch_without_complete_main_wave")
    return reasons


def _with_v1_3_long_selection_codes(
    reasons: list[str],
    *,
    features: dict[str, Any],
    ret_20d: float,
    ret_60d: float,
    rel_btc: float,
    rel_eth: float,
    pos_20d: float,
    pullback: float,
    vol_ratio: float,
    upside_space: float,
) -> list[str]:
    result = list(reasons)
    if bool(features.get("btc_first_wave_follow")):
        result.append("btc_first_wave_follow")
    if bool(features.get("btc_contraction_resilient")):
        result.append("btc_contraction_resilient")
    if upside_space >= 0.18:
        result.append("upside_space_large")
    if (
        ret_20d >= 0.25
        and ret_60d >= 0.45
        and rel_btc > 0
        and rel_eth >= -0.05
        and pos_20d >= 0.65
        and -0.16 <= pullback <= -0.01
        and vol_ratio >= 1.15
    ):
        result.append("leader_altcoin")
    if (
        bool(features.get("btc_divergence_alt_rotation"))
        or (ret_20d > 0.18 and ret_60d < 0.20 and pullback > -0.02)
    ):
        result.append("catch_up_short_hold")
    return _dedupe(result)


def _long_selection_tag(reason_codes: list[str]) -> str:
    if "leader_altcoin" in reason_codes:
        return "leader_altcoin"
    if "catch_up_short_hold" in reason_codes:
        return "catch_up_short_hold"
    return "long_watch"


def _with_v1_3_short_selection_codes(reasons: list[str], *, features: dict[str, Any]) -> list[str]:
    result = list(reasons)
    if bool(features.get("failed_rebound_below_platform")):
        result.append("failed_rebound_below_platform")
    if bool(features.get("new_listing_contraction_breakdown")):
        result.append("new_listing_contraction_breakdown")
    return _dedupe(result)


def _short_reason_codes(
    *,
    ret_3d: float,
    ret_7d: float,
    ret_20d: float,
    rel_btc: float,
    pos_20d: float,
    vol_ratio: float,
    features: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    ma_5 = _float(features.get("ma_5"))
    ma_20 = _float(features.get("ma_20"))
    latest_close = _float(features.get("latest_close"))
    if pos_20d <= 0.25 or ret_20d < -0.15:
        reasons.append("waterfall_breakdown")
    if ret_3d < 0 and ret_7d < 0:
        reasons.append("downside_acceleration")
    if rel_btc < 0:
        reasons.append("relative_to_btc_weakness")
    if latest_close > 0 and ma_5 > 0 and ma_20 > 0 and latest_close <= ma_20 and ma_5 <= ma_20:
        reasons.append("ma_stack_down")
    if vol_ratio >= 1.20:
        reasons.append("volume_downmove")
    if not reasons:
        reasons.append("short_watch_without_complete_waterfall")
    return reasons


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def _selection_row(
    trade: dict[str, Any],
    result: SymbolSelectionResult | None,
    ranking: dict[str, SymbolSelectionResult],
    prior: dict[str, Any],
) -> dict[str, Any]:
    btc = ranking.get("BTC-USDT-SWAP")
    eth = ranking.get("ETH-USDT-SWAP")
    if result is None:
        reason_codes = ["selection_data_missing"]
        features: dict[str, Any] = {}
        selected = False
        rank = ""
        score = ""
        bias = ""
    else:
        reason_codes = list(result.reason_codes)
        features = result.features
        selected = result.selected
        rank = result.selection_rank
        score = result.selection_score
        bias = result.selection_bias
    side = str(trade.get("side", "")).lower()
    side_matches_bias = bool(side and bias and side == bias)
    if bias and side and not side_matches_bias:
        reason_codes.append("trade_side_against_selection_bias")
    if prior["prior_trades_7d"] > 0 or prior["prior_trades_30d"] >= 2:
        reason_codes.append("recent_trade_attention")
    symbol_ret_20d = _float(features.get("selection_ret_20d"))
    btc_ret_20d = _float(btc.features.get("selection_ret_20d")) if btc else 0.0
    eth_ret_20d = _float(eth.features.get("selection_ret_20d")) if eth else 0.0
    return {
        "trade_id": trade.get("trade_id", ""),
        "symbol": trade["symbol"],
        "side": trade.get("side", ""),
        "entry_time": trade.get("entry_time", ""),
        "selected": selected,
        "selection_rank": rank,
        "selection_score": score,
        "selection_bias": bias,
        "side_matches_selection_bias": side_matches_bias,
        "reason_codes": "|".join(reason_codes),
        "ret_20d": symbol_ret_20d,
        "ret_60d": _float(features.get("selection_ret_60d")),
        "pos_20d": _float(features.get("selection_pos_20d"), 0.5),
        "pullback_from_20d_high": _float(features.get("selection_pullback_from_20d_high")),
        "vol_ratio_20d": _float(features.get("selection_vol_ratio_20d"), 1.0),
        "ret_20d_pctile": _float(features.get("ret_20d_pctile")),
        "ret_60d_pctile": _float(features.get("ret_60d_pctile")),
        "pos_20d_pctile": _float(features.get("pos_20d_pctile"), 0.5),
        "vol_ratio_20d_pctile": _float(features.get("vol_ratio_20d_pctile")),
        "long_selection_score": _float(features.get("long_selection_score")),
        "short_selection_score": _float(features.get("short_selection_score")),
        "btc_ret_20d": btc_ret_20d,
        "eth_ret_20d": eth_ret_20d,
        "relative_to_btc_20d": symbol_ret_20d - btc_ret_20d,
        "prior_trades_7d": prior["prior_trades_7d"],
        "prior_trades_30d": prior["prior_trades_30d"],
        "prior_pnl_30d": prior["prior_pnl_30d"],
        "pnl_usdt": trade.get("pnl_usdt", ""),
        "return_rate": trade.get("return_rate", ""),
    }


def _summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["symbol"]), []).append(row)
    summary: list[dict[str, Any]] = []
    for symbol, symbol_rows in sorted(grouped.items()):
        scores = [_float(row.get("selection_score")) for row in symbol_rows if row.get("selection_score") != ""]
        selected_count = sum(1 for row in symbol_rows if row.get("selected") in {True, "True", "true", "1", 1})
        reason_counts: dict[str, int] = {}
        for row in symbol_rows:
            for reason in str(row.get("reason_codes", "")).split("|"):
                if reason:
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
        top_reasons = ",".join(
            reason
            for reason, _count in sorted(reason_counts.items(), key=lambda item: (-item[1], item[0]))[:5]
        )
        summary.append(
            {
                "symbol": symbol,
                "trade_count": len(symbol_rows),
                "selected_rate": selected_count / len(symbol_rows) if symbol_rows else 0.0,
                "side_bias_match_rate": (
                    sum(
                        1
                        for row in symbol_rows
                        if row.get("side_matches_selection_bias") in {True, "True", "true", "1", 1}
                    )
                    / len(symbol_rows)
                    if symbol_rows
                    else 0.0
                ),
                "avg_selection_score": fmean(scores) if scores else 0.0,
                "avg_ret_20d": fmean(_float(row.get("ret_20d")) for row in symbol_rows),
                "avg_relative_to_btc_20d": fmean(_float(row.get("relative_to_btc_20d")) for row in symbol_rows),
                "top_reasons": top_reasons,
            }
        )
    return summary


def _build_report(
    rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    candles_by_symbol: dict[str, list[Candle]],
) -> str:
    selected_rate = (
        sum(1 for row in rows if row.get("selected") in {True, "True", "true", "1", 1}) / len(rows)
        if rows
        else 0.0
    )
    side_bias_match_rate = (
        sum(1 for row in rows if row.get("side_matches_selection_bias") in {True, "True", "true", "1", 1}) / len(rows)
        if rows
        else 0.0
    )
    reason_counts: dict[str, int] = {}
    for row in rows:
        for reason in str(row.get("reason_codes", "")).split("|"):
            if reason:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
    top_reasons = sorted(reason_counts.items(), key=lambda item: (-item[1], item[0]))[:12]
    top_symbols = sorted(summary_rows, key=lambda row: (-_float(row["avg_selection_score"]), str(row["symbol"])))[:20]
    lines = [
        "# LangLang Symbol Selection Analysis",
        "",
        f"- trades: {len(rows)}",
        f"- trade symbols: {len(summary_rows)}",
        f"- cached daily symbols: {len(candles_by_symbol)}",
        f"- selected rate with configured top_n/min_score: {selected_rate:.2%}",
        f"- trade side matches selection bias: {side_bias_match_rate:.2%}",
        "",
        "## Dominant Reasons",
        "",
    ]
    lines.extend(f"- {reason}: {count}" for reason, count in top_reasons)
    lines.extend(["", "## Highest Average Selection Score Symbols", ""])
    lines.extend(
        (
            f"- {row['symbol']}: score={_float(row['avg_selection_score']):.3f}, "
            f"trades={row['trade_count']}, selected_rate={_float(row['selected_rate']):.2%}, "
            f"side_bias_match={_float(row.get('side_bias_match_rate')):.2%}, "
            f"reasons={row['top_reasons']}"
        )
        for row in top_symbols
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- long-biased selection means the symbol was a cross-sectional strength leader, near a range high, or expanding volume.",
            "- short-biased selection means the symbol was a cross-sectional weakness leader, near a range low, or breaking down with activity.",
            "- recent_trade_attention marks symbols repeatedly traded by the historical account before the current entry.",
            "- selection_data_missing rows are excluded from hard conclusions until more market history is cached.",
            "",
        ]
    )
    return "\n".join(lines)


def _load_daily_cache(cache_dir: Path) -> dict[str, list[Candle]]:
    daily_dir = cache_dir / "1D"
    source_dir = daily_dir if daily_dir.exists() else cache_dir
    rows: dict[str, dict[int, Candle]] = {}
    for path in sorted(source_dir.glob("*.csv")):
        symbol = _symbol_from_cache_path(path)
        candles = _read_cached_candles(path, symbol)
        if candles:
            symbol_rows = rows.setdefault(symbol, {})
            for candle in candles:
                symbol_rows[candle.ts] = candle
    return {symbol: sorted(symbol_rows.values(), key=lambda candle: candle.ts) for symbol, symbol_rows in rows.items()}


def _symbol_from_cache_path(path: Path) -> str:
    return path.stem.split("_", 1)[0]


def _read_cached_candles(path: Path, symbol: str) -> list[Candle]:
    candles: list[Candle] = []
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                candles.append(
                    Candle(
                        symbol=symbol,
                        bar="1D",
                        ts=int(float(row.get("ts") or row.get("timestamp") or 0)),
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=float(row.get("volume") or row.get("vol") or 0.0),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
    return sorted(candles, key=lambda candle: candle.ts)


def _load_trades(path: Path) -> list[dict[str, Any]]:
    trades: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader, start=1):
            symbol = str(row.get("symbol") or row.get("inst_id") or "").strip()
            entry_time = str(row.get("entry_time") or row.get("open_time") or row.get("created_at") or "").strip()
            if not symbol or not entry_time:
                continue
            trade = dict(row)
            trade.setdefault("trade_id", str(idx))
            trade["symbol"] = symbol
            trade["entry_time"] = entry_time
            trade["entry_ts"] = _parse_time_ms(entry_time)
            trades.append(trade)
    return trades


def _parse_time_ms(value: str) -> int:
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    if raw.isdigit():
        number = int(raw)
        return number if number > 10_000_000_000 else number * 1000
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=LOCAL_TZ)
    return int(dt.timestamp() * 1000)


def _day_floor(ts_ms: int) -> int:
    return ts_ms - (ts_ms % 86_400_000)


def _prior_trade_context(prior_rows: list[dict[str, Any]], entry_ts: int) -> dict[str, Any]:
    seven_days = 7 * 86_400_000
    thirty_days = 30 * 86_400_000
    prior_7d = [row for row in prior_rows if 0 < entry_ts - int(row["entry_ts"]) <= seven_days]
    prior_30d = [row for row in prior_rows if 0 < entry_ts - int(row["entry_ts"]) <= thirty_days]
    return {
        "prior_trades_7d": len(prior_7d),
        "prior_trades_30d": len(prior_30d),
        "prior_pnl_30d": sum(_float(row.get("pnl_usdt")) for row in prior_30d),
    }


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _percentiles(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    if len(ordered) == 1:
        return {ordered[0][0]: 0.5}
    denom = len(ordered) - 1
    return {symbol: idx / denom for idx, (symbol, _value) in enumerate(ordered)}


def _float(value: Any, default: float = 0.0) -> float:
    if value in {None, ""}:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


if __name__ == "__main__":
    raise SystemExit(main())

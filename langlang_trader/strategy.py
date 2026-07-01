from __future__ import annotations

from dataclasses import asdict, dataclass

from langlang_trader.features import DailyFeatureBuilder, FeatureSnapshot
from langlang_trader.models import (
    Candle,
    EntrySetup,
    FailureFilter,
    LangLangSignal,
    MarketRegime,
    Side,
    Signal,
    StrategyAction,
    StrategyDecision,
)
from langlang_trader.micro_scalping import (
    MicroScalpVariant,
    RulesFundingBasisShadowStrategy,
    RulesOfiMicropriceScalpStrategy,
    RulesVolatilityBreakoutScalpStrategy,
    RulesVwapMeanReversionScalpStrategy,
)
from langlang_trader.scalping import RulesFiveBarScalpStrategy, ScalpingVariant


@dataclass(frozen=True)
class StrategyVariant:
    variant_id: str = "rules_v01_default"
    ret_20d_min: float = 0.12
    ret_60d_min: float = 0.32
    pos_20d_min: float = 0.45
    max_pullback_pct: float = 0.18
    breakout_tolerance: float = 0.005

    def to_dict(self) -> dict:
        return asdict(self)


def default_variant_grid() -> list[StrategyVariant]:
    variants: list[StrategyVariant] = []
    for ret_20d in (0.18, 0.22, 0.26, 0.30):
        for ret_60d in (0.42, 0.50, 0.65, 0.80):
            for pos_20d in (0.50, 0.60, 0.72):
                for pullback in (0.04, 0.08, 0.12):
                    variants.append(
                        StrategyVariant(
                            variant_id=f"r20_{ret_20d:.2f}_r60_{ret_60d:.2f}_p20_{pos_20d:.2f}_pb_{pullback:.2f}",
                            ret_20d_min=ret_20d,
                            ret_60d_min=ret_60d,
                            pos_20d_min=pos_20d,
                            max_pullback_pct=pullback,
                            breakout_tolerance=0.005,
                        )
                    )
    return variants


@dataclass(frozen=True)
class LangLangV1Variant:
    variant_id: str = "rules_langlang_v1_default"
    ret_20d_min: float = 0.18
    ret_60d_min: float = 0.42
    pos_20d_min: float = 0.55
    max_pullback_pct: float = 0.16
    min_pullback_pct: float = 0.015
    breakout_tolerance: float = 0.008
    overheat_ret_20d: float = 0.55
    overheat_pos_20d: float = 0.92
    overheat_h1_ret_24: float = 0.075
    short_ret_20d_max: float = -0.16
    short_ret_60d_max: float = -0.28
    short_pos_20d_max: float = 0.38
    waterfall_h1_ret_24_max: float = -0.055
    intraday_confirm_ret_min: float = 0.004
    intraday_breakdown_ret_max: float = -0.004
    min_vol_ratio_20d: float = 0.20
    structure_stop_pct: float = 0.08
    partial_take_profit_r: float = 2.0
    partial_exit_fraction: float = 0.5
    runner_take_profit_r: float = 4.0
    time_stop_days: int = 14
    trend_break_buffer_pct: float = 0.01
    historical_match_score: float = 0.50

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class LangLangV1_1Variant(LangLangV1Variant):
    variant_id: str = "rules_langlang_v1_1_default"
    allowed_side: str = "both"
    min_upside_space_pct: float = 0.08
    first_10x_high_pos: float = 0.92
    max_stop_loss_cluster_24h: int = 2
    min_historical_match_score: float = 0.35
    exploratory: bool = False
    high_position_reduce_pos: float = 0.86
    low_position_boost_pos: float = 0.55
    low_position_size_multiplier: float = 1.20
    high_position_size_multiplier: float = 0.60


@dataclass(frozen=True)
class LangLangV1_3Variant(LangLangV1_1Variant):
    variant_id: str = "rules_langlang_v1_3_default"
    enable_countertrend_short: bool = False
    max_small_divergence_count: int = 2
    leader_only_long: bool = True
    catch_up_runner_allowed: bool = False
    autumn_winter_only_best_positions: bool = True
    default_alt_leverage: int = 5
    default_anchor_leverage: int = 10
    entry_family: str = "default"
    experiment_family: str = ""
    strategy_tree_parent_id: str = ""
    strategy_tree_variant_id: str = ""
    payoff_probe_partial_r: float = 1.25
    payoff_probe_runner_r: float = 2.5
    payoff_probe_time_stop_days: int = 5


@dataclass(frozen=True)
class LangLangNativeVariant(LangLangV1_3Variant):
    variant_id: str = "rules_langlang_native_v1_default"


@dataclass(frozen=True)
class LangLangEnhancedVariant(LangLangNativeVariant):
    variant_id: str = "rules_langlang_enhanced_v1_default"
    max_big_loss_overlap_count: int = 0
    max_funding_rate_last: float = 0.015
    max_turnover_rank_24h: int = 200
    min_oi_expansion_3d: float = -0.35


def default_langlang_v1_grid() -> list[LangLangV1Variant]:
    variants: list[LangLangV1Variant] = []
    for ret_20d in (0.16, 0.20, 0.24):
        for ret_60d in (0.38, 0.48, 0.62):
            for pullback in (0.10, 0.16):
                for confirm in (0.003, 0.006):
                    for stop_pct in (0.06, 0.09):
                        variants.append(
                            LangLangV1Variant(
                                variant_id=(
                                    f"llv1_long_r20_{ret_20d:.2f}_r60_{ret_60d:.2f}"
                                    f"_pb_{pullback:.2f}_cf_{confirm:.3f}_st_{stop_pct:.2f}"
                                ),
                                ret_20d_min=ret_20d,
                                ret_60d_min=ret_60d,
                                max_pullback_pct=pullback,
                                intraday_confirm_ret_min=confirm,
                                structure_stop_pct=stop_pct,
                            )
                        )
    for short_ret in (-0.12, -0.16, -0.22):
        for h1_break in (-0.04, -0.06):
            for stop_pct in (0.06, 0.09):
                for runner_r in (3.0, 4.0):
                    variants.append(
                        LangLangV1Variant(
                            variant_id=(
                                f"llv1_short_r20_{abs(short_ret):.2f}"
                                f"_h1_{abs(h1_break):.2f}_st_{stop_pct:.2f}_rr_{runner_r:.1f}"
                            ),
                            short_ret_20d_max=short_ret,
                            waterfall_h1_ret_24_max=h1_break,
                            structure_stop_pct=stop_pct,
                            runner_take_profit_r=runner_r,
                            ret_20d_min=0.20,
                            ret_60d_min=0.48,
                        )
                    )
    return variants


def default_langlang_v1_1_grid() -> list[LangLangV1_1Variant]:
    variants: list[LangLangV1_1Variant] = []
    for ret_20d in (0.16, 0.22, 0.28):
        for upside in (0.06, 0.10):
            for hist_score in (0.25, 0.35, 0.45):
                for stop_cluster in (1, 2):
                    variants.append(
                        LangLangV1_1Variant(
                            variant_id=(
                                f"llv1_1_long_r20_{ret_20d:.2f}_space_{upside:.2f}"
                                f"_hm_{hist_score:.2f}_sc_{stop_cluster}"
                            ),
                            allowed_side="long",
                            ret_20d_min=ret_20d,
                            min_upside_space_pct=upside,
                            min_historical_match_score=hist_score,
                            max_stop_loss_cluster_24h=stop_cluster,
                        )
                    )
    for short_ret in (-0.12, -0.18):
        for hist_score in (0.25, 0.40):
            variants.append(
                LangLangV1_1Variant(
                    variant_id=f"llv1_1_short_r20_{abs(short_ret):.2f}_hm_{hist_score:.2f}",
                    allowed_side="short",
                    short_ret_20d_max=short_ret,
                    min_historical_match_score=hist_score,
                )
            )
    variants.append(LangLangV1_1Variant(variant_id="llv1_1_exploratory", allowed_side="both", exploratory=True))
    return variants


def default_langlang_v1_3_grid() -> list[LangLangV1_3Variant]:
    variants: list[LangLangV1_3Variant] = []
    for ret_20d in (0.18, 0.24, 0.30):
        for upside in (0.10, 0.18):
            for hist_score in (0.30, 0.45):
                variants.append(
                    LangLangV1_3Variant(
                        variant_id=f"llv1_3_long_r20_{ret_20d:.2f}_space_{upside:.2f}_hm_{hist_score:.2f}",
                        allowed_side="long",
                        ret_20d_min=ret_20d,
                        min_upside_space_pct=upside,
                        min_historical_match_score=hist_score,
                    )
                )
    for short_ret in (-0.14, -0.20, -0.28):
        for hist_score in (0.25, 0.40):
            variants.append(
                LangLangV1_3Variant(
                    variant_id=f"llv1_3_short_r20_{abs(short_ret):.2f}_hm_{hist_score:.2f}",
                    allowed_side="short",
                    short_ret_20d_max=short_ret,
                    min_historical_match_score=hist_score,
                    leader_only_long=False,
                )
            )
    variants.append(
        LangLangV1_3Variant(
            variant_id="llv1_3_exploratory",
            allowed_side="both",
            exploratory=True,
            leader_only_long=False,
            enable_countertrend_short=True,
        )
    )
    return variants


def default_langlang_native_grid() -> list[LangLangNativeVariant]:
    variants: list[LangLangNativeVariant] = []
    for side in ("long", "short"):
        variants.append(
            LangLangNativeVariant(
                variant_id=f"native_{side}_document_default",
                allowed_side=side,
                ret_20d_min=0.18,
                ret_60d_min=0.42,
                min_upside_space_pct=0.08,
                leader_only_long=True,
                enable_countertrend_short=False,
            )
        )
    variants.append(
        LangLangNativeVariant(
            variant_id="native_document_countertrend_observer",
            allowed_side="both",
            enable_countertrend_short=True,
            leader_only_long=False,
        )
    )
    return variants


def default_langlang_enhanced_grid() -> list[LangLangEnhancedVariant]:
    variants: list[LangLangEnhancedVariant] = []
    for ret_20d in (0.18, 0.24, 0.30):
        for hist_score in (0.30, 0.45):
            variants.append(
                LangLangEnhancedVariant(
                    variant_id=f"enhanced_long_r20_{ret_20d:.2f}_hm_{hist_score:.2f}",
                    allowed_side="long",
                    ret_20d_min=ret_20d,
                    min_historical_match_score=hist_score,
                    max_funding_rate_last=0.015,
                )
            )
    for short_ret in (-0.14, -0.22):
        variants.append(
            LangLangEnhancedVariant(
                variant_id=f"enhanced_short_r20_{abs(short_ret):.2f}",
                allowed_side="short",
                short_ret_20d_max=short_ret,
                leader_only_long=False,
                min_historical_match_score=0.30,
            )
        )
    variants.append(
        LangLangEnhancedVariant(
            variant_id="enhanced_exploratory",
            allowed_side="both",
            exploratory=True,
            leader_only_long=False,
            enable_countertrend_short=True,
        )
    )
    return variants


class RulesV01Strategy:
    """First distilled, human-readable version of the main-uptrend idea."""

    version = "rules_v01"

    def __init__(self, variant: StrategyVariant | None = None):
        self.variant = variant or StrategyVariant()

    def generate(self, symbol: str, candles: list[Candle]) -> Signal | None:
        snapshot = DailyFeatureBuilder().build(symbol, candles)
        if snapshot is None:
            return None
        return self.generate_from_features(snapshot)

    def generate_from_features(self, snapshot: FeatureSnapshot) -> Signal | None:
        features = snapshot.features
        ret_20d = float(features["ret_20d"])
        ret_60d = float(features["ret_60d"])
        pos_20d = float(features["pos_20d"])
        pullback_from_20d_high = float(features["pullback_from_20d_high"])
        ma_5 = float(features["ma_5"])
        ma_20 = float(features["ma_20"])
        latest_close = float(features["latest_close"])
        high_60d = float(features["high_60d"])
        high_20d = float(features["high_20d"])
        low_20d = float(features["low_20d"])
        strong_trend = ret_20d >= self.variant.ret_20d_min and ret_60d >= self.variant.ret_60d_min
        structure_ok = latest_close >= ma_20 and ma_5 >= ma_20 and pos_20d >= self.variant.pos_20d_min
        not_too_extended = pullback_from_20d_high >= -self.variant.max_pullback_pct
        if not (strong_trend and structure_ok and not_too_extended):
            return None

        strength = min(1.0, 0.35 + ret_20d * 1.2 + ret_60d * 0.4)
        reason_codes = ["main_uptrend_daily", "structure_above_ma20", f"variant:{self.variant.variant_id}"]
        if latest_close >= high_60d * (1 - self.variant.breakout_tolerance):
            reason_codes.append("near_60d_breakout")
        if pullback_from_20d_high < -0.03:
            reason_codes.append("pullback_not_broken")
        signal_features = dict(features)
        signal_features.update(self.variant.to_dict())

        return Signal(
            symbol=snapshot.symbol,
            side=Side.LONG,
            strength=strength,
            reason_codes=reason_codes,
            features=signal_features,
            invalidation_price=low_20d,
            take_profit_hint=high_20d * 1.15,
            created_at=snapshot.created_at,
        )


class RulesLangLangV1Strategy:
    version = "rules_langlang_v1"

    def __init__(self, variant: LangLangV1Variant | None = None):
        self.variant = variant or LangLangV1Variant()

    def generate(self, symbol: str, candles: list[Candle]) -> LangLangSignal | None:
        snapshot = DailyFeatureBuilder().build(symbol, candles)
        if snapshot is None:
            return None
        return self.generate_from_features(snapshot)

    def generate_from_features(self, snapshot: FeatureSnapshot) -> LangLangSignal | None:
        decision = self.decide(snapshot)
        return decision.signal if decision.action is StrategyAction.ENTER else None

    def decide(self, snapshot: FeatureSnapshot) -> StrategyDecision:
        features = snapshot.features
        variant = self.variant
        latest_close = _float_feature(features, "latest_close")
        ret_20d = _float_feature(features, "ret_20d")
        ret_60d = _float_feature(features, "ret_60d")
        pos_20d = _float_feature(features, "pos_20d", 0.5)
        pullback = _float_feature(features, "pullback_from_20d_high")
        ma_5 = _float_feature(features, "ma_5")
        ma_20 = _float_feature(features, "ma_20")
        high_20d = _float_feature(features, "high_20d", latest_close)
        low_20d = _float_feature(features, "low_20d", latest_close)
        high_60d = _float_feature(features, "high_60d", high_20d)
        vol_ratio = _float_feature(features, "vol_ratio_20d", 1.0)
        h1_ret_24 = _float_feature(features, "h1_ret_24")
        h1_pos_48 = _float_feature(features, "h1_pos_48", 0.5)
        h1_pullback = _float_feature(features, "h1_pullback_from_high")
        m15_ret_8 = _float_feature(features, "m15_ret_8")
        m5_ret_6 = _float_feature(features, "m5_ret_6")

        if latest_close <= 0:
            return _skip("skip:missing_latest_close", [FailureFilter.STRUCTURE_BREAK])
        if vol_ratio < variant.min_vol_ratio_20d:
            return _skip("skip:low_liquidity", [FailureFilter.LOW_LIQUIDITY])

        long_trend = (
            ret_20d >= variant.ret_20d_min
            and ret_60d >= variant.ret_60d_min
            and pos_20d >= variant.pos_20d_min
            and latest_close >= ma_20
            and ma_5 >= ma_20
        )
        short_trend = (
            (ret_20d <= variant.short_ret_20d_max or ret_60d <= variant.short_ret_60d_max)
            and pos_20d <= variant.short_pos_20d_max
            and latest_close <= ma_20
            and ma_5 <= ma_20
        )
        overheated = (
            long_trend
            and ret_20d >= variant.overheat_ret_20d
            and pos_20d >= variant.overheat_pos_20d
            and pullback >= -variant.min_pullback_pct
            and h1_ret_24 >= variant.overheat_h1_ret_24
            and (_ret_le(features, "m15", m15_ret_8, 0) or _ret_le(features, "m5", m5_ret_6, 0))
        )
        if overheated:
            return StrategyDecision(
                action=StrategyAction.SKIP,
                explanation="skip:regime=top_divergence filters=chase_overheat,large_divergence",
                matched_historical_patterns=[],
                risk_notes=["avoid chasing extended daily move after intraday weakness"],
                filter_codes=[FailureFilter.CHASE_OVERHEAT, FailureFilter.LARGE_DIVERGENCE],
            )

        if long_trend:
            if _ret_lt(features, "m15", m15_ret_8, -abs(variant.intraday_confirm_ret_min)) and _ret_lt(features, "m5", m5_ret_6, 0):
                return _skip("skip:large_divergence", [FailureFilter.LARGE_DIVERGENCE])
            if pullback <= -variant.min_pullback_pct and pullback >= -variant.max_pullback_pct:
                regime = MarketRegime.STRONG_PULLBACK
                setup = EntrySetup.FIRST_PULLBACK if h1_pullback <= -variant.min_pullback_pct else EntrySetup.PLATFORM_RETEST
            elif latest_close >= high_60d * (1 - variant.breakout_tolerance):
                regime = MarketRegime.BREAKOUT_RETEST
                setup = EntrySetup.FIRST_BREAKOUT
            else:
                regime = MarketRegime.MAIN_UPTREND
                setup = EntrySetup.SECOND_ENTRY
            stop_loss = max(low_20d, latest_close * (1 - variant.structure_stop_pct))
            risk_per_unit = max(latest_close - stop_loss, latest_close * 0.01)
            reason_codes = ["daily_main_uptrend", f"variant:{variant.variant_id}"]
            if _ret_ge(features, "m15", m15_ret_8, variant.intraday_confirm_ret_min) or _ret_ge(features, "m5", m5_ret_6, variant.intraday_confirm_ret_min):
                reason_codes.append("intraday_reclaim_confirmed")
            if setup is EntrySetup.FIRST_PULLBACK:
                reason_codes.append("pullback_not_broken")
            if setup is EntrySetup.FIRST_BREAKOUT:
                reason_codes.append("breakout_retest")
            return self._enter_decision(
                snapshot=snapshot,
                side=Side.LONG,
                regime=regime,
                setup=setup,
                strength=min(1.0, 0.42 + ret_20d * 0.8 + ret_60d * 0.25 + max(m15_ret_8, 0) * 6),
                reason_codes=reason_codes,
                invalidation_price=stop_loss,
                take_profit_hint=latest_close + risk_per_unit * variant.runner_take_profit_r,
            )

        if short_trend:
            if _ret_le(features, "h1", h1_ret_24, variant.waterfall_h1_ret_24_max) or _ret_le(features, "m15", m15_ret_8, variant.intraday_breakdown_ret_max):
                setup = EntrySetup.WATERFALL_CONTINUATION
            else:
                setup = EntrySetup.SHORT_REBOUND_FAILURE
            stop_loss = min(high_20d, latest_close * (1 + variant.structure_stop_pct))
            if stop_loss <= latest_close:
                stop_loss = latest_close * (1 + variant.structure_stop_pct)
            risk_per_unit = max(stop_loss - latest_close, latest_close * 0.01)
            return self._enter_decision(
                snapshot=snapshot,
                side=Side.SHORT,
                regime=MarketRegime.WEAK_WATERFALL,
                setup=setup,
                strength=min(1.0, 0.40 + abs(ret_20d) * 0.9 + abs(ret_60d) * 0.20 + abs(min(m15_ret_8, 0)) * 5),
                reason_codes=["short_weak_waterfall", f"variant:{variant.variant_id}"],
                invalidation_price=stop_loss,
                take_profit_hint=latest_close - risk_per_unit * variant.runner_take_profit_r,
            )

        return StrategyDecision(
            action=StrategyAction.SKIP,
            explanation="skip:regime=choppy_invalid filters=structure_break",
            matched_historical_patterns=[],
            risk_notes=["no complete langlang setup matched"],
            filter_codes=[FailureFilter.STRUCTURE_BREAK],
        )

    def _enter_decision(
        self,
        *,
        snapshot: FeatureSnapshot,
        side: Side,
        regime: MarketRegime,
        setup: EntrySetup,
        strength: float,
        reason_codes: list[str],
        invalidation_price: float,
        take_profit_hint: float,
    ) -> StrategyDecision:
        variant = self.variant
        decision_trace = {
            "action": StrategyAction.ENTER.value,
            "regime": regime.value,
            "setup": setup.value,
            "variant_id": variant.variant_id,
        }
        signal = LangLangSignal(
            symbol=snapshot.symbol,
            side=side,
            strength=strength,
            reason_codes=reason_codes,
            filter_codes=[FailureFilter.NO_FAILURE_FILTER.value],
            features={**snapshot.features, **variant.to_dict()},
            invalidation_price=invalidation_price,
            stop_loss=invalidation_price,
            take_profit_hint=take_profit_hint,
            take_profit_plan={
                "partial_r": variant.partial_take_profit_r,
                "partial_exit_fraction": variant.partial_exit_fraction,
                "runner_r": variant.runner_take_profit_r,
            },
            hold_plan={
                "runner": True,
                "time_stop_days": variant.time_stop_days,
                "trend_break_buffer_pct": variant.trend_break_buffer_pct,
                "exit_on": ["structure_break", "mae_stop", "trend_break_exit"],
            },
            strategy_version=self.version,
            regime=regime,
            setup=setup,
            decision_trace=decision_trace,
            historical_match_score=variant.historical_match_score,
            created_at=snapshot.created_at,
        )
        return StrategyDecision(
            action=StrategyAction.ENTER,
            explanation=f"enter:regime={regime.value} setup={setup.value} side={side.value}",
            matched_historical_patterns=[],
            risk_notes=["structure stop controls first loss; runner captures right tail"],
            filter_codes=[FailureFilter.NO_FAILURE_FILTER],
            signal=signal,
        )


class RulesLangLangV1_1Strategy(RulesLangLangV1Strategy):
    version = "rules_langlang_v1_1"

    def __init__(self, variant: LangLangV1_1Variant | None = None):
        self.variant = variant or LangLangV1_1Variant()

    def decide(self, snapshot: FeatureSnapshot) -> StrategyDecision:
        features = snapshot.features
        variant = self.variant
        latest_close = _float_feature(features, "latest_close")
        ret_20d = _float_feature(features, "ret_20d")
        ret_60d = _float_feature(features, "ret_60d")
        pos_20d = _float_feature(features, "pos_20d", 0.5)
        pullback = _float_feature(features, "pullback_from_20d_high")
        ma_5 = _float_feature(features, "ma_5")
        ma_20 = _float_feature(features, "ma_20")
        high_20d = _float_feature(features, "high_20d", latest_close)
        low_20d = _float_feature(features, "low_20d", latest_close)
        high_60d = _float_feature(features, "high_60d", high_20d)
        vol_ratio = _float_feature(features, "vol_ratio_20d", 1.0)
        h1_ret_24 = _float_feature(features, "h1_ret_24")
        h1_pullback = _float_feature(features, "h1_pullback_from_high")
        m15_ret_8 = _float_feature(features, "m15_ret_8")
        m5_ret_6 = _float_feature(features, "m5_ret_6")
        upside_space = _float_feature(features, "upside_space_pct", 1.0)
        first_10x_done = bool(features.get("first_10x_entry_done", False))
        large_divergence_recent = bool(features.get("large_divergence_recent", False))
        bottom_lift_confirmed = bool(features.get("bottom_lift_confirmed", False))
        stop_loss_cluster = int(_float_feature(features, "stop_loss_cluster_24h", 0.0))
        historical_match_score = _float_feature(features, "historical_match_score", 0.0)
        matched_trade_examples = _list_feature(features, "matched_trade_examples")

        if latest_close <= 0:
            return _skip("skip:missing_latest_close", [FailureFilter.STRUCTURE_BREAK])
        if vol_ratio < variant.min_vol_ratio_20d:
            return _skip("skip:low_liquidity", [FailureFilter.LOW_LIQUIDITY])
        if upside_space < variant.min_upside_space_pct:
            return _skip("skip:upside_space_insufficient", [FailureFilter.INSUFFICIENT_UPSIDE_SPACE])
        if first_10x_done and pos_20d >= variant.first_10x_high_pos and pullback >= -variant.min_pullback_pct:
            return _skip("skip:first_10x_entry_already_high", [FailureFilter.FIRST_10X_TOO_HIGH])
        if stop_loss_cluster >= variant.max_stop_loss_cluster_24h:
            return _skip("skip:stop_loss_cluster", [FailureFilter.STOP_LOSS_CLUSTER, FailureFilter.EMOTIONAL_REVENGE_PROXY])
        if large_divergence_recent and not bottom_lift_confirmed:
            return _skip("skip:large_divergence_without_bottom_lift", [FailureFilter.NO_BOTTOM_LIFT])
        if (
            not variant.exploratory
            and (historical_match_score < variant.min_historical_match_score or not matched_trade_examples)
        ):
            return _skip("skip:no_historical_support", [FailureFilter.NO_HISTORICAL_SUPPORT])

        long_trend = (
            ret_20d >= variant.ret_20d_min
            and ret_60d >= variant.ret_60d_min
            and pos_20d >= variant.pos_20d_min
            and latest_close >= ma_20
            and ma_5 >= ma_20
        )
        short_trend = (
            (ret_20d <= variant.short_ret_20d_max or ret_60d <= variant.short_ret_60d_max)
            and pos_20d <= variant.short_pos_20d_max
            and latest_close <= ma_20
            and ma_5 <= ma_20
        )
        allowed_side = _variant_allowed_side(variant)
        if long_trend and allowed_side == "short":
            return _skip("skip:variant_side_not_allowed_long", [FailureFilter.VARIANT_SIDE_NOT_ALLOWED])
        if short_trend and allowed_side == "long":
            return _skip("skip:variant_side_not_allowed_short", [FailureFilter.VARIANT_SIDE_NOT_ALLOWED])
        super_large_divergence = (
            long_trend
            and ret_20d >= variant.overheat_ret_20d
            and pos_20d >= variant.overheat_pos_20d
            and h1_ret_24 >= variant.overheat_h1_ret_24
            and _ret_lt(features, "m15", m15_ret_8, 0)
            and _ret_lt(features, "m5", m5_ret_6, 0)
        )
        if super_large_divergence:
            return StrategyDecision(
                action=StrategyAction.SKIP,
                explanation="skip:super_large_divergence",
                matched_historical_patterns=matched_trade_examples,
                risk_notes=["高位大分歧后先等内部结构和底部抬升，不抢不可见的钱"],
                filter_codes=[FailureFilter.CHASE_OVERHEAT, FailureFilter.SUPER_LARGE_DIVERGENCE],
            )

        if long_trend:
            if large_divergence_recent and bottom_lift_confirmed and (
                _ret_gt(features, "m15", m15_ret_8, 0) or _ret_gt(features, "m5", m5_ret_6, 0)
            ):
                regime = MarketRegime.POST_LARGE_DIVERGENCE
                setup = EntrySetup.POST_DIVERGENCE_REBOUND
                reason_codes = ["post_large_divergence_rebound", f"variant:{variant.variant_id}"]
            elif _ret_lt(features, "m15", m15_ret_8, 0) and _ret_ge(features, "m5", m5_ret_6, 0):
                regime = MarketRegime.FIRST_DIVERGENCE
                setup = EntrySetup.SMALL_DIVERGENCE_ENTRY
                reason_codes = ["daily_main_uptrend", "small_divergence_absorbed", f"variant:{variant.variant_id}"]
            elif pullback <= -variant.min_pullback_pct and pullback >= -variant.max_pullback_pct:
                regime = MarketRegime.STRONG_PULLBACK
                setup = EntrySetup.FIRST_PULLBACK if h1_pullback <= -variant.min_pullback_pct else EntrySetup.PLATFORM_RETEST
                reason_codes = ["daily_main_uptrend", "pullback_not_broken", f"variant:{variant.variant_id}"]
            elif latest_close >= high_60d * (1 - variant.breakout_tolerance):
                regime = MarketRegime.BREAKOUT_RETEST
                setup = EntrySetup.SECOND_PRESSURE_RETEST
                reason_codes = ["daily_main_uptrend", "second_pressure_retest", f"variant:{variant.variant_id}"]
            else:
                regime = MarketRegime.MAIN_UPTREND
                setup = EntrySetup.STARTER_BUY
                reason_codes = ["daily_main_uptrend", "starter_buy", f"variant:{variant.variant_id}"]
            if _ret_ge(features, "m15", m15_ret_8, variant.intraday_confirm_ret_min) or _ret_ge(features, "m5", m5_ret_6, variant.intraday_confirm_ret_min):
                reason_codes.append("intraday_reclaim_confirmed")
            stop_loss = max(low_20d, latest_close * (1 - variant.structure_stop_pct))
            risk_per_unit = max(latest_close - stop_loss, latest_close * 0.01)
            return self._enter_v1_1_decision(
                snapshot=snapshot,
                side=Side.LONG,
                regime=regime,
                setup=setup,
                strength=min(1.0, 0.44 + ret_20d * 0.8 + ret_60d * 0.22 + historical_match_score * 0.12),
                reason_codes=reason_codes,
                invalidation_price=stop_loss,
                take_profit_hint=latest_close + risk_per_unit * variant.runner_take_profit_r,
                matched_trade_examples=matched_trade_examples,
                historical_match_score=historical_match_score,
            )

        if short_trend:
            setup = (
                EntrySetup.WATERFALL_CONTINUATION
                if _ret_le(features, "h1", h1_ret_24, variant.waterfall_h1_ret_24_max) or _ret_le(features, "m15", m15_ret_8, variant.intraday_breakdown_ret_max)
                else EntrySetup.SHORT_REBOUND_FAILURE
            )
            stop_loss = min(high_20d, latest_close * (1 + variant.structure_stop_pct))
            if stop_loss <= latest_close:
                stop_loss = latest_close * (1 + variant.structure_stop_pct)
            risk_per_unit = max(stop_loss - latest_close, latest_close * 0.01)
            return self._enter_v1_1_decision(
                snapshot=snapshot,
                side=Side.SHORT,
                regime=MarketRegime.WEAK_WATERFALL,
                setup=setup,
                strength=min(1.0, 0.42 + abs(ret_20d) * 0.85 + abs(ret_60d) * 0.20 + historical_match_score * 0.12),
                reason_codes=["short_weak_waterfall", f"variant:{variant.variant_id}"],
                invalidation_price=stop_loss,
                take_profit_hint=latest_close - risk_per_unit * variant.runner_take_profit_r,
                matched_trade_examples=matched_trade_examples,
                historical_match_score=historical_match_score,
            )

        return StrategyDecision(
            action=StrategyAction.SKIP,
            explanation="skip:regime=choppy_invalid filters=structure_break",
            matched_historical_patterns=matched_trade_examples,
            risk_notes=["没有形成完整浪浪结构，等待而不是硬做"],
            filter_codes=[FailureFilter.STRUCTURE_BREAK],
        )

    def _enter_v1_1_decision(
        self,
        *,
        snapshot: FeatureSnapshot,
        side: Side,
        regime: MarketRegime,
        setup: EntrySetup,
        strength: float,
        reason_codes: list[str],
        invalidation_price: float,
        take_profit_hint: float,
        matched_trade_examples: list[dict],
        historical_match_score: float,
    ) -> StrategyDecision:
        variant = self.variant
        pos_20d = _float_feature(snapshot.features, "pos_20d", 0.5)
        risk_unit = "W"
        size_multiplier = 1.0
        risk_notes = ["结构止损控制第一笔亏损，右尾仓位用 runner 计划保留"]
        if pos_20d <= variant.low_position_boost_pos:
            size_multiplier = variant.low_position_size_multiplier
            risk_notes.append("低位且结构完整，可用更高风险单位，但仍执行固定止损")
        elif pos_20d >= variant.high_position_reduce_pos:
            risk_unit = "0.6W"
            size_multiplier = variant.high_position_size_multiplier
            risk_notes.append("高位只允许降仓试错，避免第一次10x后的执拗追价")
        decision_trace = {
            "action": StrategyAction.ENTER.value,
            "strategy_version": self.version,
            "regime": regime.value,
            "setup": setup.value,
            "variant_id": variant.variant_id,
            "risk_unit": risk_unit,
            "position_size_multiplier": size_multiplier,
            "matched_trade_examples": matched_trade_examples,
            "historical_match_score": historical_match_score,
        }
        signal = LangLangSignal(
            symbol=snapshot.symbol,
            side=side,
            strength=strength,
            reason_codes=reason_codes,
            filter_codes=[FailureFilter.NO_FAILURE_FILTER.value],
            features={
                **snapshot.features,
                **variant.to_dict(),
                "risk_unit": risk_unit,
                "position_size_multiplier": size_multiplier,
                "matched_trade_examples": matched_trade_examples,
            },
            invalidation_price=invalidation_price,
            stop_loss=invalidation_price,
            take_profit_hint=take_profit_hint,
            take_profit_plan={
                "partial_r": variant.partial_take_profit_r,
                "partial_exit_fraction": variant.partial_exit_fraction,
                "runner_r": variant.runner_take_profit_r,
                "risk_unit": risk_unit,
            },
            hold_plan={
                "runner": True,
                "time_stop_days": variant.time_stop_days,
                "trend_break_buffer_pct": variant.trend_break_buffer_pct,
                "exit_on": ["structure_break", "mae_stop", "trend_break_exit"],
                "right_tail_required": True,
            },
            strategy_version=self.version,
            regime=regime,
            setup=setup,
            decision_trace=decision_trace,
            historical_match_score=historical_match_score,
            matched_trade_examples=matched_trade_examples,
            risk_notes=risk_notes,
            created_at=snapshot.created_at,
        )
        return StrategyDecision(
            action=StrategyAction.ENTER,
            explanation=f"enter:strategy=v1.1 regime={regime.value} setup={setup.value} side={side.value}",
            matched_historical_patterns=matched_trade_examples,
            risk_notes=risk_notes,
            filter_codes=[FailureFilter.NO_FAILURE_FILTER],
            signal=signal,
        )


class RulesLangLangV1_3Strategy(RulesLangLangV1_1Strategy):
    version = "rules_langlang_v1_3"

    def __init__(self, variant: LangLangV1_3Variant | None = None):
        self.variant = variant or LangLangV1_3Variant()

    def _strategy_line(self) -> str:
        return "v1_3"

    def _requires_historical_support(self) -> bool:
        return True

    def _uses_enhanced_loss_filters(self) -> bool:
        return True

    def _enhanced_failure_filters(
        self,
        features: dict,
        variant: LangLangV1_3Variant,
        historical_match_score: float,
        matched_trade_examples: list[dict],
    ) -> tuple[str, list[FailureFilter]] | None:
        return None

    def _orthogonal_entry_decision(
        self,
        *,
        snapshot: FeatureSnapshot,
        entry_family: str,
        market_season: str,
        symbol_cycle: str,
        selection_reason_codes: list[str],
        selection_filter_codes: list[str],
        matched_trade_examples: list[dict],
        historical_match_score: float,
    ) -> StrategyDecision | None:
        features = snapshot.features
        variant = self.variant
        latest_close = _float_feature(features, "latest_close")
        ret_20d = _float_feature(features, "ret_20d")
        ret_60d = _float_feature(features, "ret_60d")
        pos_20d = _float_feature(features, "pos_20d", 0.5)
        ma_20 = _float_feature(features, "ma_20")
        high_20d = _float_feature(features, "high_20d", latest_close)
        low_20d = _float_feature(features, "low_20d", latest_close)
        m15_ret_8 = _float_feature(features, "m15_ret_8")
        m5_ret_6 = _float_feature(features, "m5_ret_6")
        wyckoff_long_tag = _str_feature(features, "wyckoff_long_setup_tag", "")
        wyckoff_short_tag = _str_feature(features, "wyckoff_short_setup_tag", "")
        wyckoff_long_score = _float_feature(features, "wyckoff_long_score")
        wyckoff_short_score = _float_feature(features, "wyckoff_short_score")

        if entry_family == "low_position_wyckoff_long":
            if _variant_allowed_side(variant) == "short":
                return None
            low_position = pos_20d <= min(variant.low_position_boost_pos, 0.55)
            spring_or_retest = wyckoff_long_tag in {"spring_reclaim", "lps_retest"}
            reclaimed_structure = ma_20 > 0 and latest_close >= ma_20 * 0.985
            if not (low_position and spring_or_retest and wyckoff_long_score >= 0.68 and reclaimed_structure):
                return None
            stop_loss = max(low_20d, latest_close * (1 - variant.structure_stop_pct))
            risk_per_unit = max(latest_close - stop_loss, latest_close * 0.01)
            return self._enter_v1_3_decision(
                snapshot=snapshot,
                side=Side.LONG,
                regime=MarketRegime.PRE_MAIN_UPTREND,
                setup=EntrySetup.STARTER_BUY,
                entry_position_id="1_low_position_wyckoff_spring_long",
                market_season=market_season,
                symbol_cycle=symbol_cycle,
                strength=min(1.0, 0.40 + wyckoff_long_score * 0.35 + max(ret_60d, 0.0) * 0.15),
                reason_codes=[
                    "orthogonal_low_position_wyckoff_long",
                    wyckoff_long_tag,
                    f"variant:{variant.variant_id}",
                ],
                invalidation_price=stop_loss,
                take_profit_hint=latest_close + risk_per_unit * variant.runner_take_profit_r,
                matched_trade_examples=matched_trade_examples,
                historical_match_score=historical_match_score,
                selection_reason_codes=selection_reason_codes,
                selection_filter_codes=selection_filter_codes,
                filter_codes=[FailureFilter.NO_FAILURE_FILTER.value],
            )

        if entry_family == "failed_breakdown_reclaim_long":
            if _variant_allowed_side(variant) == "short":
                return None
            pattern_tag = _str_feature(features, "strong_pattern_tag", "")
            pattern_reasons = set(_string_list_feature(features, "pattern_reason_codes"))
            reclaim_pattern = (
                pattern_tag == "spoon_bottom_confirmed"
                or "failed_breakdown_reclaim" in pattern_reasons
                or _truthy_feature(features, "box_rebound_candidate")
            )
            reclaimed_structure = ma_20 > 0 and latest_close >= ma_20 * 0.96
            intraday_reclaim = _ret_ge(features, "m15", m15_ret_8, variant.intraday_confirm_ret_min) or _ret_ge(
                features,
                "m5",
                m5_ret_6,
                variant.intraday_confirm_ret_min,
            )
            if not (
                reclaim_pattern
                and reclaimed_structure
                and intraday_reclaim
                and pos_20d <= 0.45
                and ret_60d >= -0.10
            ):
                return None
            stop_loss = max(low_20d, latest_close * (1 - variant.structure_stop_pct))
            risk_per_unit = max(latest_close - stop_loss, latest_close * 0.01)
            return self._enter_v1_3_decision(
                snapshot=snapshot,
                side=Side.LONG,
                regime=MarketRegime.CHOPPY_INVALID,
                setup=EntrySetup.BOX_REBOUND_LONG,
                entry_position_id="6_failed_breakdown_reclaim_long",
                market_season=market_season,
                symbol_cycle=symbol_cycle,
                strength=min(1.0, 0.36 + max(ret_20d, 0.0) * 0.20 + max(ret_60d, 0.0) * 0.12 + 0.22),
                reason_codes=[
                    "orthogonal_failed_breakdown_reclaim_long",
                    pattern_tag or "failed_breakdown_reclaim",
                    f"variant:{variant.variant_id}",
                ],
                invalidation_price=stop_loss,
                take_profit_hint=latest_close + risk_per_unit * variant.partial_take_profit_r,
                matched_trade_examples=matched_trade_examples,
                historical_match_score=historical_match_score,
                selection_reason_codes=selection_reason_codes,
                selection_filter_codes=selection_filter_codes,
                force_no_runner=True,
                filter_codes=[FailureFilter.NO_FAILURE_FILTER.value],
            )

        if entry_family == "retest_confirmed_short":
            if _variant_allowed_side(variant) == "long":
                return None
            retest_tag = wyckoff_short_tag in {"upthrust_reversal", "utad_risk", "lpsy_retest"}
            intraday_reject = _has_wyckoff_intraday_confirmation(features, "short", variant, m15_ret_8, m5_ret_6)
            if not (retest_tag and wyckoff_short_score >= 0.70 and intraday_reject):
                return None
            setup = (
                EntrySetup.TOP_SHORT
                if wyckoff_short_tag in {"upthrust_reversal", "utad_risk"}
                else EntrySetup.SHORT_REBOUND_FAILURE
            )
            stop_loss = max(high_20d, latest_close * (1 + variant.structure_stop_pct))
            risk_per_unit = max(stop_loss - latest_close, latest_close * 0.01)
            return self._enter_v1_3_decision(
                snapshot=snapshot,
                side=Side.SHORT,
                regime=MarketRegime.TOP_DIVERGENCE,
                setup=setup,
                entry_position_id="3_retest_confirmed_short",
                market_season=market_season,
                symbol_cycle=symbol_cycle,
                strength=min(1.0, 0.38 + wyckoff_short_score * 0.35 + max(pos_20d, 0.0) * 0.08),
                reason_codes=[
                    "orthogonal_retest_confirmed_short",
                    wyckoff_short_tag,
                    f"variant:{variant.variant_id}",
                ],
                invalidation_price=stop_loss,
                take_profit_hint=latest_close - risk_per_unit * variant.runner_take_profit_r,
                matched_trade_examples=matched_trade_examples,
                historical_match_score=historical_match_score,
                selection_reason_codes=selection_reason_codes,
                selection_filter_codes=selection_filter_codes,
                filter_codes=[FailureFilter.NO_FAILURE_FILTER.value],
            )

        return None

    def decide(self, snapshot: FeatureSnapshot) -> StrategyDecision:
        features = snapshot.features
        variant = self.variant
        latest_close = _float_feature(features, "latest_close")
        ret_20d = _float_feature(features, "ret_20d")
        ret_60d = _float_feature(features, "ret_60d")
        pos_20d = _float_feature(features, "pos_20d", 0.5)
        pullback = _float_feature(features, "pullback_from_20d_high")
        ma_5 = _float_feature(features, "ma_5")
        ma_20 = _float_feature(features, "ma_20")
        high_20d = _float_feature(features, "high_20d", latest_close)
        low_20d = _float_feature(features, "low_20d", latest_close)
        high_60d = _float_feature(features, "high_60d", high_20d)
        vol_ratio = _float_feature(features, "vol_ratio_20d", 1.0)
        h1_ret_24 = _float_feature(features, "h1_ret_24")
        h1_pullback = _float_feature(features, "h1_pullback_from_high")
        m15_ret_8 = _float_feature(features, "m15_ret_8")
        m5_ret_6 = _float_feature(features, "m5_ret_6")
        upside_space = _float_feature(features, "upside_space_pct", 1.0)
        first_10x_done = bool(features.get("first_10x_entry_done", False))
        large_divergence_recent = bool(features.get("large_divergence_recent", False))
        bottom_lift_confirmed = bool(features.get("bottom_lift_confirmed", False))
        stop_loss_cluster = int(_float_feature(features, "stop_loss_cluster_24h", 0.0))
        historical_match_score = _float_feature(features, "historical_match_score", 0.0)
        matched_trade_examples = _list_feature(features, "matched_trade_examples")
        selection_reason_codes = _string_list_feature(features, "selection_reason_codes")
        selection_filter_codes = _string_list_feature(features, "selection_filter_codes")
        selection_tag = _str_feature(features, "symbol_selection_tag", "")
        strong_pattern_tag = _str_feature(features, "strong_pattern_tag", "")
        risk_pattern_tag = _str_feature(features, "risk_pattern_tag", "")
        entry_family = _variant_entry_family(variant)
        leader_platform_start_score = _float_feature(features, "leader_platform_start_score")
        golden_pit_reclaim_score = _float_feature(features, "golden_pit_reclaim_score")
        small_divergence_absorb_score = _float_feature(features, "small_divergence_absorb_score")
        second_wave_start_score = _float_feature(features, "second_wave_start_score")
        five_wave_late_risk_score = _float_feature(features, "five_wave_late_risk_score")
        false_breakout_risk_score = _float_feature(features, "false_breakout_risk_score")
        wyckoff_phase_tag = _str_feature(features, "wyckoff_phase_tag", "")
        wyckoff_long_setup_tag = _str_feature(features, "wyckoff_long_setup_tag", "")
        wyckoff_short_setup_tag = _str_feature(features, "wyckoff_short_setup_tag", "")
        wyckoff_exit_tag = _str_feature(features, "wyckoff_exit_tag", "")
        wyckoff_long_score = _float_feature(features, "wyckoff_long_score")
        wyckoff_short_score = _float_feature(features, "wyckoff_short_score")
        wyckoff_risk_score = _float_feature(features, "wyckoff_risk_score")
        wyckoff_exit_score = _float_feature(features, "wyckoff_exit_score")
        market_season = _infer_market_season(features)
        symbol_cycle = _infer_symbol_cycle(features)
        requested_side = _str_feature(features, "requested_side", "").lower()
        selection_mode = _str_feature(features, "selection_mode", "").lower()
        current_position_side = _str_feature(features, "current_position_side", "").lower()
        allowed_side = _variant_allowed_side(variant)
        intent_side = _intent_side(
            requested_side=requested_side,
            allowed_side=allowed_side,
            selection_mode=selection_mode,
            selection_tag=selection_tag,
        )
        long_intent = intent_side == "long"
        short_intent = intent_side == "short"

        if latest_close <= 0:
            return _skip("skip:missing_latest_close", [FailureFilter.STRUCTURE_BREAK])
        if current_position_side == "long" and (wyckoff_exit_score >= 0.70 or wyckoff_risk_score >= 0.70):
            exit_action = (
                StrategyAction.CLOSE
                if (wyckoff_exit_tag or wyckoff_short_setup_tag) in {"sow_breakdown", "lpsy_retest"}
                else StrategyAction.REDUCE
            )
            return StrategyDecision(
                action=exit_action,
                explanation=f"{exit_action.value}:wyckoff_exit:{wyckoff_exit_tag or wyckoff_short_setup_tag or wyckoff_phase_tag}",
                matched_historical_patterns=matched_trade_examples,
                risk_notes=["威科夫派发/供给信号触发多头减仓或离场"],
                filter_codes=[FailureFilter.WYCKOFF_RISK],
            )
        if vol_ratio < variant.min_vol_ratio_20d:
            return _skip("skip:low_liquidity", [FailureFilter.LOW_LIQUIDITY])
        if self._uses_enhanced_loss_filters() and stop_loss_cluster >= variant.max_stop_loss_cluster_24h:
            return _skip("skip:stop_loss_cluster", [FailureFilter.STOP_LOSS_CLUSTER, FailureFilter.EMOTIONAL_REVENGE_PROXY])
        if long_intent and int(_float_feature(features, "small_divergence_count", 0.0)) > variant.max_small_divergence_count:
            return _skip("skip:third_small_divergence_five_wave_high", [FailureFilter.THIRD_SMALL_DIVERGENCE])
        wyckoff_short_candidate = (
            wyckoff_short_score >= 0.70
            and wyckoff_short_setup_tag in {"upthrust_reversal", "utad_risk", "sow_breakdown", "lpsy_retest"}
        )
        if (
            short_intent
            and symbol_cycle in {"main_wave", "small_divergence", "platform_start"}
            and not variant.enable_countertrend_short
            and not wyckoff_short_candidate
        ):
            return _skip("skip:main_wave_countertrend_short_disabled", [FailureFilter.COUNTER_TREND_SHORT_DISABLED])
        if (
            variant.autumn_winter_only_best_positions
            and market_season in {"autumn", "winter"}
            and symbol_cycle not in {"platform_start", "second_wave", "weak_waterfall"}
        ):
            filters = [FailureFilter.AUTUMN_WINTER_REDUCED_FREQUENCY]
            if symbol_cycle == "box_chop":
                filters.append(FailureFilter.BOX_REBOUND_LOW_QUALITY)
            return _skip("skip:autumn_winter_reduce_frequency", filters)
        if long_intent and upside_space < variant.min_upside_space_pct:
            return _skip("skip:upside_space_insufficient", [FailureFilter.INSUFFICIENT_UPSIDE_SPACE])
        if long_intent and first_10x_done and pos_20d >= variant.first_10x_high_pos and pullback >= -variant.min_pullback_pct:
            return _skip("skip:first_10x_entry_already_high", [FailureFilter.FIRST_10X_TOO_HIGH])
        if long_intent and (false_breakout_risk_score >= 0.65 or risk_pattern_tag == "false_breakout_risk"):
            return _skip("skip:false_breakout_risk_pattern", [FailureFilter.FALSE_BREAKOUT_AFTER_CONTRACTION])
        if long_intent and (five_wave_late_risk_score >= 0.70 or risk_pattern_tag == "five_wave_late_risk"):
            return _skip("skip:five_wave_late_risk", [FailureFilter.FIVE_WAVE_LATE_RISK])
        if (
            long_intent
            and (
                wyckoff_phase_tag == "distribution"
                or wyckoff_risk_score >= 0.70
                or wyckoff_exit_score >= 0.70
            )
        ):
            return _skip(f"skip:wyckoff_risk:{wyckoff_exit_tag or wyckoff_short_setup_tag or wyckoff_phase_tag}", [FailureFilter.WYCKOFF_RISK])
        if _truthy_feature(features, "btc_divergence_alt_breakout") and long_intent:
            return _skip("skip:btc_divergence_alt_breakout", [FailureFilter.BTC_DIVERGENCE_ALT_BREAKOUT])
        if long_intent and _truthy_feature(features, "false_breakout_after_contraction"):
            return _skip("skip:false_breakout_after_contraction", [FailureFilter.FALSE_BREAKOUT_AFTER_CONTRACTION])
        if long_intent and large_divergence_recent and not bottom_lift_confirmed and symbol_cycle != "first_large_divergence":
            return _skip("skip:large_divergence_without_bottom_lift", [FailureFilter.NO_BOTTOM_LIFT])
        enhanced_failure = self._enhanced_failure_filters(features, variant, historical_match_score, matched_trade_examples)
        if enhanced_failure is not None:
            explanation, filters = enhanced_failure
            return _skip(explanation, filters)
        if (
            self._requires_historical_support()
            and
            not variant.exploratory
            and (historical_match_score < variant.min_historical_match_score or not matched_trade_examples)
        ):
            return _skip("skip:no_historical_support", [FailureFilter.NO_HISTORICAL_SUPPORT])

        if entry_family in {
            "low_position_wyckoff_long",
            "failed_breakdown_reclaim_long",
            "retest_confirmed_short",
        }:
            orthogonal_decision = self._orthogonal_entry_decision(
                snapshot=snapshot,
                entry_family=entry_family,
                market_season=market_season,
                symbol_cycle=symbol_cycle,
                selection_reason_codes=selection_reason_codes,
                selection_filter_codes=selection_filter_codes,
                matched_trade_examples=matched_trade_examples,
                historical_match_score=historical_match_score,
            )
            if orthogonal_decision is not None:
                return orthogonal_decision
            return _skip(f"skip:entry_family_no_match:{entry_family}", [FailureFilter.STRUCTURE_BREAK])

        long_trend = (
            ret_20d >= variant.ret_20d_min
            and ret_60d >= variant.ret_60d_min
            and pos_20d >= variant.pos_20d_min
            and latest_close >= ma_20
            and ma_5 >= ma_20
        )
        strong_pattern_long_trend_ok = _strong_pattern_long_trend_ok(
            features=features,
            variant=variant,
            ret_20d=ret_20d,
            ret_60d=ret_60d,
            pos_20d=pos_20d,
            latest_close=latest_close,
            ma_5=ma_5,
            ma_20=ma_20,
            vol_ratio=vol_ratio,
            m15_ret_8=m15_ret_8,
            m5_ret_6=m5_ret_6,
            bottom_lift_confirmed=bottom_lift_confirmed,
        )
        wyckoff_long_trend_ok = _wyckoff_long_trend_ok(
            features=features,
            variant=variant,
            ret_20d=ret_20d,
            ret_60d=ret_60d,
            pos_20d=pos_20d,
            latest_close=latest_close,
            ma_5=ma_5,
            ma_20=ma_20,
            vol_ratio=vol_ratio,
            m15_ret_8=m15_ret_8,
            m5_ret_6=m5_ret_6,
        )
        short_trend = (
            (ret_20d <= variant.short_ret_20d_max or ret_60d <= variant.short_ret_60d_max)
            and pos_20d <= variant.short_pos_20d_max
            and latest_close <= ma_20
            and ma_5 <= ma_20
        )
        wyckoff_short_ok = _wyckoff_short_trend_ok(features=features, variant=variant, m15_ret_8=m15_ret_8, m5_ret_6=m5_ret_6)
        if (
            (long_trend or strong_pattern_long_trend_ok or wyckoff_long_trend_ok)
            and allowed_side == "short"
            and not (short_trend or wyckoff_short_ok or (short_intent and variant.enable_countertrend_short))
        ):
            return _skip("skip:variant_side_not_allowed_long", [FailureFilter.VARIANT_SIDE_NOT_ALLOWED])
        if (short_trend or wyckoff_short_ok) and allowed_side == "long":
            return _skip("skip:variant_side_not_allowed_short", [FailureFilter.VARIANT_SIDE_NOT_ALLOWED])

        if short_trend or wyckoff_short_ok:
            if wyckoff_short_ok:
                setup = (
                    EntrySetup.TOP_SHORT
                    if wyckoff_short_setup_tag in {"upthrust_reversal", "utad_risk"}
                    else EntrySetup.WATERFALL_CONTINUATION
                )
            else:
                setup = (
                    EntrySetup.WATERFALL_CONTINUATION
                    if symbol_cycle == "weak_waterfall" or _ret_le(features, "h1", h1_ret_24, variant.waterfall_h1_ret_24_max) or _ret_le(features, "m15", m15_ret_8, variant.intraday_breakdown_ret_max)
                    else EntrySetup.SHORT_REBOUND_FAILURE
                )
            stop_loss = min(high_20d, latest_close * (1 + variant.structure_stop_pct))
            if stop_loss <= latest_close:
                stop_loss = latest_close * (1 + variant.structure_stop_pct)
            risk_per_unit = max(stop_loss - latest_close, latest_close * 0.01)
            short_reasons = ["short_weak_waterfall", "waterfall_or_rebound_failure", f"variant:{variant.variant_id}"]
            if wyckoff_short_ok:
                short_reasons = [
                    "wyckoff_short_confirmed",
                    wyckoff_short_setup_tag or "wyckoff_short_setup",
                    f"variant:{variant.variant_id}",
                ]
            return self._enter_v1_3_decision(
                snapshot=snapshot,
                side=Side.SHORT,
                regime=MarketRegime.TOP_DIVERGENCE if setup is EntrySetup.TOP_SHORT else MarketRegime.WEAK_WATERFALL,
                setup=setup,
                entry_position_id="3_wyckoff_distribution_top_short" if setup is EntrySetup.TOP_SHORT else "short_waterfall_continuation",
                market_season=market_season,
                symbol_cycle=symbol_cycle,
                strength=min(1.0, 0.42 + abs(ret_20d) * 0.85 + abs(ret_60d) * 0.20 + historical_match_score * 0.12 + wyckoff_short_score * 0.12),
                reason_codes=short_reasons,
                invalidation_price=stop_loss,
                take_profit_hint=latest_close - risk_per_unit * variant.runner_take_profit_r,
                matched_trade_examples=matched_trade_examples,
                historical_match_score=historical_match_score,
                selection_reason_codes=selection_reason_codes,
                selection_filter_codes=selection_filter_codes,
            )

        if short_intent and variant.enable_countertrend_short and long_trend:
            if symbol_cycle == "first_large_divergence":
                setup = EntrySetup.TOP_SHORT
                entry_position_id = "3_first_large_divergence_top_short"
            elif symbol_cycle == "final_top":
                setup = EntrySetup.FINAL_TOP_SHORT
                entry_position_id = "5_final_top_short"
            else:
                return _skip("skip:countertrend_short_requires_top_structure", [FailureFilter.COUNTER_TREND_SHORT_DISABLED])
            stop_loss = max(high_20d, latest_close * (1 + variant.structure_stop_pct))
            risk_per_unit = max(stop_loss - latest_close, latest_close * 0.01)
            return self._enter_v1_3_decision(
                snapshot=snapshot,
                side=Side.SHORT,
                regime=MarketRegime.TOP_DIVERGENCE,
                setup=setup,
                entry_position_id=entry_position_id,
                market_season=market_season,
                symbol_cycle=symbol_cycle,
                strength=0.34,
                reason_codes=["countertrend_top_short_small_size", f"variant:{variant.variant_id}"],
                invalidation_price=stop_loss,
                take_profit_hint=latest_close - risk_per_unit * variant.partial_take_profit_r,
                matched_trade_examples=matched_trade_examples,
                historical_match_score=historical_match_score,
                selection_reason_codes=selection_reason_codes,
                selection_filter_codes=selection_filter_codes,
                force_no_runner=True,
            )

        if not (long_trend or strong_pattern_long_trend_ok or wyckoff_long_trend_ok):
            return StrategyDecision(
                action=StrategyAction.SKIP,
                explanation="skip:regime=choppy_invalid filters=structure_break",
                matched_historical_patterns=matched_trade_examples,
                risk_notes=["没有形成完整浪浪结构，等待而不是硬做"],
                filter_codes=[FailureFilter.STRUCTURE_BREAK],
            )

        entry_position_id = "1_startup_long"
        regime = MarketRegime.PRE_MAIN_UPTREND
        setup = EntrySetup.STARTER_BUY
        reason_codes = ["leader_main_wave_candidate", "starter_buy", f"variant:{variant.variant_id}"]
        if strong_pattern_tag in {"leader_platform_start", "golden_pit_reclaim"} or leader_platform_start_score >= 0.70 or golden_pit_reclaim_score >= 0.70:
            entry_position_id = "1_startup_long"
            regime = MarketRegime.PRE_MAIN_UPTREND
            setup = EntrySetup.STARTER_BUY
            reason_codes = [
                "strong_pattern_startup",
                strong_pattern_tag or "leader_platform_or_golden_pit",
                f"variant:{variant.variant_id}",
            ]
        elif wyckoff_long_setup_tag in {"spring_reclaim", "sos_breakout", "lps_retest", "reaccumulation_breakout"} or wyckoff_long_score >= 0.68:
            entry_position_id = "1_startup_long" if wyckoff_long_setup_tag in {"spring_reclaim", "sos_breakout", "reaccumulation_breakout"} else "2_small_divergence_long"
            regime = MarketRegime.PRE_MAIN_UPTREND if entry_position_id == "1_startup_long" else MarketRegime.BREAKOUT_RETEST
            setup = EntrySetup.STARTER_BUY if entry_position_id == "1_startup_long" else EntrySetup.PLATFORM_RETEST
            reason_codes = [
                "wyckoff_long_confirmed",
                wyckoff_long_setup_tag or "wyckoff_long_setup",
                f"variant:{variant.variant_id}",
            ]
        elif strong_pattern_tag == "second_wave_start" or second_wave_start_score >= 0.65:
            entry_position_id = "4_second_wave_long"
            regime = MarketRegime.POST_LARGE_DIVERGENCE
            setup = EntrySetup.POST_DIVERGENCE_REBOUND
            reason_codes = ["second_wave_start_pattern", f"variant:{variant.variant_id}"]
        elif strong_pattern_tag == "small_divergence_absorb" or small_divergence_absorb_score >= 0.65:
            entry_position_id = "2_small_divergence_long"
            regime = MarketRegime.FIRST_DIVERGENCE
            setup = EntrySetup.SMALL_DIVERGENCE_ENTRY
            reason_codes = ["daily_main_uptrend", "small_divergence_absorbed", f"variant:{variant.variant_id}"]
        elif symbol_cycle == "platform_start":
            entry_position_id = "1_startup_long"
            regime = MarketRegime.PRE_MAIN_UPTREND
            setup = EntrySetup.STARTER_BUY
            reason_codes = ["leader_platform_start", "starter_buy", f"variant:{variant.variant_id}"]
        elif large_divergence_recent and bottom_lift_confirmed and (
            _ret_gt(features, "m15", m15_ret_8, 0) or _ret_gt(features, "m5", m5_ret_6, 0)
        ):
            entry_position_id = "4_second_wave_long"
            regime = MarketRegime.POST_LARGE_DIVERGENCE
            setup = EntrySetup.POST_DIVERGENCE_REBOUND
            reason_codes = ["post_large_divergence_second_wave", f"variant:{variant.variant_id}"]
        elif symbol_cycle == "box_chop" or _truthy_feature(features, "box_rebound_candidate"):
            entry_position_id = "6_box_rebound_long"
            regime = MarketRegime.CHOPPY_INVALID
            setup = EntrySetup.BOX_REBOUND_LONG
            reason_codes = ["box_rebound_low_confidence", f"variant:{variant.variant_id}"]
        elif symbol_cycle == "small_divergence" or (
            _ret_lt(features, "m15", m15_ret_8, 0) and _ret_ge(features, "m5", m5_ret_6, 0)
        ):
            entry_position_id = "2_small_divergence_long"
            regime = MarketRegime.FIRST_DIVERGENCE
            setup = EntrySetup.SMALL_DIVERGENCE_ENTRY
            reason_codes = ["daily_main_uptrend", "small_divergence_absorbed", f"variant:{variant.variant_id}"]
        elif pullback <= -variant.min_pullback_pct and pullback >= -variant.max_pullback_pct:
            entry_position_id = "2_small_divergence_long"
            regime = MarketRegime.STRONG_PULLBACK
            setup = EntrySetup.FIRST_PULLBACK if h1_pullback <= -variant.min_pullback_pct else EntrySetup.PLATFORM_RETEST
            reason_codes = ["daily_main_uptrend", "pullback_not_broken", f"variant:{variant.variant_id}"]
        elif latest_close >= high_60d * (1 - variant.breakout_tolerance):
            entry_position_id = "2_small_divergence_long"
            regime = MarketRegime.BREAKOUT_RETEST
            setup = EntrySetup.SECOND_PRESSURE_RETEST
            reason_codes = ["daily_main_uptrend", "second_pressure_retest", f"variant:{variant.variant_id}"]

        if setup is EntrySetup.BOX_REBOUND_LONG:
            return _skip("skip:box_rebound_low_quality", [FailureFilter.BOX_REBOUND_LOW_QUALITY])
        if strong_pattern_long_trend_ok and not long_trend:
            reason_codes.append("strong_pattern_trend_substitute")
        if wyckoff_long_trend_ok and not long_trend:
            reason_codes.append("wyckoff_trend_substitute")
        if _ret_ge(features, "m15", m15_ret_8, variant.intraday_confirm_ret_min) or _ret_ge(features, "m5", m5_ret_6, variant.intraday_confirm_ret_min):
            reason_codes.append("intraday_reclaim_confirmed")
        if selection_tag == "leader_altcoin":
            reason_codes.append("leader_altcoin_selected")
        elif variant.leader_only_long and selection_tag == "catch_up_short_hold":
            reason_codes.append("catch_up_short_hold")

        stop_loss = max(low_20d, latest_close * (1 - variant.structure_stop_pct))
        risk_per_unit = max(latest_close - stop_loss, latest_close * 0.01)
        force_no_runner = selection_tag == "catch_up_short_hold" and not variant.catch_up_runner_allowed
        filter_codes = [FailureFilter.CATCH_UP_NO_RUNNER.value] if force_no_runner else [FailureFilter.NO_FAILURE_FILTER.value]
        return self._enter_v1_3_decision(
            snapshot=snapshot,
            side=Side.LONG,
            regime=regime,
            setup=setup,
            entry_position_id=entry_position_id,
            market_season=market_season,
            symbol_cycle=symbol_cycle,
            strength=min(1.0, 0.45 + ret_20d * 0.75 + ret_60d * 0.18 + historical_match_score * 0.12),
            reason_codes=reason_codes,
            invalidation_price=stop_loss,
            take_profit_hint=latest_close + risk_per_unit * (variant.partial_take_profit_r if force_no_runner else variant.runner_take_profit_r),
            matched_trade_examples=matched_trade_examples,
            historical_match_score=historical_match_score,
            selection_reason_codes=selection_reason_codes,
            selection_filter_codes=selection_filter_codes,
            force_no_runner=force_no_runner,
            filter_codes=filter_codes,
        )

    def _enter_v1_3_decision(
        self,
        *,
        snapshot: FeatureSnapshot,
        side: Side,
        regime: MarketRegime,
        setup: EntrySetup,
        entry_position_id: str,
        market_season: str,
        symbol_cycle: str,
        strength: float,
        reason_codes: list[str],
        invalidation_price: float,
        take_profit_hint: float,
        matched_trade_examples: list[dict],
        historical_match_score: float,
        selection_reason_codes: list[str],
        selection_filter_codes: list[str],
        force_no_runner: bool = False,
        filter_codes: list[str] | None = None,
    ) -> StrategyDecision:
        variant = self.variant
        pos_20d = _float_feature(snapshot.features, "pos_20d", 0.5)
        risk_unit = "W"
        size_multiplier = 1.0
        if pos_20d <= variant.low_position_boost_pos and entry_position_id in {"1_startup_long", "4_second_wave_long"}:
            size_multiplier = variant.low_position_size_multiplier
            risk_unit = "1.2W"
        elif pos_20d >= variant.high_position_reduce_pos or force_no_runner:
            risk_unit = "0.6W"
            size_multiplier = variant.high_position_size_multiplier
        leverage_cap = variant.default_anchor_leverage if snapshot.symbol.startswith(("BTC-", "ETH-")) else variant.default_alt_leverage
        risk_notes = ["固定风险单位分三份资金；结构止损先保护第一笔亏损"]
        if force_no_runner:
            risk_notes.append("补涨/箱体/逆势类不允许右尾持有，只按短拿计划处理")
        else:
            risk_notes.append("①/④或高质量主升浪结构保留 runner 捕获右尾")
        entry_family = _variant_entry_family(variant)
        experiment_family = _variant_experiment_family(variant)
        strategy_tree_variant_id = _strategy_tree_variant_id(variant)
        strategy_tree_parent_id = _strategy_tree_parent_id(variant)
        strategy_tree_path = _strategy_tree_path(variant)
        partial_r = variant.partial_take_profit_r
        partial_exit_fraction = 1.0 if force_no_runner else variant.partial_exit_fraction
        runner_r = variant.runner_take_profit_r
        runner_enabled = not force_no_runner
        time_stop_days = min(3, variant.time_stop_days) if force_no_runner else variant.time_stop_days
        payoff_probe = ""
        if entry_family == "payoff_probe":
            partial_r = variant.payoff_probe_partial_r
            runner_r = variant.payoff_probe_runner_r
            time_stop_days = min(variant.payoff_probe_time_stop_days, variant.time_stop_days)
            payoff_probe = "early_partial_breakeven_time_stop"
        decision_trace = {
            "action": StrategyAction.ENTER.value,
            "strategy_version": self.version,
            "strategy_line": self._strategy_line(),
            "experiment_family": experiment_family,
            "entry_family": entry_family,
            "strategy_tree_variant_id": strategy_tree_variant_id,
            "strategy_tree_parent_id": strategy_tree_parent_id,
            "strategy_tree_path": strategy_tree_path,
            "market_season": market_season,
            "symbol_cycle": symbol_cycle,
            "entry_position_id": entry_position_id,
            "regime": regime.value,
            "setup": setup.value,
            "regime_codes": [regime.value],
            "setup_codes": [setup.value, entry_position_id],
            "filter_codes": filter_codes or [FailureFilter.NO_FAILURE_FILTER.value],
            "selection_reason_codes": selection_reason_codes,
            "selection_filter_codes": selection_filter_codes,
            "variant_id": variant.variant_id,
            "risk_unit": risk_unit,
            "position_size_multiplier": size_multiplier,
            "max_leverage": leverage_cap,
            "matched_trade_examples": matched_trade_examples,
            "historical_match_score": historical_match_score,
        }
        if payoff_probe:
            decision_trace["payoff_probe"] = payoff_probe
        signal = LangLangSignal(
            symbol=snapshot.symbol,
            side=side,
            strength=strength,
            reason_codes=reason_codes,
            filter_codes=filter_codes or [FailureFilter.NO_FAILURE_FILTER.value],
            features={
                **snapshot.features,
                **variant.to_dict(),
                "experiment_family": experiment_family,
                "entry_family": entry_family,
                "strategy_tree_variant_id": strategy_tree_variant_id,
                "strategy_tree_parent_id": strategy_tree_parent_id,
                "strategy_tree_path": strategy_tree_path,
                "market_season": market_season,
                "symbol_cycle": symbol_cycle,
                "entry_position_id": entry_position_id,
                "risk_unit": risk_unit,
                "position_size_multiplier": size_multiplier,
                "max_leverage": leverage_cap,
                "matched_trade_examples": matched_trade_examples,
            },
            invalidation_price=invalidation_price,
            stop_loss=invalidation_price,
            take_profit_hint=take_profit_hint,
            take_profit_plan={
                "partial_r": partial_r,
                "partial_exit_fraction": partial_exit_fraction,
                "runner_r": runner_r,
                "risk_unit": risk_unit,
            },
            hold_plan={
                "runner": runner_enabled,
                "time_stop_days": time_stop_days,
                "trend_break_buffer_pct": variant.trend_break_buffer_pct,
                "exit_on": ["structure_break", "mae_stop", "trend_break_exit", "time_stop"],
                "right_tail_required": runner_enabled,
            },
            strategy_version=self.version,
            regime=regime,
            setup=setup,
            decision_trace=decision_trace,
            historical_match_score=historical_match_score,
            matched_trade_examples=matched_trade_examples,
            risk_notes=risk_notes,
            created_at=snapshot.created_at,
        )
        return StrategyDecision(
            action=StrategyAction.ENTER,
            explanation=(
                f"enter:strategy={self.version} season={market_season} cycle={symbol_cycle} "
                f"entry={entry_position_id} side={side.value}"
            ),
            matched_historical_patterns=matched_trade_examples,
            risk_notes=risk_notes,
            filter_codes=[FailureFilter.CATCH_UP_NO_RUNNER] if force_no_runner else [FailureFilter.NO_FAILURE_FILTER],
            signal=signal,
        )


class RulesLangLangNativeStrategy(RulesLangLangV1_3Strategy):
    version = "rules_langlang_native_v1"

    def __init__(self, variant: LangLangNativeVariant | None = None):
        self.variant = variant or LangLangNativeVariant()

    def _strategy_line(self) -> str:
        return "native"

    def _requires_historical_support(self) -> bool:
        return False

    def _uses_enhanced_loss_filters(self) -> bool:
        return False


class RulesLangLangEnhancedStrategy(RulesLangLangV1_3Strategy):
    version = "rules_langlang_enhanced_v1"

    def __init__(self, variant: LangLangEnhancedVariant | None = None):
        self.variant = variant or LangLangEnhancedVariant()

    def _strategy_line(self) -> str:
        return "enhanced"

    def _enhanced_failure_filters(
        self,
        features: dict,
        variant: LangLangEnhancedVariant,
        historical_match_score: float,
        matched_trade_examples: list[dict],
    ) -> tuple[str, list[FailureFilter]] | None:
        big_loss_overlap = int(_float_feature(features, "big_loss_overlap_count", 0.0))
        if big_loss_overlap > variant.max_big_loss_overlap_count:
            return "skip:big_loss_similarity", [FailureFilter.BIG_LOSS_SIMILARITY]
        funding_rate = _float_feature(features, "funding_rate_last", 0.0)
        if funding_rate >= variant.max_funding_rate_last:
            return "skip:funding_overheated", [FailureFilter.CHASE_OVERHEAT]
        turnover_rank = int(_float_feature(features, "turnover_rank_24h", 0.0))
        if turnover_rank > variant.max_turnover_rank_24h:
            return "skip:liquidity_rank_filtered", [FailureFilter.LIQUIDITY_RANK_FILTERED]
        oi_change = _float_feature(features, "oi_change_3d", 0.0)
        if oi_change <= variant.min_oi_expansion_3d and _str_feature(features, "open_interest_status") == "available":
            return "skip:open_interest_collapse", [FailureFilter.VOLUME_REVERSAL]
        return None


class RulesLangLangNativeFinalStrategy(RulesLangLangNativeStrategy):
    version = "rules_langlang_native_final"

    def _strategy_line(self) -> str:
        return "native_final"


class RulesLangLangEnhancedFinalStrategy(RulesLangLangEnhancedStrategy):
    version = "rules_langlang_enhanced_final"

    def _strategy_line(self) -> str:
        return "enhanced_final"


class RulesLangLangNativePayoffStrategy(RulesLangLangNativeStrategy):
    version = "rules_langlang_native_payoff_v1"

    def _strategy_line(self) -> str:
        return "native_payoff_v1"


class RulesLangLangEnhancedPayoffStrategy(RulesLangLangEnhancedStrategy):
    version = "rules_langlang_enhanced_payoff_v1"

    def _strategy_line(self) -> str:
        return "enhanced_payoff_v1"


class RulesLangLangV1_2Strategy(RulesLangLangV1_1Strategy):
    version = "rules_langlang_v1_2"


def strategy_from_version(
    version: str,
    variant: StrategyVariant | LangLangV1Variant | ScalpingVariant | MicroScalpVariant | None = None,
):
    micro_versions = {
        RulesOfiMicropriceScalpStrategy.version: RulesOfiMicropriceScalpStrategy,
        RulesVwapMeanReversionScalpStrategy.version: RulesVwapMeanReversionScalpStrategy,
        RulesVolatilityBreakoutScalpStrategy.version: RulesVolatilityBreakoutScalpStrategy,
        RulesFundingBasisShadowStrategy.version: RulesFundingBasisShadowStrategy,
    }
    if version in micro_versions:
        if variant is not None and not isinstance(variant, MicroScalpVariant):
            variant = MicroScalpVariant(**variant.to_dict())
        return micro_versions[version](
            variant or MicroScalpVariant(variant_id=f"{version}_default", symbol="", strategy_kind=version)
        )
    if version == RulesFiveBarScalpStrategy.version:
        if variant is not None and not isinstance(variant, ScalpingVariant):
            variant = ScalpingVariant(**variant.to_dict())
        return RulesFiveBarScalpStrategy(variant or ScalpingVariant(variant_id="five_bar_scalp_default", symbol=""))
    if version == RulesLangLangEnhancedPayoffStrategy.version:
        if variant is not None and not isinstance(variant, LangLangEnhancedVariant):
            variant = LangLangEnhancedVariant(**variant.to_dict())
        return RulesLangLangEnhancedPayoffStrategy(variant)
    if version == RulesLangLangNativePayoffStrategy.version:
        if variant is not None and not isinstance(variant, LangLangNativeVariant):
            variant = LangLangNativeVariant(**variant.to_dict())
        return RulesLangLangNativePayoffStrategy(variant)
    if version == RulesLangLangEnhancedFinalStrategy.version:
        if variant is not None and not isinstance(variant, LangLangEnhancedVariant):
            variant = LangLangEnhancedVariant(**variant.to_dict())
        return RulesLangLangEnhancedFinalStrategy(variant)
    if version == RulesLangLangNativeFinalStrategy.version:
        if variant is not None and not isinstance(variant, LangLangNativeVariant):
            variant = LangLangNativeVariant(**variant.to_dict())
        return RulesLangLangNativeFinalStrategy(variant)
    if version == RulesLangLangEnhancedStrategy.version:
        if variant is not None and not isinstance(variant, LangLangEnhancedVariant):
            variant = LangLangEnhancedVariant(**variant.to_dict())
        return RulesLangLangEnhancedStrategy(variant)
    if version == RulesLangLangNativeStrategy.version:
        if variant is not None and not isinstance(variant, LangLangNativeVariant):
            variant = LangLangNativeVariant(**variant.to_dict())
        return RulesLangLangNativeStrategy(variant)
    if version == RulesLangLangV1_3Strategy.version:
        if variant is not None and not isinstance(variant, LangLangV1_3Variant):
            variant = LangLangV1_3Variant(**variant.to_dict())
        return RulesLangLangV1_3Strategy(variant)
    if version == RulesLangLangV1_2Strategy.version:
        if variant is not None and not isinstance(variant, LangLangV1_1Variant):
            variant = LangLangV1_1Variant(**variant.to_dict())
        return RulesLangLangV1_2Strategy(variant)
    if version == RulesLangLangV1_1Strategy.version:
        if variant is not None and not isinstance(variant, LangLangV1_1Variant):
            variant = LangLangV1_1Variant(**variant.to_dict())
        return RulesLangLangV1_1Strategy(variant)
    if version == RulesLangLangV1Strategy.version:
        if variant is not None and not isinstance(variant, LangLangV1Variant):
            variant = LangLangV1Variant(**variant.to_dict())
        return RulesLangLangV1Strategy(variant)
    if variant is not None and not isinstance(variant, StrategyVariant):
        variant = StrategyVariant(**_v01_fields_from_v1(variant))
    return RulesV01Strategy(variant)


def _v01_fields_from_v1(variant: LangLangV1Variant) -> dict:
    return {
        "variant_id": variant.variant_id,
        "ret_20d_min": variant.ret_20d_min,
        "ret_60d_min": variant.ret_60d_min,
        "pos_20d_min": variant.pos_20d_min,
        "max_pullback_pct": variant.max_pullback_pct,
        "breakout_tolerance": variant.breakout_tolerance,
    }


def _float_feature(features: dict, key: str, default: float = 0.0) -> float:
    value = features.get(key, default)
    if value in {None, ""}:
        return default
    return float(value)


def _skip(explanation: str, filters: list[FailureFilter]) -> StrategyDecision:
    return StrategyDecision(
        action=StrategyAction.SKIP,
        explanation=explanation,
        matched_historical_patterns=[],
        risk_notes=[],
        filter_codes=filters,
    )


def _variant_entry_family(variant: object) -> str:
    return str(getattr(variant, "entry_family", "") or "default")


def _variant_experiment_family(variant: object) -> str:
    return str(getattr(variant, "experiment_family", "") or "")


def _strategy_tree_variant_id(variant: object) -> str:
    return str(getattr(variant, "strategy_tree_variant_id", "") or getattr(variant, "variant_id", ""))


def _strategy_tree_parent_id(variant: object) -> str:
    parent_id = str(getattr(variant, "strategy_tree_parent_id", "") or "")
    if parent_id:
        return parent_id
    if _variant_experiment_family(variant) == "orthogonal_v1":
        return "langlang_plus_01_loss"
    return ""


def _strategy_tree_path(variant: object) -> list[str]:
    variant_id = _strategy_tree_variant_id(variant)
    if _variant_experiment_family(variant) == "orthogonal_v1":
        return ["langlang_01", "langlang_plus_01", "langlang_plus_01_loss", variant_id]
    parent_id = _strategy_tree_parent_id(variant)
    if parent_id:
        return [parent_id, variant_id]
    return [variant_id] if variant_id else []


def _list_feature(features: dict, key: str) -> list[dict]:
    value = features.get(key, [])
    if not value:
        return []
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _string_list_feature(features: dict, key: str) -> list[str]:
    value = features.get(key, [])
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    if isinstance(value, str) and value:
        return [item.strip() for item in value.split(",") if item.strip()]
    return []


def _str_feature(features: dict, key: str, default: str = "") -> str:
    value = features.get(key, default)
    if value in {None, ""}:
        return default
    return str(value)


def _truthy_feature(features: dict, key: str) -> bool:
    value = features.get(key, False)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _bar_data_available(features: dict, prefix: str) -> bool:
    return bool(features.get(f"{prefix}_data_available", True))


def _ret_ge(features: dict, prefix: str, value: float, threshold: float) -> bool:
    return _bar_data_available(features, prefix) and value >= threshold


def _ret_gt(features: dict, prefix: str, value: float, threshold: float) -> bool:
    return _bar_data_available(features, prefix) and value > threshold


def _ret_le(features: dict, prefix: str, value: float, threshold: float) -> bool:
    return _bar_data_available(features, prefix) and value <= threshold


def _ret_lt(features: dict, prefix: str, value: float, threshold: float) -> bool:
    return _bar_data_available(features, prefix) and value < threshold


def _strong_pattern_long_trend_ok(
    *,
    features: dict,
    variant: LangLangV1_3Variant,
    ret_20d: float,
    ret_60d: float,
    pos_20d: float,
    latest_close: float,
    ma_5: float,
    ma_20: float,
    vol_ratio: float,
    m15_ret_8: float,
    m5_ret_6: float,
    bottom_lift_confirmed: bool,
) -> bool:
    strong_tag = _str_feature(features, "strong_pattern_tag", "")
    if strong_tag == "spoon_bottom_confirmed":
        return False
    if strong_tag not in {"leader_platform_start", "golden_pit_reclaim", "small_divergence_absorb", "second_wave_start"}:
        return False
    if _float_feature(features, "risk_pattern_score") >= 0.65 or _str_feature(features, "risk_pattern_tag", ""):
        return False
    if ma_20 <= 0 or latest_close < ma_20 * 0.985:
        return False
    if pos_20d < 0.38 or ret_60d < 0.18 or vol_ratio < variant.min_vol_ratio_20d:
        return False
    missed_trend_conditions = sum(
        (
            ret_20d < variant.ret_20d_min,
            ret_60d < variant.ret_60d_min,
            pos_20d < variant.pos_20d_min,
            latest_close < ma_20,
            ma_5 < ma_20,
        )
    )
    if missed_trend_conditions > 2:
        return False

    if strong_tag == "leader_platform_start":
        return _float_feature(features, "leader_platform_start_score") >= 0.70
    if strong_tag == "golden_pit_reclaim":
        return _float_feature(features, "golden_pit_reclaim_score") >= 0.70 and _has_intraday_pattern_confirmation(
            features,
            "golden_pit_reclaim",
            variant,
            m15_ret_8,
            m5_ret_6,
            bottom_lift_confirmed,
        )
    if strong_tag == "small_divergence_absorb":
        return _float_feature(features, "small_divergence_absorb_score") >= 0.65 and _has_intraday_pattern_confirmation(
            features,
            "small_divergence_absorb",
            variant,
            m15_ret_8,
            m5_ret_6,
            bottom_lift_confirmed,
        )
    if strong_tag == "second_wave_start":
        return _float_feature(features, "second_wave_start_score") >= 0.65 and _has_intraday_pattern_confirmation(
            features,
            "second_wave_start",
            variant,
            m15_ret_8,
            m5_ret_6,
            bottom_lift_confirmed,
        )
    return False


def _has_intraday_pattern_confirmation(
    features: dict,
    tag: str,
    variant: LangLangV1_3Variant,
    m15_ret_8: float,
    m5_ret_6: float,
    bottom_lift_confirmed: bool,
) -> bool:
    if tag == "second_wave_start" and bottom_lift_confirmed:
        return True
    score_key = f"{tag}_score"
    if max(
        _float_feature(features, f"h1_{score_key}"),
        _float_feature(features, f"m15_{score_key}"),
        _float_feature(features, f"m5_{score_key}"),
    ) >= 0.45:
        return True
    reason_codes = _string_list_feature(features, "pattern_reason_codes")
    if f"{tag}_intraday_reclaim_confirmed" in reason_codes or f"{tag}_intraday_absorb_confirmed" in reason_codes:
        return True
    return _ret_ge(features, "m15", m15_ret_8, variant.intraday_confirm_ret_min) or _ret_ge(features, "m5", m5_ret_6, variant.intraday_confirm_ret_min)


def _wyckoff_long_trend_ok(
    *,
    features: dict,
    variant: LangLangV1_3Variant,
    ret_20d: float,
    ret_60d: float,
    pos_20d: float,
    latest_close: float,
    ma_5: float,
    ma_20: float,
    vol_ratio: float,
    m15_ret_8: float,
    m5_ret_6: float,
) -> bool:
    tag = _str_feature(features, "wyckoff_long_setup_tag", "")
    if tag not in {"spring_reclaim", "sos_breakout", "lps_retest", "reaccumulation_breakout"}:
        return False
    if _float_feature(features, "wyckoff_long_score") < 0.68:
        return False
    if _float_feature(features, "wyckoff_risk_score") >= 0.70 or _float_feature(features, "wyckoff_exit_score") >= 0.70:
        return False
    if _str_feature(features, "wyckoff_phase_tag", "") == "distribution":
        return False
    if ma_20 <= 0 or latest_close < ma_20 * 0.985:
        return False
    if pos_20d < 0.38 or ret_60d < 0.18 or vol_ratio < variant.min_vol_ratio_20d:
        return False
    return _has_wyckoff_intraday_confirmation(features, "long", variant, m15_ret_8, m5_ret_6)


def _wyckoff_short_trend_ok(
    *,
    features: dict,
    variant: LangLangV1_3Variant,
    m15_ret_8: float,
    m5_ret_6: float,
) -> bool:
    tag = _str_feature(features, "wyckoff_short_setup_tag", "")
    if tag not in {"upthrust_reversal", "utad_risk", "sow_breakdown", "lpsy_retest"}:
        return False
    if _float_feature(features, "wyckoff_short_score") < 0.70:
        return False
    return _has_wyckoff_intraday_confirmation(features, "short", variant, m15_ret_8, m5_ret_6)


def _has_wyckoff_intraday_confirmation(
    features: dict,
    side: str,
    variant: LangLangV1_3Variant,
    m15_ret_8: float,
    m5_ret_6: float,
) -> bool:
    score_key = f"wyckoff_{side}_score"
    if max(
        _float_feature(features, f"h1_{score_key}"),
        _float_feature(features, f"m15_{score_key}"),
        _float_feature(features, f"m5_{score_key}"),
    ) >= 0.45:
        return True
    if side == "long":
        return _ret_ge(features, "m15", m15_ret_8, variant.intraday_confirm_ret_min) or _ret_ge(features, "m5", m5_ret_6, variant.intraday_confirm_ret_min)
    return _ret_le(features, "m15", m15_ret_8, variant.intraday_breakdown_ret_max) or _ret_le(features, "m5", m5_ret_6, -abs(variant.intraday_confirm_ret_min))


def _infer_market_season(features: dict) -> str:
    explicit = _str_feature(features, "market_season", "")
    if explicit:
        return explicit
    btc_ret_20d = _float_feature(features, "btc_ret_20d", _float_feature(features, "market_btc_ret_20d", 0.0))
    btc_ret_60d = _float_feature(features, "btc_ret_60d", _float_feature(features, "market_btc_ret_60d", 0.0))
    btc_pos_20d = _float_feature(features, "btc_pos_20d", _float_feature(features, "market_btc_pos_20d", 0.5))
    if btc_ret_20d >= 0.08 and btc_ret_60d >= 0.18 and btc_pos_20d >= 0.58:
        return "summer"
    if btc_ret_20d >= 0 and btc_ret_60d >= 0:
        return "spring"
    if btc_pos_20d >= 0.70 and btc_ret_20d < 0:
        return "autumn"
    if btc_ret_20d < 0 and btc_ret_60d < 0:
        return "winter"
    return "spring"


def _infer_symbol_cycle(features: dict) -> str:
    explicit = _str_feature(features, "symbol_cycle", "")
    if explicit:
        return explicit
    ret_20d = _float_feature(features, "ret_20d")
    ret_60d = _float_feature(features, "ret_60d")
    pos_20d = _float_feature(features, "pos_20d", 0.5)
    pullback = _float_feature(features, "pullback_from_20d_high")
    if ret_20d < -0.15 and pos_20d <= 0.30:
        return "weak_waterfall"
    if bool(features.get("large_divergence_recent")) and bool(features.get("bottom_lift_confirmed")):
        return "second_wave"
    if bool(features.get("large_divergence_recent")):
        return "first_large_divergence"
    if bool(features.get("box_rebound_candidate")):
        return "box_chop"
    if ret_20d >= 0.20 and ret_60d >= 0.40 and -0.16 <= pullback <= -0.015:
        return "small_divergence"
    if ret_20d >= 0.20 and ret_60d >= 0.40:
        return "main_wave"
    return "box_chop"


def _variant_allowed_side(variant: LangLangV1_1Variant) -> str:
    allowed = str(getattr(variant, "allowed_side", "both") or "both").lower()
    variant_id = variant.variant_id.lower()
    if allowed == "both":
        if variant_id.startswith(("llv1_1_long_", "llv1_3_long_")):
            return "long"
        if variant_id.startswith(("llv1_1_short_", "llv1_3_short_")):
            return "short"
    if allowed in {"long", "short", "both"}:
        return allowed
    if variant_id.startswith(("llv1_1_long_", "llv1_3_long_")):
        return "long"
    if variant_id.startswith(("llv1_1_short_", "llv1_3_short_")):
        return "short"
    return "both"


def _intent_side(*, requested_side: str, allowed_side: str, selection_mode: str, selection_tag: str) -> str:
    if requested_side in {"long", "short"}:
        return requested_side
    if allowed_side in {"long", "short"}:
        return allowed_side
    if selection_mode == "short_waterfall" or selection_tag == "short_waterfall":
        return "short"
    return "long"

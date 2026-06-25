import unittest

from langlang_trader.features import FeatureSnapshot
from langlang_trader.models import EntrySetup, FailureFilter, MarketRegime, Side, StrategyAction
from langlang_trader.strategy import (
    LangLangV1_3Variant,
    RulesLangLangV1_3Strategy,
    default_langlang_v1_3_grid,
    strategy_from_version,
)


def snapshot(**overrides):
    features = {
        "ret_20d": 0.38,
        "ret_60d": 0.95,
        "pos_20d": 0.72,
        "pullback_from_20d_high": -0.055,
        "ma_5": 124.0,
        "ma_20": 110.0,
        "latest_close": 122.0,
        "high_20d": 130.0,
        "low_20d": 92.0,
        "high_60d": 155.0,
        "latest_volume": 2600.0,
        "vol_ratio_20d": 1.9,
        "h1_ret_24": 0.028,
        "h1_pos_48": 0.62,
        "h1_pullback_from_high": -0.032,
        "m15_ret_8": 0.010,
        "m15_pos_32": 0.66,
        "m5_ret_6": 0.006,
        "upside_space_pct": 0.28,
        "first_10x_entry_done": False,
        "bottom_lift_confirmed": True,
        "stop_loss_cluster_24h": 0,
        "historical_match_score": 0.72,
        "matched_trade_examples": [{"trade_id": "big-win-1", "labels": ["big_win"], "score": 0.88}],
        "market_season": "summer",
        "symbol_cycle": "platform_start",
        "selection_mode": "long_main_wave",
        "selection_reason_codes": ["leader_altcoin", "btc_first_wave_follow", "btc_contraction_resilient"],
        "selection_filter_codes": [],
        "symbol_selection_tag": "leader_altcoin",
    }
    features.update(overrides)
    return FeatureSnapshot(
        symbol="WAVE-USDT-SWAP",
        bar="multi",
        last_ts=1_710_000_000_000,
        created_at="2024-03-12T00:00:00+00:00",
        features=features,
    )


class LangLangV13StrategyTest(unittest.TestCase):
    def test_position_1_startup_long_carries_original_document_trace(self):
        strategy = RulesLangLangV1_3Strategy(LangLangV1_3Variant(variant_id="v1.3-test"))

        decision = strategy.decide(snapshot(symbol_cycle="platform_start"))

        self.assertEqual(decision.action, StrategyAction.ENTER)
        signal = decision.signal
        self.assertIsNotNone(signal)
        self.assertEqual(signal.strategy_version, "rules_langlang_v1_3")
        self.assertEqual(signal.side, Side.LONG)
        self.assertEqual(signal.regime, MarketRegime.PRE_MAIN_UPTREND)
        self.assertEqual(signal.setup, EntrySetup.STARTER_BUY)
        self.assertEqual(signal.decision_trace["market_season"], "summer")
        self.assertEqual(signal.decision_trace["symbol_cycle"], "platform_start")
        self.assertEqual(signal.decision_trace["entry_position_id"], "1_startup_long")
        self.assertEqual(signal.decision_trace["risk_unit"], "W")
        self.assertIn("selection_reason_codes", signal.decision_trace)
        self.assertTrue(signal.hold_plan["runner"])

    def test_missing_m5_data_does_not_fake_small_divergence_absorption(self):
        strategy = RulesLangLangV1_3Strategy(LangLangV1_3Variant(variant_id="v1.3-test"))

        decision = strategy.decide(
            snapshot(
                symbol_cycle="main_wave",
                pullback_from_20d_high=-0.01,
                high_60d=180.0,
                m15_ret_8=-0.006,
                m15_data_available=True,
                m5_ret_6=0.0,
                m5_data_available=False,
            )
        )

        self.assertEqual(decision.action, StrategyAction.ENTER)
        self.assertEqual(decision.signal.decision_trace["entry_position_id"], "1_startup_long")
        self.assertNotIn("small_divergence_absorbed", decision.signal.reason_codes)
        self.assertNotIn("intraday_reclaim_confirmed", decision.signal.reason_codes)

    def test_position_2_third_small_divergence_is_skipped(self):
        strategy = RulesLangLangV1_3Strategy(LangLangV1_3Variant(variant_id="v1.3-test"))

        decision = strategy.decide(
            snapshot(
                symbol_cycle="small_divergence",
                small_divergence_count=3,
                m15_ret_8=-0.004,
                m5_ret_6=0.008,
            )
        )

        self.assertEqual(decision.action, StrategyAction.SKIP)
        self.assertIn(FailureFilter.THIRD_SMALL_DIVERGENCE, decision.filter_codes)

    def test_position_4_second_wave_gets_runner_after_large_divergence_bottom_lift(self):
        strategy = RulesLangLangV1_3Strategy(LangLangV1_3Variant(variant_id="v1.3-test"))

        decision = strategy.decide(
            snapshot(
                symbol_cycle="second_wave",
                large_divergence_recent=True,
                bottom_lift_confirmed=True,
                m15_ret_8=0.018,
            )
        )

        self.assertEqual(decision.action, StrategyAction.ENTER)
        self.assertEqual(decision.signal.regime, MarketRegime.POST_LARGE_DIVERGENCE)
        self.assertEqual(decision.signal.setup, EntrySetup.POST_DIVERGENCE_REBOUND)
        self.assertEqual(decision.signal.decision_trace["entry_position_id"], "4_second_wave_long")
        self.assertTrue(decision.signal.hold_plan["runner"])

    def test_main_wave_countertrend_short_is_forbidden_by_default(self):
        strategy = RulesLangLangV1_3Strategy(LangLangV1_3Variant(variant_id="v1.3-test"))

        decision = strategy.decide(
            snapshot(
                requested_side="short",
                symbol_cycle="main_wave",
                ret_20d=0.50,
                ret_60d=1.10,
                pos_20d=0.84,
                m15_ret_8=-0.012,
                m5_ret_6=-0.004,
            )
        )

        self.assertEqual(decision.action, StrategyAction.SKIP)
        self.assertIn(FailureFilter.COUNTER_TREND_SHORT_DISABLED, decision.filter_codes)

    def test_catch_up_coin_is_allowed_only_as_short_hold_not_runner(self):
        strategy = RulesLangLangV1_3Strategy(LangLangV1_3Variant(variant_id="v1.3-test"))

        decision = strategy.decide(
            snapshot(
                symbol_selection_tag="catch_up_short_hold",
                selection_reason_codes=["catch_up_short_hold", "btc_divergence_alt_rotation"],
                symbol_cycle="main_wave",
            )
        )

        self.assertEqual(decision.action, StrategyAction.ENTER)
        self.assertFalse(decision.signal.hold_plan["runner"])
        self.assertIn("catch_up_short_hold", decision.signal.decision_trace["selection_reason_codes"])
        self.assertIn("catch_up_no_runner", decision.signal.filter_codes)

    def test_autumn_winter_reduces_frequency_unless_top_quality_position_1_or_4(self):
        strategy = RulesLangLangV1_3Strategy(LangLangV1_3Variant(variant_id="v1.3-test"))

        weak_autumn = strategy.decide(
            snapshot(
                market_season="autumn",
                symbol_cycle="box_chop",
                box_rebound_candidate=True,
                selection_reason_codes=["box_rebound"],
            )
        )
        high_quality_autumn = strategy.decide(
            snapshot(
                market_season="autumn",
                symbol_cycle="second_wave",
                large_divergence_recent=True,
                bottom_lift_confirmed=True,
                m15_ret_8=0.018,
            )
        )

        self.assertEqual(weak_autumn.action, StrategyAction.SKIP)
        self.assertIn(FailureFilter.AUTUMN_WINTER_REDUCED_FREQUENCY, weak_autumn.filter_codes)
        self.assertEqual(high_quality_autumn.action, StrategyAction.ENTER)
        self.assertEqual(high_quality_autumn.signal.decision_trace["entry_position_id"], "4_second_wave_long")

    def test_default_grid_and_factory_support_v1_3(self):
        variants = default_langlang_v1_3_grid()

        self.assertGreaterEqual(len(variants), 10)
        self.assertTrue(any(variant.allowed_side == "long" for variant in variants))
        self.assertTrue(any(variant.allowed_side == "short" for variant in variants))
        self.assertIsInstance(strategy_from_version("rules_langlang_v1_3", variants[0]), RulesLangLangV1_3Strategy)

    def test_strong_pattern_tags_map_to_langlang_entry_positions(self):
        strategy = RulesLangLangV1_3Strategy(LangLangV1_3Variant(variant_id="v1.3-test"))

        golden_pit = strategy.decide(
            snapshot(
                symbol_cycle="box_chop",
                strong_pattern_tag="golden_pit_reclaim",
                golden_pit_reclaim_score=0.78,
                strong_pattern_score=0.78,
                pattern_reason_codes=["golden_pit_fast_reclaim"],
            )
        )
        small_divergence = strategy.decide(
            snapshot(
                symbol_cycle="main_wave",
                strong_pattern_tag="small_divergence_absorb",
                small_divergence_absorb_score=0.70,
                strong_pattern_score=0.70,
                pattern_reason_codes=["small_divergence_absorbed"],
            )
        )
        second_wave = strategy.decide(
            snapshot(
                symbol_cycle="main_wave",
                strong_pattern_tag="second_wave_start",
                second_wave_start_score=0.72,
                strong_pattern_score=0.72,
                pattern_reason_codes=["large_divergence_bottom_lift"],
            )
        )

        self.assertEqual(golden_pit.action, StrategyAction.ENTER)
        self.assertEqual(golden_pit.signal.decision_trace["entry_position_id"], "1_startup_long")
        self.assertEqual(small_divergence.signal.decision_trace["entry_position_id"], "2_small_divergence_long")
        self.assertEqual(second_wave.signal.decision_trace["entry_position_id"], "4_second_wave_long")

    def test_risk_pattern_overrides_positive_pattern_and_skips(self):
        strategy = RulesLangLangV1_3Strategy(LangLangV1_3Variant(variant_id="v1.3-test"))

        decision = strategy.decide(
            snapshot(
                symbol_cycle="platform_start",
                strong_pattern_tag="leader_platform_start",
                leader_platform_start_score=0.78,
                strong_pattern_score=0.78,
                risk_pattern_tag="false_breakout_risk",
                false_breakout_risk_score=0.70,
                risk_pattern_score=0.70,
                pattern_reason_codes=["leader_platform_start", "false_breakout_fell_back_into_box"],
            )
        )

        self.assertEqual(decision.action, StrategyAction.SKIP)
        self.assertIn(FailureFilter.FALSE_BREAKOUT_AFTER_CONTRACTION, decision.filter_codes)

    def test_five_wave_late_risk_skips_longs(self):
        strategy = RulesLangLangV1_3Strategy(LangLangV1_3Variant(variant_id="v1.3-test"))

        decision = strategy.decide(
            snapshot(
                symbol_cycle="main_wave",
                five_wave_late_risk_score=0.74,
                risk_pattern_tag="five_wave_late_risk",
                risk_pattern_score=0.74,
                wave_push_count=5,
            )
        )

        self.assertEqual(decision.action, StrategyAction.SKIP)
        self.assertIn(FailureFilter.FIVE_WAVE_LATE_RISK, decision.filter_codes)

    def test_golden_pit_can_safely_substitute_lagging_ma_trend(self):
        strategy = RulesLangLangV1_3Strategy(LangLangV1_3Variant(variant_id="v1.3-test"))

        decision = strategy.decide(
            snapshot(
                symbol_cycle="box_chop",
                latest_close=109.0,
                ma_5=104.0,
                ma_20=110.0,
                strong_pattern_tag="golden_pit_reclaim",
                golden_pit_reclaim_score=0.78,
                strong_pattern_score=0.78,
                h1_golden_pit_reclaim_score=0.50,
                pattern_reason_codes=["golden_pit_fast_reclaim", "golden_pit_intraday_reclaim_confirmed"],
            )
        )

        self.assertEqual(decision.action, StrategyAction.ENTER)
        self.assertEqual(decision.signal.decision_trace["entry_position_id"], "1_startup_long")
        self.assertIn("strong_pattern_trend_substitute", decision.signal.reason_codes)

    def test_second_wave_can_safely_substitute_lagging_ret_20d(self):
        strategy = RulesLangLangV1_3Strategy(LangLangV1_3Variant(variant_id="v1.3-test"))

        decision = strategy.decide(
            snapshot(
                symbol_cycle="main_wave",
                ret_20d=0.08,
                ret_60d=0.52,
                pos_20d=0.58,
                strong_pattern_tag="second_wave_start",
                second_wave_start_score=0.72,
                strong_pattern_score=0.72,
                m15_ret_8=0.012,
                m15_second_wave_start_score=0.48,
                pattern_reason_codes=["large_divergence_bottom_lift", "second_wave_reclaim"],
            )
        )

        self.assertEqual(decision.action, StrategyAction.ENTER)
        self.assertEqual(decision.signal.decision_trace["entry_position_id"], "4_second_wave_long")
        self.assertIn("strong_pattern_trend_substitute", decision.signal.reason_codes)

    def test_missing_trend_without_strong_pattern_still_skips(self):
        strategy = RulesLangLangV1_3Strategy(LangLangV1_3Variant(variant_id="v1.3-test"))

        decision = strategy.decide(
            snapshot(
                symbol_cycle="main_wave",
                ma_5=104.0,
                ma_20=110.0,
                latest_close=109.0,
            )
        )

        self.assertEqual(decision.action, StrategyAction.SKIP)
        self.assertIn(FailureFilter.STRUCTURE_BREAK, decision.filter_codes)

    def test_spoon_bottom_does_not_substitute_missing_trend(self):
        strategy = RulesLangLangV1_3Strategy(LangLangV1_3Variant(variant_id="v1.3-test"))

        decision = strategy.decide(
            snapshot(
                symbol_cycle="box_chop",
                latest_close=109.0,
                ma_5=104.0,
                ma_20=110.0,
                strong_pattern_tag="spoon_bottom_confirmed",
                spoon_bottom_confirmed_score=0.72,
                strong_pattern_score=0.72,
                pattern_reason_codes=["spoon_bottom_confirmed"],
            )
        )

        self.assertEqual(decision.action, StrategyAction.SKIP)
        self.assertIn(FailureFilter.STRUCTURE_BREAK, decision.filter_codes)

    def test_wyckoff_spring_can_safely_substitute_lagging_trend(self):
        strategy = RulesLangLangV1_3Strategy(LangLangV1_3Variant(variant_id="v1.3-test"))

        decision = strategy.decide(
            snapshot(
                symbol_cycle="box_chop",
                ret_20d=0.10,
                ret_60d=0.38,
                pos_20d=0.50,
                latest_close=109.0,
                ma_5=104.0,
                ma_20=110.0,
                wyckoff_phase_tag="accumulation",
                wyckoff_long_setup_tag="spring_reclaim",
                wyckoff_long_score=0.72,
                h1_wyckoff_long_score=0.52,
                wyckoff_reason_codes=["wyckoff_spring_reclaim"],
            )
        )

        self.assertEqual(decision.action, StrategyAction.ENTER)
        self.assertEqual(decision.signal.decision_trace["entry_position_id"], "1_startup_long")
        self.assertIn("wyckoff_long_confirmed", decision.signal.reason_codes)
        self.assertIn("wyckoff_trend_substitute", decision.signal.reason_codes)

    def test_wyckoff_distribution_risk_blocks_long_entry(self):
        strategy = RulesLangLangV1_3Strategy(LangLangV1_3Variant(variant_id="v1.3-test"))

        decision = strategy.decide(
            snapshot(
                symbol_cycle="platform_start",
                strong_pattern_tag="leader_platform_start",
                leader_platform_start_score=0.78,
                strong_pattern_score=0.78,
                wyckoff_phase_tag="distribution",
                wyckoff_risk_score=0.74,
                wyckoff_exit_score=0.72,
                wyckoff_exit_tag="utad_risk",
                wyckoff_reason_codes=["wyckoff_utad_risk"],
            )
        )

        self.assertEqual(decision.action, StrategyAction.SKIP)
        self.assertIn(FailureFilter.WYCKOFF_RISK, decision.filter_codes)

    def test_wyckoff_exit_reduces_open_long_position(self):
        strategy = RulesLangLangV1_3Strategy(LangLangV1_3Variant(variant_id="v1.3-test"))

        decision = strategy.decide(
            snapshot(
                current_position_side="long",
                symbol_cycle="platform_start",
                wyckoff_phase_tag="distribution",
                wyckoff_exit_score=0.72,
                wyckoff_exit_tag="utad_risk",
                wyckoff_risk_score=0.74,
                wyckoff_reason_codes=["wyckoff_utad_risk"],
            )
        )

        self.assertEqual(decision.action, StrategyAction.REDUCE)
        self.assertIn(FailureFilter.WYCKOFF_RISK, decision.filter_codes)
        self.assertIn("wyckoff_exit", decision.explanation)

    def test_wyckoff_short_setup_can_trigger_short_entry_when_side_allows(self):
        strategy = RulesLangLangV1_3Strategy(
            LangLangV1_3Variant(variant_id="v1.3-wyck-short", allowed_side="short")
        )

        decision = strategy.decide(
            snapshot(
                requested_side="short",
                symbol_cycle="box_chop",
                ret_20d=-0.08,
                ret_60d=0.18,
                pos_20d=0.40,
                latest_close=96.0,
                ma_5=98.0,
                ma_20=104.0,
                high_20d=112.0,
                low_20d=90.0,
                wyckoff_phase_tag="distribution",
                wyckoff_short_setup_tag="sow_breakdown",
                wyckoff_short_score=0.74,
                m15_wyckoff_short_score=0.50,
                wyckoff_reason_codes=["wyckoff_sow_breakdown"],
            )
        )

        self.assertEqual(decision.action, StrategyAction.ENTER)
        self.assertEqual(decision.signal.side, Side.SHORT)
        self.assertIn(decision.signal.setup, {EntrySetup.TOP_SHORT, EntrySetup.WATERFALL_CONTINUATION})
        self.assertIn("wyckoff_short_confirmed", decision.signal.reason_codes)


if __name__ == "__main__":
    unittest.main()

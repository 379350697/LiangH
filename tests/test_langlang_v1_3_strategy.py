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


if __name__ == "__main__":
    unittest.main()

import unittest

from langlang_trader.features import FeatureSnapshot
from langlang_trader.models import EntrySetup, MarketRegime, Side, StrategyAction
from langlang_trader.strategy import LangLangV1Variant, RulesLangLangV1Strategy, default_langlang_v1_grid


def feature_snapshot(**overrides):
    features = {
        "ret_20d": 0.28,
        "ret_60d": 0.70,
        "pos_20d": 0.72,
        "pullback_from_20d_high": -0.06,
        "ma_5": 124.0,
        "ma_20": 110.0,
        "latest_close": 122.0,
        "high_20d": 130.0,
        "low_20d": 92.0,
        "high_60d": 131.0,
        "latest_volume": 2200.0,
        "vol_ratio_20d": 1.8,
        "h1_ret_24": 0.035,
        "h1_pos_48": 0.64,
        "h1_pullback_from_high": -0.035,
        "m15_ret_8": 0.012,
        "m15_pos_32": 0.68,
        "m5_ret_6": 0.006,
        "m5_pos_24": 0.66,
    }
    features.update(overrides)
    return FeatureSnapshot(
        symbol="BTC-USDT-SWAP",
        bar="multi",
        last_ts=1_710_000_000_000,
        created_at="2024-03-12T00:00:00+00:00",
        features=features,
    )


class LangLangV1StrategyTest(unittest.TestCase):
    def test_main_uptrend_pullback_creates_explainable_long_signal(self):
        strategy = RulesLangLangV1Strategy(LangLangV1Variant(variant_id="v1-test"))

        decision = strategy.decide(feature_snapshot())

        self.assertEqual(decision.action, StrategyAction.ENTER)
        self.assertIsNotNone(decision.signal)
        self.assertEqual(decision.signal.side, Side.LONG)
        self.assertEqual(decision.signal.strategy_version, "rules_langlang_v1")
        self.assertEqual(decision.signal.regime, MarketRegime.STRONG_PULLBACK)
        self.assertEqual(decision.signal.setup, EntrySetup.FIRST_PULLBACK)
        self.assertIn("daily_main_uptrend", decision.signal.reason_codes)
        self.assertIn("intraday_reclaim_confirmed", decision.signal.reason_codes)
        self.assertGreater(decision.signal.stop_loss, 0)
        self.assertIn("runner", decision.signal.hold_plan)
        self.assertIn("regime=strong_pullback", decision.explanation)

    def test_weak_waterfall_creates_explainable_short_signal(self):
        strategy = RulesLangLangV1Strategy(LangLangV1Variant(variant_id="v1-test"))

        decision = strategy.decide(
            feature_snapshot(
                ret_20d=-0.24,
                ret_60d=-0.48,
                pos_20d=0.18,
                pullback_from_20d_high=-0.42,
                ma_5=82.0,
                ma_20=96.0,
                latest_close=78.0,
                high_20d=132.0,
                low_20d=74.0,
                high_60d=150.0,
                h1_ret_24=-0.075,
                h1_pos_48=0.22,
                m15_ret_8=-0.018,
                m5_ret_6=-0.007,
            )
        )

        self.assertEqual(decision.action, StrategyAction.ENTER)
        self.assertIsNotNone(decision.signal)
        self.assertEqual(decision.signal.side, Side.SHORT)
        self.assertEqual(decision.signal.regime, MarketRegime.WEAK_WATERFALL)
        self.assertIn(decision.signal.setup, {EntrySetup.WATERFALL_CONTINUATION, EntrySetup.SHORT_REBOUND_FAILURE})
        self.assertGreater(decision.signal.invalidation_price, 78.0)
        self.assertIn("short_weak_waterfall", decision.signal.reason_codes)

    def test_overheated_chase_is_skipped_with_filter_codes(self):
        strategy = RulesLangLangV1Strategy(LangLangV1Variant(variant_id="v1-test"))

        decision = strategy.decide(
            feature_snapshot(
                ret_20d=0.62,
                ret_60d=1.20,
                pos_20d=0.98,
                pullback_from_20d_high=-0.002,
                h1_ret_24=0.09,
                h1_pos_48=0.97,
                m15_ret_8=-0.01,
                m5_ret_6=-0.004,
            )
        )

        self.assertEqual(decision.action, StrategyAction.SKIP)
        self.assertIsNone(decision.signal)
        self.assertIn("chase_overheat", [code.value for code in decision.filter_codes])
        self.assertIn("skip", decision.explanation)

    def test_default_v1_grid_tunes_long_short_entry_and_exit_parameters(self):
        variants = default_langlang_v1_grid()

        self.assertGreaterEqual(len(variants), 90)
        self.assertTrue(any("long" in variant.variant_id for variant in variants))
        self.assertTrue(any("short" in variant.variant_id for variant in variants))
        self.assertGreater(len({variant.intraday_confirm_ret_min for variant in variants}), 1)
        self.assertGreater(len({variant.structure_stop_pct for variant in variants}), 1)
        self.assertGreater(len({variant.runner_take_profit_r for variant in variants}), 1)


if __name__ == "__main__":
    unittest.main()

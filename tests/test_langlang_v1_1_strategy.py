import unittest

from langlang_trader.features import FeatureSnapshot
from langlang_trader.models import EntrySetup, FailureFilter, MarketRegime, Side, StrategyAction
from langlang_trader.strategy import (
    LangLangV1_1Variant,
    RulesLangLangV1_1Strategy,
    default_langlang_v1_1_grid,
    strategy_from_version,
)


def snapshot(**overrides):
    features = {
        "ret_20d": 0.30,
        "ret_60d": 0.82,
        "pos_20d": 0.70,
        "pullback_from_20d_high": -0.055,
        "ma_5": 124.0,
        "ma_20": 110.0,
        "latest_close": 122.0,
        "high_20d": 130.0,
        "low_20d": 92.0,
        "high_60d": 132.0,
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
    }
    features.update(overrides)
    return FeatureSnapshot(
        symbol="BTC-USDT-SWAP",
        bar="multi",
        last_ts=1_710_000_000_000,
        created_at="2024-03-12T00:00:00+00:00",
        features=features,
    )


class LangLangV1_1StrategyTest(unittest.TestCase):
    def test_small_divergence_first_entry_requires_space_and_historical_support(self):
        strategy = RulesLangLangV1_1Strategy(LangLangV1_1Variant(variant_id="v1.1-test"))

        decision = strategy.decide(snapshot(m15_ret_8=-0.003, m5_ret_6=0.007))

        self.assertEqual(decision.action, StrategyAction.ENTER)
        self.assertIsNotNone(decision.signal)
        self.assertEqual(decision.signal.strategy_version, "rules_langlang_v1_1")
        self.assertEqual(decision.signal.side, Side.LONG)
        self.assertEqual(decision.signal.regime, MarketRegime.FIRST_DIVERGENCE)
        self.assertEqual(decision.signal.setup, EntrySetup.SMALL_DIVERGENCE_ENTRY)
        self.assertIn("small_divergence_absorbed", decision.signal.reason_codes)
        self.assertEqual(decision.signal.matched_trade_examples[0]["trade_id"], "big-win-1")
        self.assertGreaterEqual(decision.signal.historical_match_score, 0.70)
        self.assertEqual(decision.signal.features["risk_unit"], "W")

    def test_insufficient_upside_space_is_a_hard_skip(self):
        strategy = RulesLangLangV1_1Strategy(LangLangV1_1Variant(variant_id="v1.1-test"))

        decision = strategy.decide(snapshot(upside_space_pct=0.035))

        self.assertEqual(decision.action, StrategyAction.SKIP)
        self.assertIn(FailureFilter.INSUFFICIENT_UPSIDE_SPACE, decision.filter_codes)
        self.assertIn("upside_space", decision.explanation)

    def test_first_10x_after_high_extension_is_skipped(self):
        strategy = RulesLangLangV1_1Strategy(LangLangV1_1Variant(variant_id="v1.1-test"))

        decision = strategy.decide(snapshot(first_10x_entry_done=True, pos_20d=0.96, pullback_from_20d_high=-0.004))

        self.assertEqual(decision.action, StrategyAction.SKIP)
        self.assertIn(FailureFilter.FIRST_10X_TOO_HIGH, decision.filter_codes)

    def test_post_large_divergence_requires_bottom_lift_and_internal_rebound(self):
        strategy = RulesLangLangV1_1Strategy(LangLangV1_1Variant(variant_id="v1.1-test"))

        blocked = strategy.decide(
            snapshot(large_divergence_recent=True, bottom_lift_confirmed=False, m15_ret_8=0.018)
        )
        allowed = strategy.decide(
            snapshot(large_divergence_recent=True, bottom_lift_confirmed=True, m15_ret_8=0.018)
        )

        self.assertEqual(blocked.action, StrategyAction.SKIP)
        self.assertIn(FailureFilter.NO_BOTTOM_LIFT, blocked.filter_codes)
        self.assertEqual(allowed.action, StrategyAction.ENTER)
        self.assertEqual(allowed.signal.regime, MarketRegime.POST_LARGE_DIVERGENCE)
        self.assertEqual(allowed.signal.setup, EntrySetup.POST_DIVERGENCE_REBOUND)

    def test_non_exploratory_signal_without_historical_support_is_skipped(self):
        strategy = RulesLangLangV1_1Strategy(
            LangLangV1_1Variant(variant_id="v1.1-test", exploratory=False, min_historical_match_score=0.40)
        )

        decision = strategy.decide(snapshot(historical_match_score=0.10, matched_trade_examples=[]))

        self.assertEqual(decision.action, StrategyAction.SKIP)
        self.assertIn(FailureFilter.NO_HISTORICAL_SUPPORT, decision.filter_codes)

    def test_v1_1_variant_side_gate_prevents_long_short_parameter_drift(self):
        long_only = RulesLangLangV1_1Strategy(
            LangLangV1_1Variant(variant_id="llv1_1_long_test", allowed_side="long", exploratory=True)
        )
        short_only = RulesLangLangV1_1Strategy(
            LangLangV1_1Variant(variant_id="llv1_1_short_test", allowed_side="short", exploratory=True)
        )

        short_market = snapshot(
            ret_20d=-0.28,
            ret_60d=-0.44,
            pos_20d=0.12,
            pullback_from_20d_high=-0.28,
            ma_5=74.0,
            ma_20=92.0,
            latest_close=70.0,
            high_20d=110.0,
            low_20d=68.0,
            high_60d=140.0,
            h1_ret_24=-0.07,
            m15_ret_8=-0.02,
            m5_ret_6=-0.01,
        )

        long_blocked = long_only.decide(short_market)
        short_blocked = short_only.decide(snapshot())

        self.assertEqual(long_blocked.action, StrategyAction.SKIP)
        self.assertIn(FailureFilter.VARIANT_SIDE_NOT_ALLOWED, long_blocked.filter_codes)
        self.assertEqual(short_blocked.action, StrategyAction.SKIP)
        self.assertIn(FailureFilter.VARIANT_SIDE_NOT_ALLOWED, short_blocked.filter_codes)

    def test_default_grid_and_factory_support_v1_1(self):
        variants = default_langlang_v1_1_grid()

        self.assertGreaterEqual(len(variants), 30)
        self.assertTrue(any(variant.exploratory for variant in variants))
        self.assertTrue(any(not variant.exploratory for variant in variants))
        self.assertTrue(all(variant.allowed_side == "long" for variant in variants if variant.variant_id.startswith("llv1_1_long_")))
        self.assertTrue(all(variant.allowed_side == "short" for variant in variants if variant.variant_id.startswith("llv1_1_short_")))
        self.assertIsInstance(strategy_from_version("rules_langlang_v1_1", variants[0]), RulesLangLangV1_1Strategy)


if __name__ == "__main__":
    unittest.main()

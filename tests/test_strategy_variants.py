import unittest

from langlang_trader.features import FeatureSnapshot
from langlang_trader.strategy import RulesV01Strategy, StrategyVariant


def snapshot() -> FeatureSnapshot:
    return FeatureSnapshot(
        symbol="BTC-USDT-SWAP",
        bar="1D",
        last_ts=1_710_000_000_000,
        created_at="2024-03-12T00:00:00+00:00",
        features={
            "ret_20d": 0.15,
            "ret_60d": 0.36,
            "pos_20d": 0.50,
            "pullback_from_20d_high": -0.10,
            "ma_5": 120.0,
            "ma_20": 110.0,
            "latest_close": 121.0,
            "high_20d": 130.0,
            "low_20d": 90.0,
            "high_60d": 122.0,
        },
    )


class StrategyVariantTest(unittest.TestCase):
    def test_same_features_can_pass_loose_variant_and_fail_strict_variant(self):
        loose = RulesV01Strategy(
            StrategyVariant(
                variant_id="loose",
                ret_20d_min=0.12,
                ret_60d_min=0.32,
                pos_20d_min=0.45,
                max_pullback_pct=0.18,
                breakout_tolerance=0.005,
            )
        )
        strict = RulesV01Strategy(
            StrategyVariant(
                variant_id="strict",
                ret_20d_min=0.18,
                ret_60d_min=0.40,
                pos_20d_min=0.60,
                max_pullback_pct=0.06,
                breakout_tolerance=0.001,
            )
        )

        loose_signal = loose.generate_from_features(snapshot())
        strict_signal = strict.generate_from_features(snapshot())

        self.assertIsNotNone(loose_signal)
        self.assertIsNone(strict_signal)
        self.assertEqual(loose_signal.features["variant_id"], "loose")
        self.assertIn("variant:loose", loose_signal.reason_codes)


if __name__ == "__main__":
    unittest.main()

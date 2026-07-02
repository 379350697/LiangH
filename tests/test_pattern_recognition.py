import unittest

from langlang_trader.features import DailyFeatureBuilder, MultiTimeframeFeatureBuilder
from langlang_trader.models import Candle
from langlang_trader.pattern_recognition import StrongPatternDetector


def candle(idx, close, *, high=None, low=None, open_=None, volume=1000.0, bar="1D"):
    open_value = close if open_ is None else open_
    return Candle(
        symbol="PATTERN-USDT-SWAP",
        bar=bar,
        ts=1_700_000_000_000 + idx * 86_400_000,
        open=open_value,
        high=high if high is not None else max(open_value, close) * 1.01,
        low=low if low is not None else min(open_value, close) * 0.99,
        close=close,
        volume=volume,
    )


def series_from_closes(closes, *, volumes=None, bar="1D"):
    volumes = volumes or [1000.0] * len(closes)
    return [candle(idx, close, volume=volumes[idx], bar=bar) for idx, close in enumerate(closes)]


class StrongPatternDetectorTest(unittest.TestCase):
    def test_detects_leader_platform_start(self):
        closes = [
            *[100 + idx * 2.2 for idx in range(30)],
            166, 164, 167, 165, 168, 166, 169, 167, 170, 168, 171, 169, 172, 170, 173,
            176, 181, 188,
        ]

        features = StrongPatternDetector().detect(series_from_closes(closes))

        self.assertGreaterEqual(features["leader_platform_start_score"], 0.70)
        self.assertEqual(features["strong_pattern_tag"], "leader_platform_start")
        self.assertIn("leader_platform_start", features["pattern_reason_codes"])

    def test_detects_golden_pit_reclaim(self):
        closes = [
            *[100 + idx * 1.6 for idx in range(35)],
            156, 155, 154, 156, 155, 154, 95, 146, 158, 161, 164, 166,
        ]
        volumes = [1000.0] * (len(closes) - 4) + [2600.0, 2400.0, 1800.0, 1700.0]

        features = StrongPatternDetector().detect(series_from_closes(closes, volumes=volumes))

        self.assertGreaterEqual(features["golden_pit_reclaim_score"], 0.70)
        self.assertEqual(features["strong_pattern_tag"], "golden_pit_reclaim")
        self.assertIn("golden_pit_fast_reclaim", features["pattern_reason_codes"])

    def test_golden_pit_handles_zero_pre_low_without_crashing(self):
        closes = [
            100, 101, 102, 103, 104, 105, 0, 50, 55, 60, 65, 70,
            75, 80, 85, 90, 75, 80, -10, 5, 12, 20, 25,
        ]

        features = StrongPatternDetector().detect(series_from_closes(closes))

        self.assertIn("golden_pit_reclaim_score", features)

    def test_pattern_reasons_only_include_winning_positive_and_risk_patterns(self):
        closes = [
            100, 113, 106, 124, 116, 139, 130, 151, 143, 160, 154, 166, 162, 168,
            166, 167, 166, 168, 167, 169, 168, 170, 169, 171, 170, 172, 171, 173, 177, 184,
        ]

        features = StrongPatternDetector().detect(series_from_closes(closes))

        self.assertEqual(features["strong_pattern_tag"], "leader_platform_start")
        self.assertEqual(features["risk_pattern_tag"], "five_wave_late_risk")
        self.assertIn("leader_platform_start", features["strong_pattern_reason_codes"])
        self.assertIn("five_wave_late_risk", features["risk_pattern_reason_codes"])
        self.assertIn("leader_platform_start", features["pattern_reason_codes"])
        self.assertIn("five_wave_late_risk", features["pattern_reason_codes"])
        self.assertNotIn("golden_pit_fast_reclaim", features["pattern_reason_codes"])
        self.assertNotIn("small_divergence_absorbed", features["pattern_reason_codes"])

    def test_pattern_reasons_do_not_include_lower_scoring_positive_patterns(self):
        closes = [
            *[100 + idx * 1.6 for idx in range(35)],
            156, 155, 154, 156, 155, 154, 95, 146, 158, 161, 164, 166,
        ]
        volumes = [1000.0] * (len(closes) - 4) + [2600.0, 2400.0, 1800.0, 1700.0]

        features = StrongPatternDetector().detect(series_from_closes(closes, volumes=volumes))

        self.assertEqual(features["strong_pattern_tag"], "golden_pit_reclaim")
        self.assertEqual(features["risk_pattern_tag"], "")
        self.assertIn("golden_pit_fast_reclaim", features["strong_pattern_reason_codes"])
        self.assertEqual(features["pattern_reason_codes"], features["strong_pattern_reason_codes"])
        self.assertNotIn("leader_platform_prior_strength", features["pattern_reason_codes"])
        self.assertNotIn("second_wave_reclaim", features["pattern_reason_codes"])

    def test_stale_old_waves_do_not_pollute_current_pattern(self):
        stale_five_wave = [100, 113, 106, 124, 116, 139, 130, 151, 143, 160, 154, 166, 162, 168]
        unrelated_middle = [90 + idx * 0.8 for idx in range(80)]
        current_leader_platform = [
            *[100 + idx * 2.2 for idx in range(30)],
            166, 164, 167, 165, 168, 166, 169, 167, 170, 168, 171, 169, 172, 170, 173,
            176, 181, 188,
        ]

        features = StrongPatternDetector().detect(
            series_from_closes([*stale_five_wave, *unrelated_middle, *current_leader_platform])
        )

        self.assertEqual(features["risk_pattern_tag"], "")
        self.assertNotIn("five_wave_late_risk", features["pattern_reason_codes"])

    def test_detects_small_divergence_absorb(self):
        closes = [
            *[100 + idx * 1.9 for idx in range(35)],
            166, 171, 176, 171, 168, 171, 174, 178,
        ]

        features = StrongPatternDetector().detect(series_from_closes(closes))

        self.assertGreaterEqual(features["small_divergence_absorb_score"], 0.65)
        self.assertEqual(features["strong_pattern_tag"], "small_divergence_absorb")
        self.assertGreaterEqual(features["small_divergence_count"], 1)

    def test_ordinary_overlapping_volatility_does_not_count_as_third_small_divergence(self):
        closes = []
        price = 100.0
        for _ in range(18):
            closes.extend([price, price * 0.962, price * 0.988, price * 0.965, price * 0.992])
            price *= 1.002

        features = StrongPatternDetector().detect(series_from_closes(closes))

        self.assertLessEqual(features["small_divergence_count"], 2)

    def test_counts_non_overlapping_high_position_small_divergences(self):
        closes = [
            *[100 + idx * 2.0 for idx in range(30)],
            160, 169, 160, 173,
            176, 166, 181,
            184, 174, 188,
            190,
        ]

        features = StrongPatternDetector().detect(series_from_closes(closes))

        self.assertGreaterEqual(features["small_divergence_count"], 3)
        self.assertGreaterEqual(features["five_wave_late_risk_score"], 0.70)

    def test_detects_second_wave_start_after_large_divergence(self):
        closes = [
            *[100 + idx * 2.0 for idx in range(35)],
            170, 158, 145, 132, 139, 136, 144, 141, 151, 158, 166,
        ]

        features = StrongPatternDetector().detect(series_from_closes(closes))

        self.assertGreaterEqual(features["second_wave_start_score"], 0.65)
        self.assertEqual(features["strong_pattern_tag"], "second_wave_start")
        self.assertIn("large_divergence_bottom_lift", features["pattern_reason_codes"])

    def test_spoon_bottom_requires_right_side_confirmation(self):
        left_side = [160, 150, 142, 136, 132, 129, 127, 126, 125, 125, 126, 126, 127, 127, 128]
        confirmed = [
            *left_side,
            129, 130, 132, 135, 138, 142, 147, 153, 160, 168,
        ]

        left_features = StrongPatternDetector().detect(series_from_closes(left_side))
        confirmed_features = StrongPatternDetector().detect(series_from_closes(confirmed))

        self.assertLess(left_features["spoon_bottom_confirmed_score"], 0.65)
        self.assertGreaterEqual(confirmed_features["spoon_bottom_confirmed_score"], 0.65)
        self.assertEqual(confirmed_features["strong_pattern_tag"], "spoon_bottom_confirmed")

    def test_detects_five_wave_late_risk(self):
        closes = [
            100, 113, 106, 124, 116, 139, 130, 151, 143, 160, 154, 166, 162, 168,
        ]

        features = StrongPatternDetector().detect(series_from_closes(closes))

        self.assertGreaterEqual(features["five_wave_late_risk_score"], 0.70)
        self.assertGreaterEqual(features["risk_pattern_score"], 0.70)
        self.assertIn("five_wave_late_risk", features["pattern_reason_codes"])
        self.assertGreaterEqual(features["wave_push_count"], 5)

    def test_detects_false_breakout_risk(self):
        closes = [
            100, 102, 101, 103, 102, 104, 103, 105, 104, 106, 105, 107,
            121, 104, 102, 101,
        ]
        volumes = [1000.0] * 12 + [900.0, 1300.0, 1200.0, 1100.0]

        features = StrongPatternDetector().detect(series_from_closes(closes, volumes=volumes))

        self.assertGreaterEqual(features["false_breakout_risk_score"], 0.65)
        self.assertGreaterEqual(features["risk_pattern_score"], 0.65)
        self.assertIn("false_breakout_fell_back_into_box", features["pattern_reason_codes"])

    def test_daily_feature_builder_includes_strong_pattern_fields(self):
        closes = [
            *[100 + idx * 1.5 for idx in range(45)],
            168, 166, 169, 167, 170, 168, 171, 169, 172, 170, 173, 171, 174, 172, 175,
            180, 186, 194,
        ]

        snapshot = DailyFeatureBuilder().build("PATTERN-USDT-SWAP", series_from_closes(closes))

        self.assertIsNotNone(snapshot)
        self.assertGreaterEqual(snapshot.features["leader_platform_start_score"], 0.70)
        self.assertEqual(snapshot.features["strong_pattern_tag"], "leader_platform_start")
        self.assertIn("pattern_reason_codes", snapshot.features)

    def test_multi_timeframe_feature_builder_adds_consensus_fields(self):
        daily_closes = [
            *[100 + idx * 1.0 for idx in range(61)],
            160, 159, 158, 160, 159, 158, 101, 151, 162, 166, 169, 172,
        ]
        h1_closes = [
            *[100 + idx * 0.8 for idx in range(24)],
            119, 118, 117, 82, 114, 121, 124, 127,
        ]

        snapshot = MultiTimeframeFeatureBuilder().build(
            "PATTERN-USDT-SWAP",
            {
                "1D": series_from_closes(daily_closes),
                "1H": series_from_closes(h1_closes, bar="1H"),
            },
        )

        self.assertIsNotNone(snapshot)
        self.assertGreaterEqual(snapshot.features["golden_pit_reclaim_score"], 0.70)
        self.assertIn("h1_golden_pit_reclaim_score", snapshot.features)
        self.assertIn("golden_pit_intraday_reclaim_confirmed", snapshot.features["pattern_reason_codes"])


if __name__ == "__main__":
    unittest.main()

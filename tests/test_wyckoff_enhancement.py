import unittest

from langlang_trader.features import DailyFeatureBuilder, MultiTimeframeFeatureBuilder
from langlang_trader.models import Candle
from langlang_trader.wyckoff_enhancement import WyckoffEventDetector


def candle(idx, close, *, high=None, low=None, open_=None, volume=1000.0, bar="1D"):
    open_value = close if open_ is None else open_
    return Candle(
        symbol="WYCK-USDT-SWAP",
        bar=bar,
        ts=1_700_000_000_000 + idx * 86_400_000,
        open=open_value,
        high=high if high is not None else max(open_value, close) * 1.01,
        low=low if low is not None else min(open_value, close) * 0.99,
        close=close,
        volume=volume,
    )


def series_from_ohlcv(rows, *, bar="1D"):
    return [
        candle(
            idx,
            row["close"],
            open_=row.get("open"),
            high=row.get("high"),
            low=row.get("low"),
            volume=row.get("volume", 1000.0),
            bar=bar,
        )
        for idx, row in enumerate(rows)
    ]


def series_from_closes(closes, *, volumes=None, bar="1D"):
    volumes = volumes or [1000.0] * len(closes)
    return [candle(idx, close, volume=volumes[idx], bar=bar) for idx, close in enumerate(closes)]


class WyckoffEventDetectorTest(unittest.TestCase):
    def test_detects_accumulation_spring_reclaim(self):
        range_rows = [
            {"open": 100, "high": 106, "low": 96, "close": 101, "volume": 900},
            {"open": 101, "high": 107, "low": 97, "close": 103, "volume": 860},
            {"open": 103, "high": 108, "low": 98, "close": 102, "volume": 830},
            {"open": 102, "high": 106, "low": 97, "close": 100, "volume": 790},
        ] * 7
        spring_rows = [
            {"open": 100, "high": 102, "low": 88, "close": 94, "volume": 2600},
            {"open": 94, "high": 106, "low": 93, "close": 104, "volume": 2300},
            {"open": 104, "high": 109, "low": 101, "close": 107, "volume": 1700},
        ]

        features = WyckoffEventDetector().detect(series_from_ohlcv([*range_rows, *spring_rows]))

        self.assertEqual(features["wyckoff_phase_tag"], "accumulation")
        self.assertEqual(features["wyckoff_long_setup_tag"], "spring_reclaim")
        self.assertGreaterEqual(features["wyckoff_long_score"], 0.68)
        self.assertIn("wyckoff_spring_reclaim", features["wyckoff_long_reason_codes"])

    def test_detects_sos_breakout_and_lps_retest(self):
        base = [{"open": 102, "high": 108, "low": 98, "close": 103 + (idx % 3), "volume": 900} for idx in range(30)]
        sos_lps = [
            {"open": 105, "high": 119, "low": 104, "close": 117, "volume": 2500},
            {"open": 117, "high": 120, "low": 113, "close": 116, "volume": 1600},
            {"open": 116, "high": 117, "low": 110, "close": 112, "volume": 1050},
            {"open": 112, "high": 121, "low": 111, "close": 120, "volume": 1800},
        ]

        features = WyckoffEventDetector().detect(series_from_ohlcv([*base, *sos_lps]))

        self.assertIn(features["wyckoff_long_setup_tag"], {"sos_breakout", "lps_retest", "reaccumulation_breakout"})
        self.assertGreaterEqual(features["wyckoff_long_score"], 0.68)
        self.assertTrue(
            {"wyckoff_sos_breakout", "wyckoff_lps_retest", "wyckoff_reaccumulation_breakout"}
            & set(features["wyckoff_long_reason_codes"])
        )

    def test_detects_distribution_utad_risk(self):
        range_rows = [{"open": 104, "high": 110, "low": 98, "close": 105 + (idx % 2), "volume": 1000} for idx in range(30)]
        utad_rows = [
            {"open": 106, "high": 126, "low": 105, "close": 121, "volume": 2400},
            {"open": 121, "high": 123, "low": 104, "close": 107, "volume": 2600},
            {"open": 107, "high": 109, "low": 101, "close": 103, "volume": 2100},
        ]

        features = WyckoffEventDetector().detect(series_from_ohlcv([*range_rows, *utad_rows]))

        self.assertEqual(features["wyckoff_phase_tag"], "distribution")
        self.assertEqual(features["wyckoff_short_setup_tag"], "utad_risk")
        self.assertGreaterEqual(features["wyckoff_risk_score"], 0.70)
        self.assertGreaterEqual(features["wyckoff_short_score"], 0.70)

    def test_detects_sow_breakdown_and_lpsy_retest(self):
        range_rows = [{"open": 104, "high": 110, "low": 99, "close": 104 + (idx % 3), "volume": 950} for idx in range(30)]
        sow_rows = [
            {"open": 104, "high": 105, "low": 88, "close": 91, "volume": 2600},
            {"open": 91, "high": 101, "low": 90, "close": 98, "volume": 1300},
            {"open": 98, "high": 100, "low": 90, "close": 92, "volume": 1900},
        ]

        features = WyckoffEventDetector().detect(series_from_ohlcv([*range_rows, *sow_rows]))

        self.assertEqual(features["wyckoff_phase_tag"], "markdown")
        self.assertIn(features["wyckoff_short_setup_tag"], {"sow_breakdown", "lpsy_retest"})
        self.assertGreaterEqual(features["wyckoff_short_score"], 0.70)
        self.assertTrue({"wyckoff_sow_breakdown", "wyckoff_lpsy_retest"} & set(features["wyckoff_short_reason_codes"]))

    def test_detects_no_demand_or_effort_result_divergence(self):
        rows = [{"open": 100 + idx * 0.2, "high": 105, "low": 98, "close": 101 + idx * 0.2, "volume": 1000} for idx in range(25)]
        rows.extend(
            [
                {"open": 106, "high": 116, "low": 105, "close": 108, "volume": 2800},
                {"open": 108, "high": 118, "low": 107, "close": 109, "volume": 3000},
                {"open": 109, "high": 119, "low": 108, "close": 110, "volume": 3200},
            ]
        )

        features = WyckoffEventDetector().detect(series_from_ohlcv(rows))

        self.assertGreaterEqual(features["wyckoff_risk_score"], 0.65)
        self.assertTrue(
            {"wyckoff_effort_result_divergence", "wyckoff_no_demand_breakout"}
            & set(features["wyckoff_risk_reason_codes"])
        )

    def test_feature_builders_include_wyckoff_fields_and_consensus(self):
        daily_closes = [100 + (idx % 4) for idx in range(65)]
        daily = series_from_closes(daily_closes)
        h1 = series_from_ohlcv(
            [
                *[{"open": 100, "high": 106, "low": 96, "close": 101 + (idx % 3), "volume": 900} for idx in range(26)],
                {"open": 101, "high": 102, "low": 88, "close": 94, "volume": 2600},
                {"open": 94, "high": 107, "low": 93, "close": 105, "volume": 2300},
            ],
            bar="1H",
        )

        daily_snapshot = DailyFeatureBuilder().build("WYCK-USDT-SWAP", daily)
        multi_snapshot = MultiTimeframeFeatureBuilder().build("WYCK-USDT-SWAP", {"1D": daily, "1H": h1})

        self.assertIsNotNone(daily_snapshot)
        self.assertIn("wyckoff_long_score", daily_snapshot.features)
        self.assertIsNotNone(multi_snapshot)
        self.assertIn("h1_wyckoff_long_score", multi_snapshot.features)
        self.assertIn("wyckoff_reason_codes", multi_snapshot.features)


if __name__ == "__main__":
    unittest.main()

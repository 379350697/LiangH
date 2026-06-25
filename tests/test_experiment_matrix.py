import csv
import json
import os
import tempfile
import unittest

from langlang_trader.experiment_matrix import (
    FEATURE_PROFILE_BASELINE_V1_3,
    FEATURE_PROFILE_STRONG_PATTERN_V1_3,
    FEATURE_PROFILE_WYCKOFF_ENHANCED_V1_3,
    apply_feature_profile,
    run_v1_3_experiment_matrix,
)
from langlang_trader.features import FeatureSnapshot
from langlang_trader.strategy import LangLangV1_3Variant


def write_trades(path):
    rows = [
        {
            "trade_id": "sol-long",
            "entry_time": "2024-02-15 00:00:00",
            "exit_time": "2024-02-16 00:00:00",
            "symbol": "SOL-USDT-SWAP",
            "side": "long",
            "pnl_usdt": "500",
            "return_rate": "0.50",
        }
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def write_cache(cache_dir, symbol):
    one_d = os.path.join(cache_dir, "1D")
    os.makedirs(one_d, exist_ok=True)
    path = os.path.join(one_d, f"{symbol}_test.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ts", "open", "high", "low", "close", "vol", "vol_ccy", "vol_quote", "confirm"])
        start = 1_701_580_800_000
        price = 100.0
        for idx in range(130):
            price *= 1.012
            writer.writerow([start + idx * 86_400_000, price * 0.99, price * 1.02, price * 0.98, price, 1000 + idx, 100, 100000, 1])


class ExperimentMatrixTest(unittest.TestCase):
    def test_feature_profile_masks_pattern_and_wyckoff_layers(self):
        snapshot = FeatureSnapshot(
            symbol="SOL-USDT-SWAP",
            bar="multi",
            last_ts=1,
            created_at="2024-01-01T00:00:00Z",
            features={
                "ret_20d": 0.2,
                "strong_pattern_tag": "golden_pit_reclaim",
                "strong_pattern_score": 0.8,
                "risk_pattern_tag": "five_wave_late_risk",
                "risk_pattern_score": 0.7,
                "pattern_reason_codes": ["golden_pit_fast_reclaim"],
                "wyckoff_phase_tag": "accumulation",
                "wyckoff_long_setup_tag": "spring_reclaim",
                "wyckoff_long_score": 0.72,
                "wyckoff_reason_codes": ["wyckoff_spring_reclaim"],
            },
        )

        baseline = apply_feature_profile(snapshot, FEATURE_PROFILE_BASELINE_V1_3)
        self.assertEqual(baseline.features["strong_pattern_tag"], "")
        self.assertEqual(baseline.features["strong_pattern_score"], 0.0)
        self.assertEqual(baseline.features["risk_pattern_tag"], "")
        self.assertEqual(baseline.features["pattern_reason_codes"], [])
        self.assertEqual(baseline.features["wyckoff_phase_tag"], "none")
        self.assertEqual(baseline.features["wyckoff_long_setup_tag"], "")
        self.assertEqual(baseline.features["wyckoff_long_score"], 0.0)

        strong_only = apply_feature_profile(snapshot, FEATURE_PROFILE_STRONG_PATTERN_V1_3)
        self.assertEqual(strong_only.features["strong_pattern_tag"], "golden_pit_reclaim")
        self.assertEqual(strong_only.features["wyckoff_phase_tag"], "none")
        self.assertEqual(strong_only.features["wyckoff_long_score"], 0.0)

        enhanced = apply_feature_profile(snapshot, FEATURE_PROFILE_WYCKOFF_ENHANCED_V1_3)
        self.assertEqual(enhanced.features["strong_pattern_tag"], "golden_pit_reclaim")
        self.assertEqual(enhanced.features["wyckoff_long_setup_tag"], "spring_reclaim")

    def test_v1_3_matrix_writes_three_profile_attribution_and_zero_signal_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            trades = os.path.join(tmp, "standard_trades.csv")
            cache = os.path.join(tmp, "kline_cache")
            out = os.path.join(tmp, "matrix")
            write_trades(trades)
            write_cache(cache, "SOL-USDT-SWAP")

            result = run_v1_3_experiment_matrix(
                trades_csv=trades,
                kline_cache_dir=cache,
                out_dir=out,
                variants=[
                    LangLangV1_3Variant(
                        variant_id="llv1_3_matrix_smoke",
                        allowed_side="long",
                        exploratory=True,
                        ret_20d_min=0.10,
                        ret_60d_min=0.20,
                        min_upside_space_pct=0.0,
                    )
                ],
                top_n=1,
                min_validation_signals=0,
                max_validation_signals=300,
            )

            self.assertEqual(
                set(result.profile_dirs.keys()),
                {
                    FEATURE_PROFILE_BASELINE_V1_3,
                    FEATURE_PROFILE_STRONG_PATTERN_V1_3,
                    FEATURE_PROFILE_WYCKOFF_ENHANCED_V1_3,
                },
            )
            self.assertTrue(os.path.exists(os.path.join(out, "experiment_matrix_summary.csv")))
            self.assertTrue(os.path.exists(os.path.join(out, "experiment_matrix_attribution.json")))
            self.assertTrue(os.path.exists(os.path.join(out, "experiment_matrix_report.md")))

            with open(os.path.join(out, "experiment_matrix_summary.csv"), encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 3)
            self.assertEqual({row["experiment_label"] for row in rows}, set(result.profile_dirs.keys()))
            self.assertIn("zero_signal_top_filters", rows[0])
            self.assertIn("strong_pattern_score_bins", rows[0])
            self.assertIn("wyckoff_long_score_bins", rows[0])

            with open(os.path.join(out, "experiment_matrix_attribution.json"), encoding="utf-8") as f:
                attribution = json.load(f)
            self.assertEqual(set(attribution["profiles"].keys()), set(result.profile_dirs.keys()))
            self.assertIn("leaderboard_path", attribution["profiles"][FEATURE_PROFILE_WYCKOFF_ENHANCED_V1_3])

            with open(os.path.join(out, "experiment_matrix_report.md"), encoding="utf-8") as f:
                report = f.read()
            self.assertIn("Zero Signal Diagnostics", report)
            self.assertIn("best_signal_variant", report)
            self.assertIn("strong_pattern_score", report)
            self.assertIn("wyckoff_long_score", report)


if __name__ == "__main__":
    unittest.main()

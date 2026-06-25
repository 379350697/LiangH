import unittest

from langlang_trader.config import SymbolSelectionConfig
from langlang_trader.features import FeatureSnapshot
from langlang_trader.symbol_selection import SelectionEngine


def snapshot(symbol, **features):
    base = {
        "ret_3d": 0.0,
        "ret_7d": 0.0,
        "ret_20d": 0.0,
        "ret_60d": 0.0,
        "pos_20d": 0.5,
        "pullback_from_20d_high": -0.03,
        "vol_ratio_20d": 1.0,
        "latest_close": 100.0,
        "high_20d": 110.0,
        "high_60d": 120.0,
        "ma_5": 100.0,
        "ma_20": 99.0,
    }
    base.update(features)
    return FeatureSnapshot(
        symbol=symbol,
        bar="multi",
        last_ts=1_710_000_000_000,
        created_at="2024-03-12T00:00:00+00:00",
        features=base,
    )


class SelectionEngineV13Test(unittest.TestCase):
    def test_leader_altcoin_requires_btc_resonance_resilience_space_and_volume(self):
        engine = SelectionEngine(
            SymbolSelectionConfig(enabled=True, style="dual_board", long_top_n=3, short_top_n=2)
        )
        snapshots = {
            "BTC-USDT-SWAP": snapshot("BTC-USDT-SWAP", ret_20d=0.16, ret_60d=0.34, ret_7d=0.04, pos_20d=0.68),
            "ETH-USDT-SWAP": snapshot("ETH-USDT-SWAP", ret_20d=0.10, ret_60d=0.28, ret_7d=0.03, pos_20d=0.62),
            "LEADER-USDT-SWAP": snapshot(
                "LEADER-USDT-SWAP",
                ret_3d=0.08,
                ret_7d=0.22,
                ret_20d=0.62,
                ret_60d=1.20,
                pos_20d=0.82,
                pullback_from_20d_high=-0.045,
                vol_ratio_20d=2.1,
                upside_space_pct=0.42,
                btc_first_wave_follow=True,
                btc_contraction_resilient=True,
            ),
            "CATCH-USDT-SWAP": snapshot(
                "CATCH-USDT-SWAP",
                ret_3d=0.18,
                ret_7d=0.34,
                ret_20d=0.40,
                ret_60d=0.08,
                pos_20d=0.72,
                pullback_from_20d_high=-0.01,
                vol_ratio_20d=1.5,
                btc_divergence_alt_rotation=True,
            ),
            "FALL-USDT-SWAP": snapshot(
                "FALL-USDT-SWAP",
                ret_3d=-0.10,
                ret_7d=-0.24,
                ret_20d=-0.42,
                ret_60d=-0.70,
                pos_20d=0.10,
                pullback_from_20d_high=-0.40,
                vol_ratio_20d=2.0,
                latest_close=70.0,
                ma_5=76.0,
                ma_20=90.0,
                failed_rebound_below_platform=True,
            ),
        }

        boards = engine.rank_all_market(snapshots)
        long_by_symbol = {row.symbol: row for row in boards["long_main_wave"]}
        short_by_symbol = {row.symbol: row for row in boards["short_waterfall"]}

        leader = long_by_symbol["LEADER-USDT-SWAP"]
        catch = long_by_symbol["CATCH-USDT-SWAP"]
        fall = short_by_symbol["FALL-USDT-SWAP"]

        self.assertTrue(leader.selected)
        self.assertIn("leader_altcoin", leader.reason_codes)
        self.assertIn("btc_first_wave_follow", leader.reason_codes)
        self.assertIn("btc_contraction_resilient", leader.reason_codes)
        self.assertIn("upside_space_large", leader.reason_codes)
        self.assertEqual(leader.features["symbol_selection_tag"], "leader_altcoin")

        self.assertIn("catch_up_short_hold", catch.reason_codes)
        self.assertEqual(catch.features["symbol_selection_tag"], "catch_up_short_hold")
        self.assertIn("not_leader_catch_up", catch.filter_codes)

        self.assertTrue(fall.selected)
        self.assertIn("failed_rebound_below_platform", fall.reason_codes)
        self.assertEqual(fall.features["symbol_selection_tag"], "short_waterfall")

    def test_auxiliary_market_features_affect_selection_and_reasons(self):
        engine = SelectionEngine(
            SymbolSelectionConfig(enabled=True, style="dual_board", long_top_n=2, short_top_n=2)
        )
        snapshots = {
            "BTC-USDT-SWAP": snapshot("BTC-USDT-SWAP", ret_20d=0.10, ret_60d=0.20, ret_7d=0.02),
            "ETH-USDT-SWAP": snapshot("ETH-USDT-SWAP", ret_20d=0.08, ret_60d=0.18, ret_7d=0.01),
            "HEALTHY-USDT-SWAP": snapshot(
                "HEALTHY-USDT-SWAP",
                ret_3d=0.06,
                ret_7d=0.16,
                ret_20d=0.42,
                ret_60d=0.80,
                pos_20d=0.78,
                pullback_from_20d_high=-0.04,
                vol_ratio_20d=1.4,
                turnover_rank=25,
                turnover_rank_top_n=200,
                turnover_usdt=80_000_000,
                oi_change_3d=0.18,
                funding_rate_last=0.0003,
            ),
            "CROWDED-USDT-SWAP": snapshot(
                "CROWDED-USDT-SWAP",
                ret_3d=0.07,
                ret_7d=0.17,
                ret_20d=0.44,
                ret_60d=0.82,
                pos_20d=0.80,
                pullback_from_20d_high=-0.04,
                vol_ratio_20d=1.4,
                turnover_rank=15,
                turnover_rank_top_n=200,
                turnover_usdt=120_000_000,
                oi_change_3d=0.20,
                funding_rate_last=0.025,
            ),
            "THIN-USDT-SWAP": snapshot(
                "THIN-USDT-SWAP",
                ret_3d=0.05,
                ret_7d=0.14,
                ret_20d=0.40,
                ret_60d=0.78,
                pos_20d=0.76,
                pullback_from_20d_high=-0.04,
                vol_ratio_20d=1.4,
                turnover_rank=260,
                turnover_rank_top_n=200,
                turnover_usdt=3_000_000,
                oi_change_3d=0.15,
                funding_rate_last=0.0002,
            ),
        }

        boards = engine.rank_all_market(snapshots)
        long_by_symbol = {row.symbol: row for row in boards["long_main_wave"]}

        healthy = long_by_symbol["HEALTHY-USDT-SWAP"]
        crowded = long_by_symbol["CROWDED-USDT-SWAP"]
        thin = long_by_symbol["THIN-USDT-SWAP"]

        self.assertIn("liquid_top200_turnover", healthy.reason_codes)
        self.assertIn("oi_expansion_confirmation", healthy.reason_codes)
        self.assertGreater(healthy.selection_score, crowded.selection_score)
        self.assertIn("funding_overheated", crowded.filter_codes)
        self.assertIn("liquidity_rank_filtered", thin.filter_codes)

    def test_native_selection_profile_does_not_add_auxiliary_reason_or_filter_codes(self):
        engine = SelectionEngine(
            SymbolSelectionConfig(
                enabled=True,
                style="dual_board",
                scoring_profile="native",
                long_top_n=2,
                short_top_n=2,
            )
        )
        snapshots = {
            "BTC-USDT-SWAP": snapshot("BTC-USDT-SWAP", ret_20d=0.10, ret_60d=0.20, ret_7d=0.02),
            "ETH-USDT-SWAP": snapshot("ETH-USDT-SWAP", ret_20d=0.08, ret_60d=0.18, ret_7d=0.01),
            "NATIVE-USDT-SWAP": snapshot(
                "NATIVE-USDT-SWAP",
                ret_3d=0.06,
                ret_7d=0.16,
                ret_20d=0.42,
                ret_60d=0.80,
                pos_20d=0.78,
                pullback_from_20d_high=-0.04,
                vol_ratio_20d=1.4,
                turnover_rank=260,
                turnover_rank_top_n=200,
                turnover_usdt=3_000_000,
                oi_change_3d=0.18,
                funding_rate_last=0.025,
            ),
        }

        boards = engine.rank_all_market(snapshots)
        result = {row.symbol: row for row in boards["long_main_wave"]}["NATIVE-USDT-SWAP"]

        self.assertNotIn("liquid_top200_turnover", result.reason_codes)
        self.assertNotIn("oi_expansion_confirmation", result.reason_codes)
        self.assertNotIn("funding_overheated", result.filter_codes)
        self.assertNotIn("liquidity_rank_filtered", result.filter_codes)

    def test_strategy_forest_profiles_change_selection_scoring_with_same_universe(self):
        snapshots = {
            "BTC-USDT-SWAP": snapshot("BTC-USDT-SWAP", ret_20d=0.06, ret_60d=0.18, ret_7d=0.02),
            "ETH-USDT-SWAP": snapshot("ETH-USDT-SWAP", ret_20d=0.04, ret_60d=0.12, ret_7d=0.01),
            "LEADER-USDT-SWAP": snapshot(
                "LEADER-USDT-SWAP",
                ret_3d=0.04,
                ret_7d=0.14,
                ret_20d=0.45,
                ret_60d=0.90,
                pos_20d=0.83,
                pullback_from_20d_high=-0.04,
                vol_ratio_20d=1.4,
                turnover_rank=80,
                turnover_rank_top_n=200,
                oi_change_3d=0.08,
                funding_rate_last=0.001,
                upside_space_pct=0.35,
                btc_first_wave_follow=True,
                btc_contraction_resilient=True,
            ),
            "FAST-USDT-SWAP": snapshot(
                "FAST-USDT-SWAP",
                ret_3d=0.10,
                ret_7d=0.22,
                ret_20d=0.42,
                ret_60d=0.35,
                pos_20d=0.96,
                pullback_from_20d_high=-0.004,
                vol_ratio_20d=1.8,
                turnover_rank=30,
                turnover_rank_top_n=200,
                oi_change_3d=0.25,
                funding_rate_last=0.002,
            ),
            "CROWDED-USDT-SWAP": snapshot(
                "CROWDED-USDT-SWAP",
                ret_3d=0.07,
                ret_7d=0.16,
                ret_20d=0.48,
                ret_60d=0.95,
                pos_20d=0.82,
                pullback_from_20d_high=-0.04,
                vol_ratio_20d=1.6,
                turnover_rank=15,
                turnover_rank_top_n=200,
                oi_change_3d=0.30,
                funding_rate_last=0.025,
                upside_space_pct=0.32,
                btc_first_wave_follow=True,
                btc_contraction_resilient=True,
            ),
        }

        select_boards = SelectionEngine(
            SymbolSelectionConfig(
                enabled=True,
                style="dual_board",
                scoring_profile="langlang_01_select",
                long_top_n=3,
                short_top_n=0,
            )
        ).rank_all_market(snapshots)
        entry_boards = SelectionEngine(
            SymbolSelectionConfig(
                enabled=True,
                style="dual_board",
                scoring_profile="langlang_01_entry",
                long_top_n=3,
                short_top_n=0,
            )
        ).rank_all_market(snapshots)
        loss_boards = SelectionEngine(
            SymbolSelectionConfig(
                enabled=True,
                style="dual_board",
                scoring_profile="langlang_plus_01_loss",
                long_top_n=3,
                short_top_n=0,
            )
        ).rank_all_market(snapshots)

        select_leader = {row.symbol: row for row in select_boards["long_main_wave"]}["LEADER-USDT-SWAP"]
        entry_leader = {row.symbol: row for row in entry_boards["long_main_wave"]}["LEADER-USDT-SWAP"]
        loss_crowded = {row.symbol: row for row in loss_boards["long_main_wave"]}["CROWDED-USDT-SWAP"]

        self.assertIn("profile_select_leader_priority", select_leader.reason_codes)
        self.assertIn("profile_entry_retest_priority", entry_leader.reason_codes)
        self.assertIn("profile_plus_funding_heat_cut", loss_crowded.filter_codes)
        self.assertEqual(select_leader.features["selection_profile"], "langlang_01_select")
        self.assertGreater(select_leader.features["selection_profile_delta_long"], 0)
        self.assertLess(loss_crowded.features["selection_profile_delta_long"], 0)

    def test_strong_positive_pattern_boosts_long_selection_and_structure(self):
        engine = SelectionEngine(
            SymbolSelectionConfig(enabled=True, style="dual_board", long_top_n=2, short_top_n=0)
        )
        snapshots = {
            "BTC-USDT-SWAP": snapshot("BTC-USDT-SWAP", ret_20d=0.06, ret_60d=0.18, ret_7d=0.02),
            "ETH-USDT-SWAP": snapshot("ETH-USDT-SWAP", ret_20d=0.04, ret_60d=0.12, ret_7d=0.01),
            "PATTERN-USDT-SWAP": snapshot(
                "PATTERN-USDT-SWAP",
                ret_3d=0.02,
                ret_7d=0.05,
                ret_20d=0.16,
                ret_60d=0.50,
                pos_20d=0.62,
                pullback_from_20d_high=-0.03,
                vol_ratio_20d=1.0,
                leader_platform_start_score=0.76,
                strong_pattern_tag="leader_platform_start",
                strong_pattern_score=0.76,
                risk_pattern_score=0.0,
                pattern_reason_codes=["leader_platform_start"],
            ),
            "PLAIN-USDT-SWAP": snapshot(
                "PLAIN-USDT-SWAP",
                ret_3d=0.01,
                ret_7d=0.035,
                ret_20d=0.13,
                ret_60d=0.42,
                pos_20d=0.58,
                pullback_from_20d_high=-0.03,
                vol_ratio_20d=1.0,
            ),
        }

        boards = engine.rank_all_market(snapshots)
        rows = {row.symbol: row for row in boards["long_main_wave"]}

        self.assertTrue(rows["PATTERN-USDT-SWAP"].selected)
        self.assertIn("leader_platform_start", rows["PATTERN-USDT-SWAP"].reason_codes)
        self.assertGreater(rows["PATTERN-USDT-SWAP"].selection_score, rows["PLAIN-USDT-SWAP"].selection_score)
        self.assertNotIn("incomplete_long_main_wave_structure", rows["PATTERN-USDT-SWAP"].filter_codes)

    def test_risk_pattern_filters_and_penalizes_long_selection(self):
        engine = SelectionEngine(
            SymbolSelectionConfig(enabled=True, style="dual_board", long_top_n=2, short_top_n=0)
        )
        snapshots = {
            "BTC-USDT-SWAP": snapshot("BTC-USDT-SWAP", ret_20d=0.08, ret_60d=0.20, ret_7d=0.02),
            "ETH-USDT-SWAP": snapshot("ETH-USDT-SWAP", ret_20d=0.05, ret_60d=0.14, ret_7d=0.01),
            "SAFE-USDT-SWAP": snapshot(
                "SAFE-USDT-SWAP",
                ret_3d=0.08,
                ret_7d=0.18,
                ret_20d=0.38,
                ret_60d=0.80,
                pos_20d=0.74,
                pullback_from_20d_high=-0.04,
                vol_ratio_20d=1.5,
            ),
            "RISK-USDT-SWAP": snapshot(
                "RISK-USDT-SWAP",
                ret_3d=0.08,
                ret_7d=0.18,
                ret_20d=0.38,
                ret_60d=0.80,
                pos_20d=0.74,
                pullback_from_20d_high=-0.04,
                vol_ratio_20d=1.5,
                five_wave_late_risk_score=0.74,
                risk_pattern_tag="five_wave_late_risk",
                risk_pattern_score=0.74,
                pattern_reason_codes=["five_wave_late_risk"],
            ),
        }

        boards = engine.rank_all_market(snapshots)
        rows = {row.symbol: row for row in boards["long_main_wave"]}

        self.assertIn("five_wave_late_risk", rows["RISK-USDT-SWAP"].filter_codes)
        self.assertLess(rows["RISK-USDT-SWAP"].selection_score, rows["SAFE-USDT-SWAP"].selection_score)

    def test_strong_pattern_trend_substitute_candidate_is_labeled_in_selection(self):
        engine = SelectionEngine(
            SymbolSelectionConfig(enabled=True, style="dual_board", long_top_n=2, short_top_n=0)
        )
        snapshots = {
            "BTC-USDT-SWAP": snapshot("BTC-USDT-SWAP", ret_20d=0.06, ret_60d=0.18, ret_7d=0.02),
            "ETH-USDT-SWAP": snapshot("ETH-USDT-SWAP", ret_20d=0.04, ret_60d=0.12, ret_7d=0.01),
            "PIT-USDT-SWAP": snapshot(
                "PIT-USDT-SWAP",
                ret_3d=0.03,
                ret_7d=0.06,
                ret_20d=0.10,
                ret_60d=0.36,
                pos_20d=0.48,
                pullback_from_20d_high=-0.06,
                vol_ratio_20d=1.2,
                latest_close=109.0,
                ma_5=104.0,
                ma_20=110.0,
                golden_pit_reclaim_score=0.78,
                strong_pattern_tag="golden_pit_reclaim",
                strong_pattern_score=0.78,
                h1_golden_pit_reclaim_score=0.50,
                pattern_reason_codes=["golden_pit_fast_reclaim"],
            ),
            "PLAIN-USDT-SWAP": snapshot(
                "PLAIN-USDT-SWAP",
                ret_3d=0.02,
                ret_7d=0.04,
                ret_20d=0.09,
                ret_60d=0.30,
                pos_20d=0.46,
                pullback_from_20d_high=-0.04,
                vol_ratio_20d=1.2,
            ),
        }

        boards = engine.rank_all_market(snapshots)
        rows = {row.symbol: row for row in boards["long_main_wave"]}

        self.assertTrue(rows["PIT-USDT-SWAP"].selected)
        self.assertIn("golden_pit_reclaim", rows["PIT-USDT-SWAP"].reason_codes)
        self.assertIn("strong_pattern_trend_substitute_candidate", rows["PIT-USDT-SWAP"].reason_codes)
        self.assertNotIn("incomplete_long_main_wave_structure", rows["PIT-USDT-SWAP"].filter_codes)

    def test_spoon_bottom_stays_catch_up_even_when_numeric_leader_conditions_match(self):
        engine = SelectionEngine(
            SymbolSelectionConfig(enabled=True, style="dual_board", long_top_n=2, short_top_n=0)
        )
        snapshots = {
            "BTC-USDT-SWAP": snapshot("BTC-USDT-SWAP", ret_20d=0.08, ret_60d=0.20, ret_7d=0.02),
            "ETH-USDT-SWAP": snapshot("ETH-USDT-SWAP", ret_20d=0.05, ret_60d=0.14, ret_7d=0.01),
            "SPOON-USDT-SWAP": snapshot(
                "SPOON-USDT-SWAP",
                ret_3d=0.06,
                ret_7d=0.14,
                ret_20d=0.32,
                ret_60d=0.52,
                pos_20d=0.68,
                pullback_from_20d_high=-0.04,
                vol_ratio_20d=1.4,
                upside_space_pct=0.30,
                rel_btc=0.08,
                spoon_bottom_confirmed_score=0.72,
                strong_pattern_tag="spoon_bottom_confirmed",
                strong_pattern_score=0.72,
                pattern_reason_codes=["spoon_bottom_confirmed"],
            ),
            "PLAIN-USDT-SWAP": snapshot(
                "PLAIN-USDT-SWAP",
                ret_3d=0.02,
                ret_7d=0.05,
                ret_20d=0.12,
                ret_60d=0.24,
                pos_20d=0.52,
                pullback_from_20d_high=-0.04,
                vol_ratio_20d=1.0,
            ),
        }

        boards = engine.rank_all_market(snapshots)
        rows = {row.symbol: row for row in boards["long_main_wave"]}

        self.assertIn("spoon_bottom_confirmed", rows["SPOON-USDT-SWAP"].reason_codes)
        self.assertIn("catch_up_short_hold", rows["SPOON-USDT-SWAP"].reason_codes)
        self.assertEqual(rows["SPOON-USDT-SWAP"].features["symbol_selection_tag"], "catch_up_short_hold")
        self.assertNotIn("leader_altcoin", rows["SPOON-USDT-SWAP"].reason_codes)

    def test_wyckoff_long_confirmation_boosts_long_board(self):
        engine = SelectionEngine(
            SymbolSelectionConfig(enabled=True, style="dual_board", long_top_n=2, short_top_n=0)
        )
        snapshots = {
            "BTC-USDT-SWAP": snapshot("BTC-USDT-SWAP", ret_20d=0.08, ret_60d=0.20, ret_7d=0.02),
            "ETH-USDT-SWAP": snapshot("ETH-USDT-SWAP", ret_20d=0.05, ret_60d=0.14, ret_7d=0.01),
            "WYCK-LONG-USDT-SWAP": snapshot(
                "WYCK-LONG-USDT-SWAP",
                ret_3d=0.03,
                ret_7d=0.07,
                ret_20d=0.16,
                ret_60d=0.38,
                pos_20d=0.50,
                pullback_from_20d_high=-0.06,
                vol_ratio_20d=1.2,
                upside_space_pct=0.24,
                latest_close=109.0,
                ma_5=104.0,
                ma_20=110.0,
                wyckoff_phase_tag="accumulation",
                wyckoff_long_setup_tag="spring_reclaim",
                wyckoff_long_score=0.72,
                h1_wyckoff_long_score=0.52,
                wyckoff_reason_codes=["wyckoff_spring_reclaim"],
            ),
            "PLAIN-USDT-SWAP": snapshot(
                "PLAIN-USDT-SWAP",
                ret_3d=0.02,
                ret_7d=0.04,
                ret_20d=0.15,
                ret_60d=0.34,
                pos_20d=0.48,
                pullback_from_20d_high=-0.04,
                vol_ratio_20d=1.1,
            ),
        }

        boards = engine.rank_all_market(snapshots)
        rows = {row.symbol: row for row in boards["long_main_wave"]}

        self.assertTrue(rows["WYCK-LONG-USDT-SWAP"].selected)
        self.assertIn("wyckoff_long_confirmed", rows["WYCK-LONG-USDT-SWAP"].reason_codes)
        self.assertIn("wyckoff_trend_substitute_candidate", rows["WYCK-LONG-USDT-SWAP"].reason_codes)
        self.assertNotIn("incomplete_long_main_wave_structure", rows["WYCK-LONG-USDT-SWAP"].filter_codes)

    def test_wyckoff_short_confirmation_adds_short_board_reason(self):
        engine = SelectionEngine(
            SymbolSelectionConfig(enabled=True, style="dual_board", long_top_n=0, short_top_n=2)
        )
        snapshots = {
            "BTC-USDT-SWAP": snapshot("BTC-USDT-SWAP", ret_20d=0.08, ret_60d=0.20, ret_7d=0.02),
            "ETH-USDT-SWAP": snapshot("ETH-USDT-SWAP", ret_20d=0.05, ret_60d=0.14, ret_7d=0.01),
            "WYCK-SHORT-USDT-SWAP": snapshot(
                "WYCK-SHORT-USDT-SWAP",
                ret_3d=-0.03,
                ret_7d=-0.08,
                ret_20d=-0.12,
                ret_60d=0.10,
                pos_20d=0.34,
                vol_ratio_20d=1.4,
                latest_close=92.0,
                ma_5=95.0,
                ma_20=100.0,
                wyckoff_phase_tag="distribution",
                wyckoff_short_setup_tag="sow_breakdown",
                wyckoff_short_score=0.74,
                m15_wyckoff_short_score=0.50,
                wyckoff_reason_codes=["wyckoff_sow_breakdown"],
            ),
        }

        boards = engine.rank_all_market(snapshots)
        rows = {row.symbol: row for row in boards["short_waterfall"]}

        self.assertTrue(rows["WYCK-SHORT-USDT-SWAP"].selected)
        self.assertIn("wyckoff_short_confirmed", rows["WYCK-SHORT-USDT-SWAP"].reason_codes)
        self.assertEqual(rows["WYCK-SHORT-USDT-SWAP"].features["symbol_selection_tag"], "short_waterfall")

    def test_wyckoff_distribution_risk_penalizes_long_score(self):
        engine = SelectionEngine(
            SymbolSelectionConfig(enabled=True, style="dual_board", long_top_n=2, short_top_n=0)
        )
        common = dict(
            ret_3d=0.06,
            ret_7d=0.15,
            ret_20d=0.42,
            ret_60d=0.86,
            pos_20d=0.78,
            pullback_from_20d_high=-0.04,
            vol_ratio_20d=1.4,
            upside_space_pct=0.30,
        )
        snapshots = {
            "BTC-USDT-SWAP": snapshot("BTC-USDT-SWAP", ret_20d=0.08, ret_60d=0.20, ret_7d=0.02),
            "ETH-USDT-SWAP": snapshot("ETH-USDT-SWAP", ret_20d=0.05, ret_60d=0.14, ret_7d=0.01),
            "HEALTHY-USDT-SWAP": snapshot("HEALTHY-USDT-SWAP", **common),
            "DISTRIBUTION-USDT-SWAP": snapshot(
                "DISTRIBUTION-USDT-SWAP",
                **common,
                wyckoff_phase_tag="distribution",
                wyckoff_risk_score=0.74,
                wyckoff_exit_score=0.72,
                wyckoff_exit_tag="utad_risk",
                wyckoff_reason_codes=["wyckoff_utad_risk", "wyckoff_effort_result_divergence"],
            ),
        }

        boards = engine.rank_all_market(snapshots)
        rows = {row.symbol: row for row in boards["long_main_wave"]}

        self.assertIn("wyckoff_distribution_risk", rows["DISTRIBUTION-USDT-SWAP"].filter_codes)
        self.assertLess(rows["DISTRIBUTION-USDT-SWAP"].selection_score, rows["HEALTHY-USDT-SWAP"].selection_score)

    def test_long_risk_profile_uses_long_only_filters(self):
        engine = SelectionEngine(
            SymbolSelectionConfig(
                enabled=True,
                style="dual_board",
                long_top_n=2,
                short_top_n=0,
                scoring_profile="langlang_01_risk",
            )
        )
        common = dict(
            ret_3d=0.06,
            ret_7d=0.15,
            ret_20d=0.42,
            ret_60d=0.86,
            vol_ratio_20d=1.4,
            upside_space_pct=0.30,
        )
        snapshots = {
            "BTC-USDT-SWAP": snapshot("BTC-USDT-SWAP", ret_20d=0.08, ret_60d=0.20, ret_7d=0.02),
            "ETH-USDT-SWAP": snapshot("ETH-USDT-SWAP", ret_20d=0.05, ret_60d=0.14, ret_7d=0.01),
            "RETEST-USDT-SWAP": snapshot(
                "RETEST-USDT-SWAP",
                **common,
                pos_20d=0.74,
                pullback_from_20d_high=-0.04,
            ),
            "CHASE-USDT-SWAP": snapshot(
                "CHASE-USDT-SWAP",
                **common,
                pos_20d=0.96,
                pullback_from_20d_high=-0.004,
            ),
        }

        boards = engine.rank_all_market(snapshots)
        rows = {row.symbol: row for row in boards["long_main_wave"]}

        self.assertIn("high_position_no_structure", rows["CHASE-USDT-SWAP"].filter_codes)
        self.assertIn("profile_risk_high_position_cut", rows["CHASE-USDT-SWAP"].filter_codes)
        self.assertLess(rows["CHASE-USDT-SWAP"].selection_score, rows["RETEST-USDT-SWAP"].selection_score)

    def test_short_board_does_not_inherit_long_only_risk_filters(self):
        engine = SelectionEngine(
            SymbolSelectionConfig(enabled=True, style="dual_board", long_top_n=0, short_top_n=2)
        )
        snapshots = {
            "BTC-USDT-SWAP": snapshot("BTC-USDT-SWAP", ret_20d=0.08, ret_60d=0.20, ret_7d=0.02),
            "ETH-USDT-SWAP": snapshot("ETH-USDT-SWAP", ret_20d=0.05, ret_60d=0.14, ret_7d=0.01),
            "SHORT-RISK-USDT-SWAP": snapshot(
                "SHORT-RISK-USDT-SWAP",
                ret_3d=-0.06,
                ret_7d=-0.14,
                ret_20d=-0.34,
                ret_60d=-0.44,
                pos_20d=0.12,
                pullback_from_20d_high=-0.42,
                vol_ratio_20d=1.7,
                latest_close=78.0,
                ma_5=82.0,
                ma_20=96.0,
                high_20d=118.0,
                failed_rebound_below_platform=True,
                five_wave_late_risk_score=0.74,
                false_breakout_risk_score=0.70,
                risk_pattern_tag="five_wave_late_risk",
                risk_pattern_score=0.74,
                wyckoff_phase_tag="distribution",
                wyckoff_risk_score=0.72,
                wyckoff_exit_score=0.71,
            ),
        }

        boards = engine.rank_all_market(snapshots)
        rows = {row.symbol: row for row in boards["short_waterfall"]}
        result = rows["SHORT-RISK-USDT-SWAP"]

        self.assertTrue(result.selected)
        self.assertIn("failed_rebound_below_platform", result.reason_codes)
        self.assertNotIn("five_wave_late_risk", result.filter_codes)
        self.assertNotIn("false_breakout_risk", result.filter_codes)
        self.assertNotIn("wyckoff_distribution_risk", result.filter_codes)


if __name__ == "__main__":
    unittest.main()

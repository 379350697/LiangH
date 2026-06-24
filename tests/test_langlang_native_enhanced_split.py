import csv
import json
import os
import tempfile
import unittest

from langlang_trader.features import FeatureSnapshot
from langlang_trader.models import FailureFilter, Side, StrategyAction
from langlang_trader.optimize import (
    HistoricalReplayOptimizer,
    OptimizerConfig,
    _classify_wave_stage,
    _default_variants,
    _effective_min_validation_signals,
    _entry_position_from_wave,
    _output_suffix,
)
from langlang_trader.strategy import (
    LangLangEnhancedVariant,
    LangLangNativeVariant,
    RulesLangLangEnhancedFinalStrategy,
    RulesLangLangEnhancedPayoffStrategy,
    RulesLangLangEnhancedStrategy,
    RulesLangLangNativeFinalStrategy,
    RulesLangLangNativePayoffStrategy,
    RulesLangLangNativeStrategy,
    default_langlang_enhanced_grid,
    default_langlang_native_grid,
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
        "vol_ratio_20d": 1.9,
        "h1_ret_24": 0.028,
        "h1_pullback_from_high": -0.032,
        "m15_ret_8": 0.010,
        "m5_ret_6": 0.006,
        "upside_space_pct": 0.28,
        "bottom_lift_confirmed": True,
        "stop_loss_cluster_24h": 0,
        "historical_match_score": 0.0,
        "matched_trade_examples": [],
        "market_season": "summer",
        "symbol_cycle": "platform_start",
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


def write_trades(path):
    rows = [
        {
            "trade_id": "wave-long",
            "entry_time": "2024-02-15 00:00:00",
            "exit_time": "2024-02-16 00:00:00",
            "symbol": "WAVE-USDT-SWAP",
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


class LangLangNativeEnhancedSplitTest(unittest.TestCase):
    def test_native_enters_document_setup_without_historical_support_gate(self):
        strategy = RulesLangLangNativeStrategy(LangLangNativeVariant(variant_id="native-doc"))

        decision = strategy.decide(snapshot())

        self.assertEqual(decision.action, StrategyAction.ENTER)
        signal = decision.signal
        self.assertIsNotNone(signal)
        self.assertEqual(signal.strategy_version, "rules_langlang_native_v1")
        self.assertEqual(signal.side, Side.LONG)
        self.assertEqual(signal.decision_trace["entry_position_id"], "1_startup_long")
        self.assertEqual(signal.decision_trace["strategy_line"], "native")
        self.assertEqual(signal.historical_match_score, 0.0)

    def test_enhanced_requires_historical_support_for_non_exploratory_bots(self):
        strategy = RulesLangLangEnhancedStrategy(LangLangEnhancedVariant(variant_id="enhanced-strict"))

        decision = strategy.decide(snapshot())

        self.assertEqual(decision.action, StrategyAction.SKIP)
        self.assertIn(FailureFilter.NO_HISTORICAL_SUPPORT, decision.filter_codes)

    def test_enhanced_filters_big_loss_and_derivative_overheat_risks(self):
        strategy = RulesLangLangEnhancedStrategy(LangLangEnhancedVariant(variant_id="enhanced-risk"))

        big_loss = strategy.decide(
            snapshot(
                historical_match_score=0.80,
                matched_trade_examples=[{"trade_id": "loss-1", "labels": ["big_loss"]}],
                big_loss_overlap_count=1,
            )
        )
        funding = strategy.decide(
            snapshot(
                historical_match_score=0.80,
                matched_trade_examples=[{"trade_id": "win-1", "labels": ["big_win"]}],
                funding_rate_last=0.025,
            )
        )

        self.assertEqual(big_loss.action, StrategyAction.SKIP)
        self.assertIn(FailureFilter.BIG_LOSS_SIMILARITY, big_loss.filter_codes)
        self.assertEqual(funding.action, StrategyAction.SKIP)
        self.assertIn(FailureFilter.CHASE_OVERHEAT, funding.filter_codes)

    def test_factory_grids_and_optimizer_suffixes_are_split(self):
        native_grid = default_langlang_native_grid()
        enhanced_grid = default_langlang_enhanced_grid()

        self.assertGreaterEqual(len(native_grid), 2)
        self.assertGreaterEqual(len(enhanced_grid), 2)
        self.assertIsInstance(strategy_from_version("rules_langlang_native_v1", native_grid[0]), RulesLangLangNativeStrategy)
        self.assertIsInstance(strategy_from_version("rules_langlang_enhanced_v1", enhanced_grid[0]), RulesLangLangEnhancedStrategy)
        self.assertTrue(all(isinstance(row, LangLangNativeVariant) for row in _default_variants("rules_langlang_native_v1")))
        self.assertTrue(all(isinstance(row, LangLangEnhancedVariant) for row in _default_variants("rules_langlang_enhanced_v1")))
        self.assertEqual(_output_suffix("rules_langlang_native_v1"), "_native_v1")
        self.assertEqual(_output_suffix("rules_langlang_enhanced_v1"), "_enhanced_v1")
        self.assertIsInstance(strategy_from_version("rules_langlang_native_final", native_grid[0]), RulesLangLangNativeFinalStrategy)
        self.assertIsInstance(strategy_from_version("rules_langlang_enhanced_final", enhanced_grid[0]), RulesLangLangEnhancedFinalStrategy)
        self.assertTrue(all(isinstance(row, LangLangNativeVariant) for row in _default_variants("rules_langlang_native_final")))
        self.assertTrue(all(isinstance(row, LangLangEnhancedVariant) for row in _default_variants("rules_langlang_enhanced_final")))
        self.assertEqual(_output_suffix("rules_langlang_native_final"), "_native_final")
        self.assertEqual(_output_suffix("rules_langlang_enhanced_final"), "_enhanced_final")

    def test_final_strategy_versions_keep_native_and_enhanced_semantics(self):
        native = RulesLangLangNativeFinalStrategy(LangLangNativeVariant(variant_id="native-final"))
        enhanced = RulesLangLangEnhancedFinalStrategy(LangLangEnhancedVariant(variant_id="enhanced-final"))

        native_decision = native.decide(snapshot())
        enhanced_decision = enhanced.decide(snapshot())

        self.assertEqual(native_decision.action, StrategyAction.ENTER)
        self.assertEqual(native_decision.signal.strategy_version, "rules_langlang_native_final")
        self.assertEqual(native_decision.signal.decision_trace["strategy_line"], "native_final")
        self.assertEqual(enhanced_decision.action, StrategyAction.SKIP)
        self.assertIn(FailureFilter.NO_HISTORICAL_SUPPORT, enhanced_decision.filter_codes)

    def test_final_optimizer_never_treats_zero_signal_variants_as_valid(self):
        self.assertEqual(_effective_min_validation_signals("rules_langlang_native_final", 0), 1)
        self.assertEqual(_effective_min_validation_signals("rules_langlang_enhanced_final", 0), 1)
        self.assertEqual(_effective_min_validation_signals("rules_langlang_enhanced_v1", 0), 0)

    def test_optimizer_writes_separate_native_and_enhanced_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            trades = os.path.join(tmp, "standard_trades.csv")
            cache = os.path.join(tmp, "kline_cache")
            native_out = os.path.join(tmp, "native")
            enhanced_out = os.path.join(tmp, "enhanced")
            write_trades(trades)
            write_cache(cache, "WAVE-USDT-SWAP")

            native = HistoricalReplayOptimizer(
                OptimizerConfig(
                    trades_csv=trades,
                    kline_cache_dir=cache,
                    out_dir=native_out,
                    strategy_version="rules_langlang_native_final",
                    variants=[
                        LangLangNativeVariant(
                            variant_id="native-smoke",
                            allowed_side="long",
                            ret_20d_min=0.10,
                            ret_60d_min=0.20,
                            min_upside_space_pct=0.0,
                        )
                    ],
                    top_n=1,
                    min_validation_signals=0,
                )
            ).run()
            enhanced = HistoricalReplayOptimizer(
                OptimizerConfig(
                    trades_csv=trades,
                    kline_cache_dir=cache,
                    out_dir=enhanced_out,
                    strategy_version="rules_langlang_enhanced_final",
                    variants=[
                        LangLangEnhancedVariant(
                            variant_id="enhanced-smoke",
                            allowed_side="long",
                            exploratory=True,
                            ret_20d_min=0.10,
                            ret_60d_min=0.20,
                            min_upside_space_pct=0.0,
                        )
                    ],
                    top_n=1,
                    min_validation_signals=0,
                )
            ).run()

            self.assertEqual(native.leaderboard_path, os.path.join(native_out, "leaderboard_native_final.csv"))
            self.assertEqual(enhanced.leaderboard_path, os.path.join(enhanced_out, "leaderboard_enhanced_final.csv"))
            self.assertTrue(os.path.exists(os.path.join(native_out, "native_fit_report_final.md")))
            self.assertTrue(os.path.exists(os.path.join(enhanced_out, "enhanced_fit_report_final.md")))
            for out_dir in (native_out, enhanced_out):
                self.assertTrue(os.path.exists(os.path.join(out_dir, "distill_dataset_final.csv")))
                self.assertTrue(os.path.exists(os.path.join(out_dir, "trade_explanation_matrix_final.csv")))
                self.assertTrue(os.path.exists(os.path.join(out_dir, "overfit_audit_final.md")))
                with open(os.path.join(out_dir, "distill_dataset_final.csv"), encoding="utf-8") as f:
                    distill_rows = list(csv.DictReader(f))
                with open(os.path.join(out_dir, "trade_explanation_matrix_final.csv"), encoding="utf-8") as f:
                    explanation_rows = list(csv.DictReader(f))
                self.assertEqual(distill_rows[0]["state_action_outcome_status"], "available")
                self.assertIn(distill_rows[0]["expert_trade_label"], {"rational_win", "right_tail"})
                self.assertNotEqual(explanation_rows[0]["native_explanation"], "")
                self.assertNotEqual(explanation_rows[0]["why_stop_or_hold"], "")
            with open(native.selected_config_path, encoding="utf-8") as f:
                native_config = json.load(f)
            with open(enhanced.selected_config_path, encoding="utf-8") as f:
                enhanced_config = json.load(f)
            self.assertEqual(native_config["strategy_version"], "rules_langlang_native_final")
            self.assertEqual(enhanced_config["strategy_version"], "rules_langlang_enhanced_final")
            self.assertEqual(native_config["selection"]["scoring_profile"], "native")
            self.assertEqual(enhanced_config["selection"]["scoring_profile"], "enhanced")

    def test_payoff_strategy_versions_add_wave_stage_and_payoff_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            trades = os.path.join(tmp, "standard_trades.csv")
            cache = os.path.join(tmp, "kline_cache")
            native_out = os.path.join(tmp, "native_payoff")
            enhanced_out = os.path.join(tmp, "enhanced_payoff")
            write_trades(trades)
            write_cache(cache, "WAVE-USDT-SWAP")
            evidence_dir = os.path.join(tmp, "excel_digest")
            os.makedirs(evidence_dir, exist_ok=True)
            with open(os.path.join(evidence_dir, "trade_sheet_label_matrix.csv"), "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "trade_id",
                        "sheet_membership",
                        "manual_review_text",
                        "btc_cycle_label",
                        "entry_position_label",
                        "space_bucket",
                        "stop_loss_bucket",
                        "hold_time_bucket",
                        "sheet_label_status",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "trade_id": "wave-long",
                        "sheet_membership": "5%波动以上单子|单笔收益率（看止损）|实际振幅与收益分析（看空间）",
                        "manual_review_text": "主升浪二波启动，回踩企稳",
                        "btc_cycle_label": "大饼调整结束后共振",
                        "entry_position_label": "4",
                        "space_bucket": "move_gt_20pct",
                        "stop_loss_bucket": "profit_gt_20pct",
                        "hold_time_bucket": "hold_gt_1d",
                        "sheet_label_status": "labeled",
                    }
                )
            with open(os.path.join(evidence_dir, "excel_evidence_event_dataset.csv"), "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "event_id",
                        "trade_id",
                        "symbol",
                        "side",
                        "workbook",
                        "sheet_name",
                        "sheet_role",
                        "source_row_number",
                        "field_name",
                        "field_value",
                        "event_semantic",
                        "evidence_role",
                        "evidence_weight",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "event_id": "ev-1",
                        "trade_id": "wave-long",
                        "symbol": "WAVE-USDT-SWAP",
                        "side": "long",
                        "workbook": "langlang.xlsx",
                        "sheet_name": "5%波动以上单子",
                        "sheet_role": "manual_review",
                        "source_row_number": "2",
                        "field_name": "复盘操作细节",
                        "field_value": "主升浪二波启动，回踩企稳",
                        "event_semantic": "manual_review_text",
                        "evidence_role": "manual_review",
                        "evidence_weight": "1.5",
                    }
                )
                writer.writerow(
                    {
                        "event_id": "ev-2",
                        "trade_id": "wave-long",
                        "symbol": "WAVE-USDT-SWAP",
                        "side": "long",
                        "workbook": "langlang.xlsx",
                        "sheet_name": "实际振幅与收益分析（看空间）",
                        "sheet_role": "derived_stats",
                        "source_row_number": "2",
                        "field_name": "收益率/倍数=实际振幅",
                        "field_value": "0.31",
                        "event_semantic": "space_or_move",
                        "evidence_role": "derived_stats",
                        "evidence_weight": "1.2",
                    }
                )

            native = HistoricalReplayOptimizer(
                OptimizerConfig(
                    trades_csv=trades,
                    kline_cache_dir=cache,
                    out_dir=native_out,
                    strategy_version="rules_langlang_native_payoff_v1",
                    variants=[
                        LangLangNativeVariant(
                            variant_id="native-payoff-smoke",
                            allowed_side="long",
                            ret_20d_min=0.10,
                            ret_60d_min=0.20,
                            min_upside_space_pct=0.0,
                        )
                    ],
                    top_n=1,
                    min_validation_signals=0,
                )
            ).run()
            enhanced = HistoricalReplayOptimizer(
                OptimizerConfig(
                    trades_csv=trades,
                    kline_cache_dir=cache,
                    out_dir=enhanced_out,
                    strategy_version="rules_langlang_enhanced_payoff_v1",
                    variants=[
                        LangLangEnhancedVariant(
                            variant_id="enhanced-payoff-smoke",
                            allowed_side="long",
                            exploratory=True,
                            ret_20d_min=0.10,
                            ret_60d_min=0.20,
                            min_upside_space_pct=0.0,
                        )
                    ],
                    top_n=1,
                    min_validation_signals=0,
                )
            ).run()

            self.assertEqual(native.leaderboard_path, os.path.join(native_out, "leaderboard_native_payoff_v1.csv"))
            self.assertEqual(enhanced.leaderboard_path, os.path.join(enhanced_out, "leaderboard_enhanced_payoff_v1.csv"))
            self.assertIsInstance(
                strategy_from_version("rules_langlang_native_payoff_v1", LangLangNativeVariant()),
                RulesLangLangNativePayoffStrategy,
            )
            self.assertIsInstance(
                strategy_from_version("rules_langlang_enhanced_payoff_v1", LangLangEnhancedVariant()),
                RulesLangLangEnhancedPayoffStrategy,
            )
            for out_dir in (native_out, enhanced_out):
                for filename in (
                    "wave_stage_dataset_v1.csv",
                    "symbol_selection_cross_section_v1.csv",
                    "trade_wave_explanation_matrix_v1.csv",
                    "right_tail_capture_report_v1.md",
                    "loss_suppression_report_v1.md",
                ):
                    self.assertTrue(os.path.exists(os.path.join(out_dir, filename)), filename)
                with open(os.path.join(out_dir, "wave_stage_dataset_v1.csv"), encoding="utf-8") as f:
                    wave_rows = list(csv.DictReader(f))
                with open(os.path.join(out_dir, "symbol_selection_cross_section_v1.csv"), encoding="utf-8") as f:
                    selection_rows = list(csv.DictReader(f))
                with open(os.path.join(out_dir, "trade_wave_explanation_matrix_v1.csv"), encoding="utf-8") as f:
                    explanation_rows = list(csv.DictReader(f))
                with open(os.path.join(out_dir, "leaderboard" + _output_suffix("rules_langlang_native_payoff_v1" if out_dir == native_out else "rules_langlang_enhanced_payoff_v1") + ".csv"), encoding="utf-8") as f:
                    leaderboard_rows = list(csv.DictReader(f))
                self.assertEqual(wave_rows[0]["wave_stage"], "main_wave")
                self.assertEqual(wave_rows[0]["entry_position_id"], "1_startup_long")
                self.assertIn("单笔收益率", wave_rows[0]["excel_sheet_membership"])
                self.assertIn("leader_altcoin", selection_rows[0]["selection_reason_codes"])
                self.assertEqual(selection_rows[0]["excel_space_bucket"], "move_gt_20pct")
                self.assertEqual(explanation_rows[0]["why_this_stage"], "main_wave")
                self.assertIn("主升浪二波启动", explanation_rows[0]["excel_manual_review_text"])
                self.assertEqual(explanation_rows[0]["excel_entry_position_label"], "4")
                self.assertEqual(explanation_rows[0]["excel_event_count"], "2")
                self.assertIn("manual_review_text", explanation_rows[0]["excel_event_semantics"])
                self.assertIn("5%波动以上单子", explanation_rows[0]["excel_event_sources"])
                self.assertIn("right_tail_capture_score", leaderboard_rows[0])
                self.assertIn("payoff_asymmetry_score", leaderboard_rows[0])
                self.assertIn("loss_suppression_score", leaderboard_rows[0])
                self.assertIn("excel_event_support_score", leaderboard_rows[0])

    def test_wave_stage_classifier_uses_only_current_feature_snapshot(self):
        stage = _classify_wave_stage(
            {
                "ret_20d": 0.42,
                "ret_60d": 1.10,
                "pos_20d": 0.76,
                "pullback_from_20d_high": -0.04,
                "vol_ratio_20d": 1.8,
            }
        )

        self.assertEqual(stage, "main_wave")
        self.assertEqual(_entry_position_from_wave(stage, "long", 0.50), "1_startup_long")
        self.assertEqual(
            _classify_wave_stage({"ret_20d": 0.18, "ret_60d": 0.55, "pos_20d": 0.96, "pullback_from_20d_high": -0.01}),
            "exhaustion",
        )


if __name__ == "__main__":
    unittest.main()

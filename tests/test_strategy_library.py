import csv
from datetime import datetime, timedelta, timezone
import json
import os
import sqlite3
import tempfile
import unittest

from langlang_trader.optimize import HistoricalReplayOptimizer, OptimizerConfig
from langlang_trader.strategy import StrategyVariant
from langlang_trader.strategy_library import (
    DEFAULT_REGISTRY_PATH,
    StrategyLibrary,
    StrategyLibraryError,
    compare_variants,
    ingest_leaderboard,
    render_strategy_library_report,
)


class StrategyLibraryTest(unittest.TestCase):
    def _write_registry(self, path: str) -> None:
        registry = {
            "schema_version": 1,
            "families": [
                {
                    "family_id": "langlang",
                    "name": "浪浪交易法",
                    "source_basis": ["confirmed_pdf", "excel_evidence"],
                }
            ],
            "strategies": [
                {
                    "strategy_id": "native_payoff",
                    "family_id": "langlang",
                    "strategy_version": "rules_langlang_native_payoff_v1",
                    "status": "backtested",
                    "hypothesis": "复刻原文五浪和博大损小结构。",
                    "promotion_rules": ["right_tail_capture_score > 0"],
                }
            ],
            "variants": [
                {
                    "variant_id": "native_parent",
                    "display_name": "langlang-test-01",
                    "lineage_group": "langlang_payoff",
                    "strategy_id": "native_payoff",
                    "parent_id": None,
                    "strategy_version": "rules_langlang_native_payoff_v1",
                    "status": "backtested",
                    "hypothesis": "原生基线。",
                    "factor_set": {
                        "line": "native",
                        "entry": ["document_six_positions"],
                        "filters": ["document_filters"],
                    },
                    "changed_factors": [],
                    "core_logic": ["document_core", "six_entry_positions"],
                    "source_basis": ["pdf"],
                    "risk_profile": {"leverage": "document_default"},
                    "iteration_notes": "测试原生主干。",
                    "promotion_rules": ["manual_review_required"],
                },
                {
                    "variant_id": "enhanced_child",
                    "display_name": "langlang-test-plus-01",
                    "lineage_group": "langlang_payoff",
                    "strategy_id": "native_payoff",
                    "parent_id": "native_parent",
                    "strategy_version": "rules_langlang_enhanced_payoff_v1",
                    "status": "paper_running",
                    "hypothesis": "在原生候选上压制大亏。",
                    "factor_set": {
                        "line": "enhanced",
                        "entry": ["document_six_positions"],
                        "filters": ["document_filters", "big_loss_similarity_filter"],
                    },
                    "changed_factors": ["big_loss_similarity_filter"],
                    "core_logic": ["document_core", "big_loss_similarity_filter"],
                    "source_basis": ["pdf", "excel_events"],
                    "risk_profile": {"leverage": "document_default"},
                    "iteration_notes": "测试增强主干。",
                    "promotion_rules": ["loss_suppression_score >= parent"],
                },
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(registry, f, ensure_ascii=False, indent=2)

    def _write_leaderboard(self, path: str) -> None:
        rows = [
            {
                "rank": "1",
                "variant_id": "native_parent",
                "score": "0.5",
                "eligible": "True",
                "validation_signals": "10",
                "validation_profit_factor": "1.1",
                "max_drawdown": "0.12",
                "right_tail_capture_score": "0.2",
                "loss_suppression_score": "0.7",
                "avg_win_loss_ratio": "3.0",
                "big_loss_overlap": "0.3",
                "validation_net_pnl": "0.4",
            },
            {
                "rank": "2",
                "variant_id": "enhanced_child",
                "score": "0.7",
                "eligible": "True",
                "validation_signals": "12",
                "validation_profit_factor": "1.4",
                "max_drawdown": "0.08",
                "right_tail_capture_score": "0.25",
                "loss_suppression_score": "0.9",
                "avg_win_loss_ratio": "4.0",
                "big_loss_overlap": "0.1",
                "validation_net_pnl": "0.6",
            },
        ]
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def test_registry_validates_tree_and_rejects_missing_parent_or_cycles(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = os.path.join(tmp, "registry.json")
            self._write_registry(registry_path)

            library = StrategyLibrary.load(registry_path)
            self.assertEqual(library.lineage("enhanced_child"), ["native_parent", "enhanced_child"])
            self.assertEqual(library.variant("enhanced_child").changed_factors, ["big_loss_similarity_filter"])
            self.assertEqual(library.variant("enhanced_child").display_name, "langlang-test-plus-01")
            self.assertEqual(library.variant("enhanced_child").lineage_group, "langlang_payoff")

            with open(registry_path, encoding="utf-8") as f:
                raw = json.load(f)
            raw["variants"][1]["parent_id"] = "missing_parent"
            bad_path = os.path.join(tmp, "bad_missing_parent.json")
            with open(bad_path, "w", encoding="utf-8") as f:
                json.dump(raw, f)
            with self.assertRaisesRegex(StrategyLibraryError, "missing parent"):
                StrategyLibrary.load(bad_path)

            raw["variants"][0]["parent_id"] = "enhanced_child"
            raw["variants"][1]["parent_id"] = "native_parent"
            cycle_path = os.path.join(tmp, "bad_cycle.json")
            with open(cycle_path, "w", encoding="utf-8") as f:
                json.dump(raw, f)
            with self.assertRaisesRegex(StrategyLibraryError, "cycle"):
                StrategyLibrary.load(cycle_path)

    def test_ingests_leaderboard_runs_and_compares_factor_deltas(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = os.path.join(tmp, "registry.json")
            leaderboard_path = os.path.join(tmp, "leaderboard.csv")
            db_path = os.path.join(tmp, "strategy_library.sqlite3")
            self._write_registry(registry_path)
            self._write_leaderboard(leaderboard_path)

            result = ingest_leaderboard(
                registry_path=registry_path,
                leaderboard_path=leaderboard_path,
                db_path=db_path,
                run_id="run-001",
                strategy_version="rules_langlang_enhanced_payoff_v1",
                data_snapshot_id="snapshot-a",
                artifact_paths={"leaderboard": leaderboard_path},
            )
            self.assertEqual(result.inserted_runs, 2)

            conn = sqlite3.connect(db_path)
            rows = conn.execute(
                "select run_id, variant_id, parent_variant_id, profit_factor, right_tail_capture, loss_suppression "
                "from strategy_runs order by variant_id"
            ).fetchall()
            conn.close()
            self.assertEqual(rows[0][0], "run-001")
            self.assertEqual(rows[1][1], "native_parent")
            self.assertEqual(rows[0][2], "native_parent")
            self.assertAlmostEqual(rows[0][3], 1.4)
            self.assertAlmostEqual(rows[0][4], 0.25)
            self.assertAlmostEqual(rows[0][5], 0.9)

            delta = compare_variants(
                registry_path=registry_path,
                db_path=db_path,
                parent_variant_id="native_parent",
                child_variant_id="enhanced_child",
            )
            self.assertEqual(delta["changed_factors"], ["big_loss_similarity_filter"])
            self.assertAlmostEqual(delta["right_tail_capture_delta"], 0.05)
            self.assertAlmostEqual(delta["loss_suppression_delta"], 0.2)
            self.assertAlmostEqual(delta["max_drawdown_delta"], -0.04)
            self.assertAlmostEqual(delta["signal_count_delta"], 2.0)

    def test_generates_markdown_report_from_registry_and_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = os.path.join(tmp, "registry.json")
            leaderboard_path = os.path.join(tmp, "leaderboard.csv")
            db_path = os.path.join(tmp, "strategy_library.sqlite3")
            out_dir = os.path.join(tmp, "docs")
            self._write_registry(registry_path)
            self._write_leaderboard(leaderboard_path)
            ingest_leaderboard(
                registry_path=registry_path,
                leaderboard_path=leaderboard_path,
                db_path=db_path,
                run_id="run-001",
                strategy_version="rules_langlang_enhanced_payoff_v1",
                data_snapshot_id="snapshot-a",
                artifact_paths={"leaderboard": leaderboard_path},
            )

            report_path = render_strategy_library_report(
                registry_path=registry_path,
                db_path=db_path,
                out_dir=out_dir,
            )
            with open(report_path, encoding="utf-8") as f:
                text = f.read()
            self.assertIn("# Strategy Library", text)
            self.assertIn("langlang-test-plus-01", text)
            self.assertIn("parent=`langlang-test-01`", text)
            self.assertIn("document_core", text)
            self.assertIn("langlang / native_payoff / enhanced_child", text)
            self.assertIn("big_loss_similarity_filter", text)
            self.assertIn("run-001", text)
            self.assertNotIn("TBD", text)
            self.assertTrue(os.path.exists(os.path.join(out_dir, "langlang.md")))

    def test_real_registry_contains_canonical_langlang_mainline_variants(self):
        library = StrategyLibrary.load(DEFAULT_REGISTRY_PATH)

        native = library.variant("langlang_01")
        plus = library.variant("langlang_plus_01")

        self.assertEqual(native.display_name, "langlang-01")
        self.assertEqual(native.lineage_group, "langlang_payoff")
        self.assertIsNone(native.parent_id)
        self.assertEqual(native.strategy_version, "rules_langlang_native_payoff_v1")
        self.assertIn("full_market_selection", native.core_logic)
        self.assertIn("six_document_entry_positions", native.core_logic)

        self.assertEqual(plus.display_name, "langlang-plus-01")
        self.assertEqual(plus.parent_id, "langlang_01")
        self.assertEqual(library.lineage("langlang_plus_01"), ["langlang_01", "langlang_plus_01"])
        self.assertEqual(plus.lineage_group, "langlang_payoff")
        self.assertEqual(plus.strategy_version, "rules_langlang_enhanced_payoff_v1")
        self.assertIn("big_loss_similarity_filter", plus.changed_factors)
        self.assertIn("right_tail_eligibility_filter", plus.changed_factors)

    def test_real_registry_contains_langlang_10bot_strategy_forest_variants(self):
        library = StrategyLibrary.load(DEFAULT_REGISTRY_PATH)
        expected = {
            "langlang_01": ("langlang-01", None),
            "langlang_plus_01": ("langlang-plus-01", "langlang_01"),
            "langlang_01_select": ("langlang-01-select", "langlang_01"),
            "langlang_01_entry": ("langlang-01-entry", "langlang_01"),
            "langlang_01_exit": ("langlang-01-exit", "langlang_01"),
            "langlang_01_risk": ("langlang-01-risk", "langlang_01"),
            "langlang_plus_01_select": ("langlang-plus-01-select", "langlang_plus_01"),
            "langlang_plus_01_entry": ("langlang-plus-01-entry", "langlang_plus_01"),
            "langlang_plus_01_exit": ("langlang-plus-01-exit", "langlang_plus_01"),
            "langlang_plus_01_loss": ("langlang-plus-01-loss", "langlang_plus_01"),
        }

        for variant_id, (display_name, parent_id) in expected.items():
            with self.subTest(variant_id=variant_id):
                variant = library.variant(variant_id)
                self.assertEqual(variant.display_name, display_name)
                self.assertEqual(variant.parent_id, parent_id)
                self.assertEqual(variant.lineage_group, "langlang_payoff")
                self.assertTrue(variant.core_logic)
                self.assertTrue(variant.iteration_notes)
                if parent_id is not None:
                    self.assertTrue(variant.changed_factors)

        self.assertEqual(library.lineage("langlang_plus_01_loss"), ["langlang_01", "langlang_plus_01", "langlang_plus_01_loss"])

    def test_optimizer_appends_strategy_library_ledger_without_changing_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = os.path.join(tmp, "registry.json")
            db_path = os.path.join(tmp, "strategy_library.sqlite3")
            trades_path = os.path.join(tmp, "trades.csv")
            kline_dir = os.path.join(tmp, "kline")
            out_dir = os.path.join(tmp, "out")
            os.makedirs(kline_dir, exist_ok=True)
            self._write_registry(registry_path)
            with open(trades_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "trade_id",
                        "symbol",
                        "side",
                        "entry_time",
                        "exit_time",
                        "entry_price",
                        "exit_price",
                        "leverage",
                        "margin",
                        "pnl_usdt",
                        "return_rate",
                        "hold_minutes",
                        "realized_move",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "trade_id": "1",
                        "symbol": "BTC-USDT-SWAP",
                        "side": "long",
                        "entry_time": "2024-01-02 00:00:00",
                        "exit_time": "2024-01-03 00:00:00",
                        "entry_price": "100",
                        "exit_price": "110",
                        "leverage": "1",
                        "margin": "100",
                        "pnl_usdt": "10",
                        "return_rate": "0.10",
                        "hold_minutes": "1440",
                        "realized_move": "0.10",
                    }
                )
            one_day_dir = os.path.join(kline_dir, "1D")
            os.makedirs(one_day_dir, exist_ok=True)
            start = datetime(2023, 10, 29, tzinfo=timezone.utc)
            with open(os.path.join(one_day_dir, "BTC-USDT-SWAP_unit.csv"), "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["ts", "open", "high", "low", "close", "volume"])
                writer.writeheader()
                for i in range(65):
                    ts = int((start + timedelta(days=i)).timestamp() * 1000)
                    writer.writerow(
                        {
                            "ts": str(ts),
                            "open": "100",
                            "high": "120",
                            "low": "90",
                            "close": str(100 + i),
                            "volume": "1000",
                        }
                    )

            result = HistoricalReplayOptimizer(
                OptimizerConfig(
                    trades_csv=trades_path,
                    kline_cache_dir=kline_dir,
                    out_dir=out_dir,
                    strategy_version="rules_v01",
                    variants=[StrategyVariant(variant_id="native_parent", ret_20d_min=0.01, ret_60d_min=0.01)],
                    min_validation_signals=0,
                    strategy_library_registry_path=registry_path,
                    strategy_library_db_path=db_path,
                    data_snapshot_id="unit-snapshot",
                )
            ).run()

            self.assertTrue(os.path.exists(result.leaderboard_path))
            conn = sqlite3.connect(db_path)
            rows = conn.execute("select strategy_version, variant_id, data_snapshot_id from strategy_runs").fetchall()
            conn.close()
            self.assertEqual(rows, [("rules_v01", "native_parent", "unit-snapshot")])


if __name__ == "__main__":
    unittest.main()

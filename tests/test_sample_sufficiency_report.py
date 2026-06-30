import json
import os
import sqlite3
import tempfile
import unittest

from langlang_trader.ledger import Ledger
from langlang_trader.sample_sufficiency import (
    summarize_sample_sufficiency,
    write_sample_sufficiency_reports,
)


def _insert_trade(
    conn,
    *,
    trade_id,
    run_id="run-a",
    bot_id="bot-a",
    variant_id="var-a",
    symbol="BTC-USDT-SWAP",
    side="long",
    status="closed",
    opened_at="2026-01-01T00:00:00+00:00",
    closed_at="2026-01-01T01:00:00+00:00",
    pnl=10.0,
    r_multiple=1.0,
    entry_reasons=None,
    entry_trace=None,
    exit_reasons=None,
    features=None,
    data_quality_flags=None,
):
    conn.execute(
        """
        insert into trade_lifecycle (
            trade_id, run_id, bot_id, variant_id, exchange, symbol, side, status,
            opened_at, closed_at, entry_price, exit_price, qty, leverage,
            initial_stop_loss, initial_risk_usdt, entry_fee, exit_fee, total_fees,
            gross_pnl_usdt, realized_pnl_usdt, mae_usdt, mfe_usdt, r_multiple,
            mfe_capture_ratio, entry_reason_codes_json, entry_reason_summary,
            entry_decision_trace_json, entry_feature_snapshot_json,
            exit_reason_codes_json, exit_reason_summary, exit_decision_trace_json,
            exit_feature_snapshot_json, data_quality_flags_json
        ) values (?, ?, ?, ?, 'binance', ?, ?, ?, ?, ?, 100, 110, 1, 3,
                  95, 5, 0.1, 0.2, 0.3, ?, ?, -1, 12, ?, 0.8,
                  ?, 'entry', ?, ?,
                  ?, 'exit', '{}', '{}', ?)
        """,
        (
            trade_id,
            run_id,
            bot_id,
            variant_id,
            symbol,
            side,
            status,
            opened_at,
            closed_at if status == "closed" else None,
            pnl if status == "closed" else None,
            pnl if status == "closed" else None,
            r_multiple if status == "closed" else None,
            json.dumps(entry_reasons or ["golden_pit_reclaim"]),
            json.dumps(entry_trace or {"entry_position_id": "1_startup_long"}),
            json.dumps(
                features
                if features is not None
                else {
                    "strong_pattern_tag": "golden_pit_reclaim",
                    "strong_pattern_score": 0.72,
                    "wyckoff_long_setup_tag": "spring_reclaim",
                    "wyckoff_long_score": 0.7,
                }
            ),
            json.dumps(exit_reasons or ["take_profit_reached"]),
            json.dumps(data_quality_flags or []),
        ),
    )


class SampleSufficiencyReportTest(unittest.TestCase):
    def test_marks_under_20_closed_trades_as_diagnostic_only_and_does_not_expand(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = os.path.join(tmp, "fleet.sqlite3")
            Ledger(ledger_path)
            conn = sqlite3.connect(ledger_path)
            for idx in range(3):
                _insert_trade(conn, trade_id=f"t{idx}", variant_id="var-under-20")
            conn.commit()
            conn.close()

            summary = summarize_sample_sufficiency(
                [{"fleet_id": "fleet-a", "ledger_path": ledger_path, "run_id": "run-a"}]
            )

            row = summary["variant_sample_sufficiency"]["variants"][0]
            self.assertEqual(row["closed_trades"], 3)
            self.assertEqual(row["sample_status"], "diagnostic_only")
            self.assertEqual(row["variant_action"], "do_not_expand")

    def test_marks_30_to_50_closed_trades_as_early_elimination_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = os.path.join(tmp, "fleet.sqlite3")
            Ledger(ledger_path)
            conn = sqlite3.connect(ledger_path)
            for idx in range(35):
                _insert_trade(conn, trade_id=f"t{idx}", variant_id="var-35", symbol=f"S{idx}-USDT-SWAP")
            conn.commit()
            conn.close()

            summary = summarize_sample_sufficiency(
                [{"fleet_id": "fleet-a", "ledger_path": ledger_path, "run_id": "run-a"}]
            )

            row = summary["variant_sample_sufficiency"]["variants"][0]
            self.assertEqual(row["sample_status"], "early_elimination_reference")
            self.assertEqual(row["variant_action"], "early_elimination_only")

    def test_correlated_same_symbol_side_time_trades_count_as_one_independent_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = os.path.join(tmp, "fleet.sqlite3")
            Ledger(ledger_path)
            conn = sqlite3.connect(ledger_path)
            for idx, variant in enumerate(["var-a", "var-b", "var-c"]):
                _insert_trade(
                    conn,
                    trade_id=f"cluster-{idx}",
                    bot_id=f"bot-{idx}",
                    variant_id=variant,
                    symbol="ACT-USDT-SWAP",
                    side="short",
                    opened_at="2026-01-01T00:00:04+00:00",
                    closed_at="2026-01-01T01:00:06+00:00",
                )
            conn.commit()
            conn.close()

            summary = summarize_sample_sufficiency(
                [{"fleet_id": "fleet-a", "ledger_path": ledger_path, "run_id": "run-a"}]
            )

            clusters = summary["variant_sample_sufficiency"]["correlated_trade_clusters"]
            self.assertEqual(len(clusters), 1)
            self.assertEqual(clusters[0]["trade_count"], 3)
            self.assertEqual(clusters[0]["independent_group_count"], 1)

    def test_long_filter_diagnosis_counts_long_skip_reasons_without_strategy_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = os.path.join(tmp, "fleet.sqlite3")
            Ledger(ledger_path)
            conn = sqlite3.connect(ledger_path)
            conn.execute(
                """
                insert into risk_events (
                    run_id, bot_id, variant_id, exchange, created_at, reason, payload_json
                ) values (
                    'run-a', 'bot-a', 'var-a', 'multi', '2026-01-01T00:00:00+00:00',
                    'intent_rejected',
                    '{"requested_side":"long","risk_rejection_reason":"upside_space_insufficient","filter_codes":["historical_support_missing","wyckoff_risk"]}'
                )
                """
            )
            conn.commit()
            conn.close()

            summary = summarize_sample_sufficiency(
                [{"fleet_id": "fleet-a", "ledger_path": ledger_path, "run_id": "run-a"}]
            )

            diagnosis = summary["long_filter_diagnosis"]
            self.assertEqual(diagnosis["reason_counts"]["upside_space_insufficient"], 1)
            self.assertEqual(diagnosis["filter_code_counts"]["historical_support_missing"], 1)
            self.assertEqual(diagnosis["filter_code_counts"]["wyckoff_risk"], 1)

    def test_long_filter_diagnosis_reads_symbol_selection_ranked_long_payloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = os.path.join(tmp, "fleet.sqlite3")
            Ledger(ledger_path)
            conn = sqlite3.connect(ledger_path)
            conn.execute(
                """
                insert into risk_events (
                    run_id, bot_id, variant_id, exchange, created_at, reason, payload_json
                ) values (
                    'run-a', 'fleet', 'fleet', 'multi', '2026-01-01T00:00:00+00:00',
                    'symbol_selection',
                    '{"profiles":{"enhanced":{"ranked_long":[
                        {"symbol":"A-USDT-SWAP","selection_bias":"long","selected":true,"filter_codes":["not_leader_catch_up"],"reason_codes":["spoon_bottom_confirmed"]},
                        {"symbol":"B-USDT-SWAP","selection_bias":"long","selected":false,"filter_codes":["five_wave_late_risk","false_breakout_risk"],"reason_codes":["relative_to_btc_strength"]}
                    ]}}}'
                )
                """
            )
            conn.commit()
            conn.close()

            summary = summarize_sample_sufficiency(
                [{"fleet_id": "fleet-a", "ledger_path": ledger_path, "run_id": "run-a"}]
            )

            diagnosis = summary["long_filter_diagnosis"]
            self.assertEqual(diagnosis["selection_long_candidates"], 2)
            self.assertEqual(diagnosis["selection_long_selected"], 1)
            self.assertEqual(diagnosis["filter_code_counts"]["not_leader_catch_up"], 1)
            self.assertEqual(diagnosis["filter_code_counts"]["five_wave_late_risk"], 1)

    def test_writes_fixed_report_files_and_preserves_data_quality_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = os.path.join(tmp, "fleet.sqlite3")
            Ledger(ledger_path)
            conn = sqlite3.connect(ledger_path)
            _insert_trade(
                conn,
                trade_id="t-quality",
                variant_id="var-a",
                data_quality_flags=["missing_feature_snapshot"],
                features={},
            )
            conn.commit()
            conn.close()

            paths = write_sample_sufficiency_reports(
                ledgers=[{"fleet_id": "fleet-a", "ledger_path": ledger_path, "run_id": "run-a"}],
                out_dir=tmp,
            )

            for filename in (
                "variant_sample_sufficiency.json",
                "variant_attribution_report.json",
                "long_filter_diagnosis.json",
                "exit_management_attribution.json",
                "sample_sufficiency_summary.md",
            ):
                self.assertTrue(os.path.exists(os.path.join(tmp, filename)))
                self.assertEqual(paths[filename], os.path.join(tmp, filename))
            with open(os.path.join(tmp, "variant_attribution_report.json"), encoding="utf-8") as handle:
                attribution = json.load(handle)
            self.assertEqual(attribution["data_quality_flags"]["missing_feature_snapshot"], 1)

    def test_reports_strategy_tree_trace_buckets_for_orthogonal_variants(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = os.path.join(tmp, "fleet.sqlite3")
            Ledger(ledger_path)
            conn = sqlite3.connect(ledger_path)
            _insert_trade(
                conn,
                trade_id="t-orthogonal",
                bot_id="orthogonal-bot",
                variant_id="orthogonal_v1_low_position_wyckoff_long_a",
                entry_trace={
                    "entry_position_id": "1_low_position_wyckoff_spring_long",
                    "experiment_family": "orthogonal_v1",
                    "entry_family": "low_position_wyckoff_long",
                    "strategy_tree_variant_id": "orthogonal_v1_low_position_wyckoff_long_a",
                    "strategy_tree_path": [
                        "langlang_01",
                        "langlang_plus_01",
                        "langlang_plus_01_loss",
                        "orthogonal_v1_low_position_wyckoff_long_a",
                    ],
                },
            )
            conn.commit()
            conn.close()

            summary = summarize_sample_sufficiency(
                [{"fleet_id": "fleet-a", "ledger_path": ledger_path, "run_id": "run-a"}]
            )

            row = summary["variant_sample_sufficiency"]["variants"][0]
            self.assertEqual(row["experiment_family"], "orthogonal_v1")
            self.assertEqual(row["entry_family"], "low_position_wyckoff_long")
            self.assertEqual(row["strategy_tree_variant_id"], "orthogonal_v1_low_position_wyckoff_long_a")
            attribution = summary["variant_attribution_report"]
            self.assertEqual(attribution["experiment_family_buckets"]["orthogonal_v1"], 1)
            self.assertEqual(attribution["entry_family_buckets"]["low_position_wyckoff_long"], 1)
            self.assertEqual(
                attribution["strategy_tree_variant_buckets"]["orthogonal_v1_low_position_wyckoff_long_a"],
                1,
            )


if __name__ == "__main__":
    unittest.main()

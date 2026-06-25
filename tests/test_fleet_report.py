import json
import os
import sqlite3
import tempfile
import unittest

from langlang_trader.fleet_report import summarize_fleet_ledger, write_fleet_report
from langlang_trader.ledger import Ledger


class FleetReportTest(unittest.TestCase):
    def test_summarizes_latest_multi_equity_without_double_counting_exchange_snapshots(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "fleet.sqlite3")
            Ledger(path)
            conn = sqlite3.connect(path)
            conn.executescript(
                """
                insert into equity_snapshots (
                    run_id, bot_id, variant_id, exchange, created_at, equity_usdt, cash_usdt,
                    margin_used_usdt, realized_pnl_usdt, raw_json
                ) values
                ('run-clean', 'bot-a', 'var-a', 'okx', '2026-01-01T00:00:00+00:00', 10050, 9990, 100, 50, '{}'),
                ('run-clean', 'bot-a', 'var-a', 'binance', '2026-01-01T00:00:00+00:00', 10020, 9980, 100, 20, '{}'),
                ('run-clean', 'bot-a', 'var-a', 'multi', '2026-01-01T00:01:00+00:00', 9975, 9950, 200, -25, '{}'),
                ('run-clean', 'bot-b', 'var-b', 'multi', '2026-01-01T00:01:00+00:00', 10000, 10000, 0, 0, '{}');
                insert into orders (
                    run_id, bot_id, variant_id, exchange, created_at, symbol, side, order_type,
                    qty, leverage, reduce_only, status, raw_payload_json
                ) values
                ('run-clean', 'bot-a', 'var-a', 'multi', '2026-01-01T00:02:00+00:00', 'BTC-USDT-SWAP', 'long', 'market', 1, 5, 0, 'filled', '{}'),
                ('run-clean', 'bot-a', 'var-a', 'multi', '2026-01-01T00:03:00+00:00', 'BTC-USDT-SWAP', 'short', 'market', 1, 5, 1, 'filled', '{}');
                insert into fills (
                    run_id, bot_id, variant_id, exchange, created_at, order_id, exchange_order_id, symbol,
                    side, qty, price, fee, liquidity, raw_payload_json
                ) values
                ('run-clean', 'bot-a', 'var-a', 'multi', '2026-01-01T00:02:00+00:00', 1, 'a', 'BTC-USDT-SWAP', 'long', 1, 100, 0.5, 'taker', '{}');
                insert into positions (
                    run_id, bot_id, variant_id, exchange, symbol, side, qty, avg_price, leverage, updated_at
                ) values
                ('run-clean', 'bot-a', 'var-a', 'multi', 'ETH-USDT-SWAP', 'short', 2, 50, 5, '2026-01-01T00:04:00+00:00');
                insert into signals (
                    run_id, bot_id, variant_id, created_at, strategy_version, symbol, side, strength,
                    reason_codes_json, features_json, invalidation_price
                ) values
                ('run-clean', 'bot-a', 'var-a', '2026-01-01T00:01:00+00:00', 'rules', 'BTC-USDT-SWAP', 'long', 1.0, '[]', '{}', 90);
                insert into risk_events (
                    run_id, bot_id, variant_id, exchange, created_at, reason, payload_json
                ) values
                ('run-clean', 'bot-a', 'var-a', 'multi', '2026-01-01T00:05:00+00:00', 'intent_rejected', '{"risk_rejection_reason":"max_open_positions"}');
                """
            )
            conn.commit()
            conn.close()

            summary = summarize_fleet_ledger(path, run_id="run-clean", initial_equity_usdt=10_000)

            bot_a = next(row for row in summary["bots"] if row["bot_id"] == "bot-a")
            self.assertEqual(bot_a["equity_pnl_net"], -25.0)
            self.assertEqual(bot_a["opened_orders"], 1)
            self.assertEqual(bot_a["closed_orders"], 1)
            self.assertEqual(bot_a["fills"], 1)
            self.assertEqual(bot_a["positions"], 1)
            self.assertEqual(summary["risk_rejections"]["max_open_positions"], 1)
            self.assertEqual(summary["trade_journal"]["legacy_unjournaled_fills"], 1)
            self.assertIn("legacy_fills_without_trade_lifecycle", summary["trade_journal"]["journal_data_quality"])

    def test_reconstructs_multi_equity_from_newer_exchange_snapshots_after_fills(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "fleet.sqlite3")
            Ledger(path)
            conn = sqlite3.connect(path)
            conn.executescript(
                """
                insert into equity_snapshots (
                    run_id, bot_id, variant_id, exchange, created_at, equity_usdt, cash_usdt,
                    margin_used_usdt, realized_pnl_usdt, raw_json
                ) values
                ('run-clean', 'bot-a', 'var-a', 'multi', '2026-01-01T00:00:00+00:00', 10000, 10000, 0, 0, '{}'),
                ('run-clean', 'bot-a', 'var-a', 'okx', '2026-01-01T00:01:00+00:00', 9990, 9995, 100, -5, '{}'),
                ('run-clean', 'bot-a', 'var-a', 'binance', '2026-01-01T00:02:00+00:00', 9980, 9985, 200, -15, '{}');
                """
            )
            conn.commit()
            conn.close()

            summary = summarize_fleet_ledger(path, run_id="run-clean", initial_equity_usdt=10_000)

            bot_a = next(row for row in summary["bots"] if row["bot_id"] == "bot-a")
            self.assertEqual(bot_a["equity_usdt"], 9970.0)
            self.assertEqual(bot_a["equity_pnl_net"], -30.0)
            self.assertEqual(bot_a["margin_used"], 300.0)
            self.assertEqual(bot_a["latest_snapshot_source"], "reconstructed_exchange")

    def test_writes_markdown_and_json_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = os.path.join(tmp, "fleet.sqlite3")
            Ledger(ledger_path)
            out_dir = os.path.join(tmp, "reports")

            result = write_fleet_report(ledger_path=ledger_path, run_id="empty-run", out_dir=out_dir)

            self.assertTrue(os.path.exists(result["markdown_path"]))
            self.assertTrue(os.path.exists(result["json_path"]))
            with open(result["json_path"], encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(payload["run_id"], "empty-run")

    def test_summarizes_trade_journal_and_attribution_buckets(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "fleet.sqlite3")
            Ledger(path)
            conn = sqlite3.connect(path)
            conn.executescript(
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
                ) values
                (
                    't1', 'run-journal', 'bot-a', 'var-a', 'binance', 'BTC-USDT-SWAP', 'long', 'closed',
                    '2026-01-01T00:00:00+00:00', '2026-01-01T04:00:00+00:00',
                    100, 108, 1, 3, 95, 5, 0.1, 0.1, 0.2, 8, 7.8, -2, 10, 1.56, 0.8,
                    '["golden_pit_reclaim","wyckoff_spring_reclaim"]', '1_startup_long',
                    '{"entry_position_id":"1_startup_long"}',
                    '{"strong_pattern_tag":"golden_pit_reclaim","wyckoff_long_setup_tag":"spring_reclaim"}',
                    '["take_profit_reached"]', 'take_profit_reached', '{}',
                    '{"wyckoff_exit_tag":"none"}', '[]'
                ),
                (
                    't2', 'run-journal', 'bot-b', 'var-b', 'binance', 'ETH-USDT-SWAP', 'short', 'open',
                    '2026-01-01T01:00:00+00:00', null,
                    50, null, 2, 3, 55, 10, 0.1, 0, 0.1, null, null, -1, 3, null, null,
                    '["wyckoff_sow_breakdown"]', 'top_short',
                    '{"entry_position_id":"top_short"}',
                    '{"strong_pattern_tag":"none","wyckoff_short_setup_tag":"sow_breakdown"}',
                    '[]', null, '{}', '{}', '["missing_exit"]'
                );
                """
            )
            conn.commit()
            conn.close()

            summary = summarize_fleet_ledger(path, run_id="run-journal", initial_equity_usdt=10_000)

            journal = summary["trade_journal"]
            self.assertEqual(journal["total_trades"], 2)
            self.assertEqual(journal["closed_trades"], 1)
            self.assertEqual(journal["open_trades"], 1)
            self.assertEqual(journal["entry_reason_buckets"]["golden_pit_reclaim"], 1)
            self.assertEqual(journal["wyckoff_setup_buckets"]["spring_reclaim"], 1)
            self.assertEqual(journal["exit_reason_buckets"]["take_profit_reached"], 1)
            self.assertEqual(journal["data_quality_flags"]["missing_exit"], 1)

    def test_trade_journal_reports_legacy_fills_even_after_schema_is_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "fleet.sqlite3")
            Ledger(path)
            conn = sqlite3.connect(path)
            conn.executescript(
                """
                insert into fills (
                    id, run_id, bot_id, variant_id, exchange, created_at, order_id, exchange_order_id, symbol,
                    side, qty, price, fee, liquidity, raw_payload_json
                ) values
                (10, 'mixed-run', 'bot-a', 'var-a', 'binance', '2026-01-01T00:00:00+00:00', 1, 'legacy',
                 'BTC-USDT-SWAP', 'long', 1, 100, 0.1, 'taker', '{}'),
                (11, 'mixed-run', 'bot-a', 'var-a', 'binance', '2026-01-01T01:00:00+00:00', 2, 'journaled',
                 'ETH-USDT-SWAP', 'long', 1, 50, 0.1, 'taker', '{}');
                insert into orders (
                    id, run_id, bot_id, variant_id, exchange, created_at, symbol, side, order_type,
                    qty, leverage, reduce_only, status, raw_payload_json
                ) values
                (1, 'mixed-run', 'bot-a', 'var-a', 'binance', '2026-01-01T00:00:00+00:00',
                 'BTC-USDT-SWAP', 'short', 'market', 1, 3, 1, 'filled', '{}'),
                (2, 'mixed-run', 'bot-a', 'var-a', 'binance', '2026-01-01T01:00:00+00:00',
                 'ETH-USDT-SWAP', 'long', 'market', 1, 3, 0, 'filled', '{}');
                insert into trade_lifecycle (
                    trade_id, run_id, bot_id, variant_id, exchange, symbol, side, status,
                    opened_at, entry_order_id, entry_fill_id, entry_price, qty, leverage,
                    entry_reason_codes_json, entry_reason_summary, entry_decision_trace_json,
                    entry_feature_snapshot_json, exit_reason_codes_json, exit_decision_trace_json,
                    exit_feature_snapshot_json, data_quality_flags_json
                ) values
                ('t-journaled', 'mixed-run', 'bot-a', 'var-a', 'binance', 'ETH-USDT-SWAP', 'long', 'open',
                 '2026-01-01T01:00:00+00:00', 2, 11, 50, 1, 3,
                 '["leader_platform_start"]', '1_startup_long', '{}', '{}', '[]', '{}', '{}', '[]');
                insert into trade_events (
                    trade_id, run_id, bot_id, variant_id, exchange, created_at, event_type, symbol,
                    fill_id, reason_codes_json, decision_trace_json, feature_snapshot_json,
                    data_quality_flags_json, raw_payload_json
                ) values
                ('t-journaled', 'mixed-run', 'bot-a', 'var-a', 'binance', '2026-01-01T01:00:00+00:00',
                 'entry_fill', 'ETH-USDT-SWAP', 11, '["leader_platform_start"]', '{}', '{}', '[]', '{}');
                """
            )
            conn.commit()
            conn.close()

            summary = summarize_fleet_ledger(path, run_id="mixed-run", initial_equity_usdt=10_000)

            journal = summary["trade_journal"]
            self.assertEqual(journal["total_trades"], 1)
            self.assertEqual(journal["legacy_unjournaled_fills"], 1)
            self.assertEqual(journal["legacy_unjournaled_closed_orders"], 1)
            self.assertIn("legacy_fills_without_trade_lifecycle", journal["journal_data_quality"])


if __name__ == "__main__":
    unittest.main()

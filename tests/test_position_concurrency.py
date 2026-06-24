import csv
import tempfile
import unittest
from pathlib import Path

from langlang_trader.position_concurrency import analyze_trade_concurrency, build_concurrency_report


class PositionConcurrencyTest(unittest.TestCase):
    def test_analyzes_entry_time_concurrency_and_repairs_bad_or_zero_duration_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            trades = Path(tmp) / "trades.csv"
            with trades.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["trade_id", "symbol", "entry_time", "exit_time", "hold_minutes", "return_rate"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "trade_id": "1",
                        "symbol": "BTC-USDT-SWAP",
                        "entry_time": "2024-01-01 00:00:00",
                        "exit_time": "2024-01-01 01:00:00",
                        "hold_minutes": "60",
                        "return_rate": "0.10",
                    }
                )
                writer.writerow(
                    {
                        "trade_id": "2",
                        "symbol": "ETH-USDT-SWAP",
                        "entry_time": "2024-01-01 00:10:00",
                        "exit_time": "2024-01-01 00:10:00",
                        "hold_minutes": "0",
                        "return_rate": "-0.01",
                    }
                )
                writer.writerow(
                    {
                        "trade_id": "3",
                        "symbol": "SOL-USDT-SWAP",
                        "entry_time": "2024-01-01 00:10:00",
                        "exit_time": "2024-01-01 00:05:00",
                        "hold_minutes": "30",
                        "return_rate": "0.50",
                    }
                )

            report = analyze_trade_concurrency(trades)

            self.assertEqual(report["trade_count"], 3)
            self.assertEqual(report["time_repair_counts"]["zero_duration_min_1m"], 1)
            self.assertEqual(report["time_repair_counts"]["exit_rebuilt_from_hold_minutes"], 1)
            self.assertEqual(report["entry_concurrency"]["max_open_positions"], 3)
            self.assertEqual(report["entry_concurrency"]["max_open_symbols"], 3)
            self.assertEqual(report["recommended_risk"]["max_open_positions"], 3)
            self.assertEqual(report["recommended_risk"]["max_open_symbols"], 3)

    def test_build_concurrency_report_writes_machine_and_human_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trades = root / "trades.csv"
            out = root / "out"
            trades.write_text(
                "trade_id,symbol,entry_time,exit_time,hold_minutes,return_rate\n"
                "1,BTC-USDT-SWAP,2024-01-01 00:00:00,2024-01-01 01:00:00,60,0.1\n",
                encoding="utf-8",
            )

            result = build_concurrency_report(trades_csv=trades, out_dir=out)

            self.assertTrue(Path(result["summary_json"]).exists())
            self.assertTrue(Path(result["entry_snapshot_csv"]).exists())
            self.assertTrue(Path(result["report_md"]).exists())
            self.assertIn("max_open_positions", Path(result["report_md"]).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

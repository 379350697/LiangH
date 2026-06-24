import csv
import json
import os
import tempfile
import unittest

from langlang_trader.optimize import OptimizerConfig, HistoricalReplayOptimizer
from langlang_trader.strategy import StrategyVariant


def write_trades(path):
    rows = [
        {
            "trade_id": "1",
            "entry_time": "2024-01-20 00:00:00",
            "exit_time": "2024-01-20 01:00:00",
            "symbol": "BTC-USDT-SWAP",
            "side": "long",
            "pnl_usdt": "300",
            "return_rate": "0.30",
        },
        {
            "trade_id": "2",
            "entry_time": "2024-02-20 00:00:00",
            "exit_time": "2024-02-20 01:00:00",
            "symbol": "BTC-USDT-SWAP",
            "side": "long",
            "pnl_usdt": "-200",
            "return_rate": "-0.20",
        },
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def write_daily_cache(cache_dir):
    one_d = os.path.join(cache_dir, "1D")
    os.makedirs(one_d, exist_ok=True)
    path = os.path.join(one_d, "BTC-USDT-SWAP_test.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ts", "open", "high", "low", "close", "vol", "vol_ccy", "vol_quote", "confirm"])
        start = 1_701_580_800_000
        for idx in range(130):
            close = 100.0 * (1 + idx * 0.012)
            writer.writerow([start + idx * 86_400_000, close * 0.99, close * 1.02, close * 0.98, close, 1000, 100, 100000, 1])


class OptimizerTest(unittest.TestCase):
    def test_optimizer_writes_leaderboard_and_selected_fleet_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            trades = os.path.join(tmp, "standard_trades.csv")
            cache = os.path.join(tmp, "kline_cache")
            out = os.path.join(tmp, "out")
            write_trades(trades)
            write_daily_cache(cache)

            optimizer = HistoricalReplayOptimizer(
                OptimizerConfig(
                    trades_csv=trades,
                    kline_cache_dir=cache,
                    out_dir=out,
                    variants=[
                        StrategyVariant("loose", 0.12, 0.32, 0.45, 0.18, 0.005),
                        StrategyVariant("strict", 0.30, 0.60, 0.80, 0.04, 0.001),
                    ],
                    top_n=1,
                    min_validation_signals=1,
                    max_validation_signals=300,
                )
            )

            result = optimizer.run()

            self.assertTrue(os.path.exists(os.path.join(out, "leaderboard.csv")))
            self.assertTrue(os.path.exists(os.path.join(out, "selected_fleet_config.json")))
            self.assertTrue(os.path.exists(os.path.join(out, "optimizer_report.md")))
            self.assertGreaterEqual(len(result.leaderboard), 1)
            with open(os.path.join(out, "selected_fleet_config.json"), encoding="utf-8") as f:
                config = json.load(f)
            self.assertEqual(config["execution"]["executor"], "paper_okx")
            self.assertTrue(config["selection"]["enabled"])
            self.assertEqual(config["selection"]["top_n"], 20)
            self.assertEqual(config["bots"][0]["variant"]["variant_id"], result.leaderboard[0]["variant_id"])


if __name__ == "__main__":
    unittest.main()

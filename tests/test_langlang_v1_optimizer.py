import csv
import json
import os
import tempfile
import unittest

from langlang_trader.optimize import HistoricalReplayOptimizer, OptimizerConfig
from langlang_trader.strategy import LangLangV1Variant


def write_trades(path):
    rows = [
        {
            "trade_id": "long-win",
            "entry_time": "2024-02-15 00:00:00",
            "exit_time": "2024-02-16 00:00:00",
            "symbol": "BTC-USDT-SWAP",
            "side": "long",
            "pnl_usdt": "500",
            "return_rate": "0.50",
        },
        {
            "trade_id": "short-win",
            "entry_time": "2024-04-20 00:00:00",
            "exit_time": "2024-04-21 00:00:00",
            "symbol": "ETH-USDT-SWAP",
            "side": "short",
            "pnl_usdt": "400",
            "return_rate": "0.40",
        },
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def write_daily_cache(cache_dir, symbol, up=True):
    one_d = os.path.join(cache_dir, "1D")
    os.makedirs(one_d, exist_ok=True)
    path = os.path.join(one_d, f"{symbol}_test.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ts", "open", "high", "low", "close", "vol", "vol_ccy", "vol_quote", "confirm"])
        start = 1_701_580_800_000
        for idx in range(150):
            drift = idx * 0.012 if up else -idx * 0.006
            close = max(10.0, 100.0 * (1 + drift))
            writer.writerow([start + idx * 86_400_000, close * 0.99, close * 1.02, close * 0.98, close, 1000 + idx, 100, 100000, 1])


class LangLangV1OptimizerTest(unittest.TestCase):
    def test_v1_optimizer_uses_event_replay_and_writes_v1_fleet_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            trades = os.path.join(tmp, "standard_trades.csv")
            cache = os.path.join(tmp, "kline_cache")
            out = os.path.join(tmp, "out")
            write_trades(trades)
            write_daily_cache(cache, "BTC-USDT-SWAP", up=True)
            write_daily_cache(cache, "ETH-USDT-SWAP", up=False)

            result = HistoricalReplayOptimizer(
                OptimizerConfig(
                    trades_csv=trades,
                    kline_cache_dir=cache,
                    out_dir=out,
                    strategy_version="rules_langlang_v1",
                    variants=[LangLangV1Variant(variant_id="v1-loose")],
                    top_n=1,
                    min_validation_signals=1,
                    max_validation_signals=300,
                )
            ).run()

            self.assertGreaterEqual(result.leaderboard[0]["validation_signals"], 1)
            self.assertGreaterEqual(result.leaderboard[0]["raw_validation_signals"], result.leaderboard[0]["validation_signals"])
            self.assertIn("validation_realized_pnl_usdt", result.leaderboard[0])
            with open(os.path.join(out, "selected_fleet_config.json"), encoding="utf-8") as f:
                config = json.load(f)
            self.assertEqual(config["strategy_version"], "rules_langlang_v1")
            self.assertEqual(config["bots"][0]["variant"]["variant_id"], "v1-loose")
            with open(os.path.join(out, "optimizer_report.md"), encoding="utf-8") as f:
                self.assertIn("event_replay", f.read())


if __name__ == "__main__":
    unittest.main()

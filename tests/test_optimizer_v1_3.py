import csv
import json
import os
import tempfile
import unittest

from langlang_trader.optimize import HistoricalReplayOptimizer, OptimizerConfig
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


class OptimizerV13Test(unittest.TestCase):
    def test_v1_3_optimizer_writes_final_strategy_multi_exchange_fleet_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            trades = os.path.join(tmp, "standard_trades.csv")
            cache = os.path.join(tmp, "kline_cache")
            out = os.path.join(tmp, "out")
            write_trades(trades)
            write_cache(cache, "SOL-USDT-SWAP")

            result = HistoricalReplayOptimizer(
                OptimizerConfig(
                    trades_csv=trades,
                    kline_cache_dir=cache,
                    out_dir=out,
                    strategy_version="rules_langlang_v1_3",
                    variants=[
                        LangLangV1_3Variant(
                            variant_id="llv1_3_long_smoke",
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
            ).run()

            self.assertTrue(os.path.exists(os.path.join(out, "leaderboard_v1_3.csv")))
            self.assertTrue(os.path.exists(os.path.join(out, "selected_fleet_config_v1_3.json")))
            self.assertTrue(os.path.exists(os.path.join(out, "optimizer_report_v1_3.md")))
            self.assertEqual(result.selected_config_path, os.path.join(out, "selected_fleet_config_v1_3.json"))
            with open(os.path.join(out, "selected_fleet_config_v1_3.json"), encoding="utf-8") as f:
                config = json.load(f)
            self.assertEqual(config["strategy_version"], "rules_langlang_v1_3")
            self.assertEqual(config["execution"]["exchange"], "multi")
            self.assertEqual(config["execution"]["executor"], "paper_multi")
            self.assertEqual(config["universe"]["mode"], "okx_binance_usdt_swap_observe")
            self.assertEqual(config["selection"]["style"], "dual_board")


if __name__ == "__main__":
    unittest.main()

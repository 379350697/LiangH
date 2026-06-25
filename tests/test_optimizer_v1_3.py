import csv
import json
import os
import tempfile
import unittest

from langlang_trader.optimize import HistoricalReplayOptimizer, OptimizerConfig, _rank_rows
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
    def test_v1_3_optimizer_prefers_nonzero_signal_variants_for_selection(self):
        rows = _rank_rows(
            [
                {
                    "variant_id": "zero_signal_pretty",
                    "eligible": True,
                    "validation_signals": 0,
                    "raw_validation_signals": 0,
                    "validation_net_pnl": 0.0,
                    "validation_profit_factor": 0.0,
                    "max_drawdown": 0.0,
                    "big_win_recall": 0.0,
                    "big_loss_overlap": 0.0,
                    "validation_realized_pnl_usdt": 0.0,
                    "right_tail_capture_score": 1.0,
                    "right_tail_return_capture": 1.0,
                    "loss_suppression_score": 1.0,
                    "payoff_asymmetry_score": 1.0,
                    "avg_win_loss_ratio": 0.0,
                    "max_single_loss": 0.0,
                    "loss_cap_score": 1.0,
                    "excel_event_support_score": 0.0,
                },
                {
                    "variant_id": "nonzero_signal_real",
                    "eligible": True,
                    "validation_signals": 3,
                    "raw_validation_signals": 3,
                    "validation_net_pnl": 0.01,
                    "validation_profit_factor": 1.1,
                    "max_drawdown": 0.02,
                    "big_win_recall": 0.0,
                    "big_loss_overlap": 0.0,
                    "validation_realized_pnl_usdt": 100.0,
                    "right_tail_capture_score": 0.0,
                    "right_tail_return_capture": 0.0,
                    "loss_suppression_score": 0.5,
                    "payoff_asymmetry_score": 0.5,
                    "avg_win_loss_ratio": 1.0,
                    "max_single_loss": -0.01,
                    "loss_cap_score": 0.5,
                    "excel_event_support_score": 0.0,
                },
            ]
        )

        self.assertEqual(rows[0]["variant_id"], "nonzero_signal_real")
        self.assertEqual(rows[0]["rank"], 1)

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
            self.assertEqual(config["universe"]["snapshot_path"], os.path.join(out, "universe_snapshot.json"))
            self.assertEqual(config["selection"]["style"], "dual_board")


if __name__ == "__main__":
    unittest.main()

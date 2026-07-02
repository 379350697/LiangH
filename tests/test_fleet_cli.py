from collections import Counter
import json
from pathlib import Path
import plistlib
import unittest

from langlang_trader.config import MarketDataConfig
from langlang_trader.fleet import FleetConfig, load_fleet_config
from langlang_trader.fleet_cli import config_with_symbol_override, market_data_for_config
from langlang_trader.hft_scalping import load_hft_scalp_fleet_config
from langlang_trader.market_data import BinanceRestMarketData
from liangh_trader.market_maker.config import load_market_maker_config


class FleetCliTest(unittest.TestCase):
    def test_symbol_override_limits_startup_universe_without_touching_other_config(self):
        config = FleetConfig(
            run_id="fleet-test",
            market_data=MarketDataConfig(symbols=["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP"]),
            strategy_version="rules_langlang_v1_1",
        )

        limited = config_with_symbol_override(config, "BTC-USDT-SWAP, ETH-USDT-SWAP")

        self.assertEqual(limited.market_data.symbols, ["BTC-USDT-SWAP", "ETH-USDT-SWAP"])
        self.assertEqual(limited.strategy_version, "rules_langlang_v1_1")
        self.assertEqual(limited.run_id, "fleet-test")

    def test_clean_ten_bot_config_uses_shared_public_market_cache(self):
        config = load_fleet_config("configs/fleet/selected_fleet_config_langlang_10bot_clean.json")

        self.assertTrue(config.market_data.cache_enabled)
        self.assertTrue(config.market_data.market_metrics_cache_enabled)
        self.assertEqual(config.market_data.cache_dir, "output/fleet/shared_public_market_cache/kline_cache")
        self.assertEqual(config.market_data.market_metrics_cache_dir, "output/fleet/shared_public_market_cache/market_metrics")
        self.assertEqual(config.market_data.market_snapshot_cache_dir, "output/fleet/langlang_strategy_forest/clean/market_snapshots")

    def test_orthogonal_v1_config_runs_as_independent_paper_fleet_with_trace_fields(self):
        config = load_fleet_config("configs/fleet/selected_fleet_config_orthogonal_v1.json")

        self.assertEqual(config.run_id, "langlang-paper-orthogonal-v1")
        self.assertEqual(config.strategy_version, "rules_langlang_v1_3")
        self.assertEqual(config.execution.executor, "paper_multi")
        self.assertFalse(config.execution.allow_live_orders)
        self.assertIsNone(config.risk.max_daily_loss_usdt)
        self.assertEqual(config.risk.max_open_positions, 2)
        self.assertEqual(config.risk.max_open_symbols, 2)
        self.assertEqual(len(config.bots), 8)
        self.assertTrue(all(bot.variant.experiment_family == "orthogonal_v1" for bot in config.bots))
        self.assertEqual(
            sorted({bot.variant.entry_family for bot in config.bots}),
            [
                "failed_breakdown_reclaim_long",
                "low_position_wyckoff_long",
                "payoff_probe",
                "retest_confirmed_short",
            ],
        )
        self.assertEqual(len({bot.variant.variant_id for bot in config.bots}), 8)

    def test_five_bar_scalp_paper_config_registers_18_symbol_period_bots(self):
        config = load_fleet_config("configs/fleet/five_bar_scalp_18bot_paper.json")

        self.assertEqual(config.run_id, "five-bar-scalp-18bot-paper-v1")
        self.assertEqual(config.strategy_version, "five_bar_fractal_scalp_v1")
        self.assertEqual(config.execution.exchange, "binance")
        self.assertEqual(config.execution.executor, "paper_binance")
        self.assertEqual(config.universe.provider, "binance")
        self.assertFalse(config.execution.allow_live_orders)
        self.assertEqual(len(config.bots), 18)
        self.assertEqual(config.risk.max_open_positions, 18)
        self.assertEqual(config.risk.max_open_symbols, 6)
        self.assertEqual(set(config.market_data.bars), {"1s", "5s", "15s", "1m", "3m", "5m"})
        self.assertEqual(
            sorted({bot.variant.symbol for bot in config.bots}),
            [
                "BNB-USDT-SWAP",
                "BTC-USDT-SWAP",
                "DOGE-USDT-SWAP",
                "ETH-USDT-SWAP",
                "HYPE-USDT-SWAP",
                "XRP-USDT-SWAP",
            ],
        )
        self.assertEqual(sorted({bot.variant.scalp_bar for bot in config.bots}), ["15s", "1s", "5s"])
        self.assertTrue(all(bot.variant.min_stop_bps == 8.0 for bot in config.bots))
        self.assertTrue(all(bot.variant.max_stop_bps == 35.0 for bot in config.bots))
        one_second = [bot.variant for bot in config.bots if bot.variant.scalp_bar == "1s"]
        mainline = [bot.variant for bot in config.bots if bot.variant.scalp_bar in {"5s", "15s"}]
        self.assertEqual(len(one_second), 6)
        self.assertTrue(all(row.entry_mode == "fractal_confirm" for row in one_second))
        self.assertTrue(all(row.order_flow_mode == "weak" for row in one_second))
        self.assertTrue(all(row.position_size_multiplier == 0.25 for row in one_second))
        self.assertTrue(all(row.entry_mode == "breakout" for row in mainline))
        self.assertTrue(all(row.order_flow_mode == "strong" for row in mainline))
        self.assertTrue(all(row.position_size_multiplier == 1.0 for row in mainline))

    def test_five_bar_scalp_paper_config_uses_binance_market_data(self):
        config = load_fleet_config("configs/fleet/five_bar_scalp_18bot_paper.json")

        self.assertIsInstance(market_data_for_config(config), BinanceRestMarketData)

    def test_scalp_suite_batch5_manifest_plans_5_by_6_paper_bots(self):
        with open("configs/scalping/scalp_suite_batch5_30bot_manifest.json", encoding="utf-8") as f:
            manifest = json.load(f)

        fourth_batch = load_fleet_config("configs/fleet/five_bar_scalp_18bot_paper.json")
        signal_fleet = load_fleet_config(manifest["fleet_config"])
        maker_configs = [load_market_maker_config(path) for path in manifest["market_maker_configs"]]

        self.assertEqual(manifest["batch_id"], "scalp-suite-batch5-30bot-paper-v1")
        self.assertEqual(manifest["strategy_count_per_symbol"], 5)
        self.assertEqual(manifest["total_bots"], 30)
        self.assertEqual(sorted(manifest["symbols"]), sorted(fourth_batch.market_data.symbols))
        self.assertEqual(sorted(signal_fleet.market_data.symbols), sorted(fourth_batch.market_data.symbols))
        self.assertEqual(len(signal_fleet.bots), 24)
        self.assertEqual(len(maker_configs), 6)
        self.assertEqual(len(signal_fleet.bots) + len(maker_configs), 30)
        self.assertFalse(signal_fleet.execution.allow_live_orders)
        self.assertTrue(all(not config.execution.allow_live_orders for config in maker_configs))
        self.assertEqual(signal_fleet.risk.max_open_positions, 24)
        self.assertEqual(signal_fleet.risk.max_open_symbols, 6)
        self.assertEqual(
            {bot.strategy_version for bot in signal_fleet.bots},
            {
                "scalp_ofi_microprice_directional_v1",
                "scalp_funding_basis_delta_neutral_v1",
                "scalp_vwap_mean_reversion_v1",
                "scalp_volatility_breakout_v1",
            },
        )
        self.assertEqual(set(Counter(bot.variant.symbol for bot in signal_fleet.bots).values()), {4})
        self.assertEqual(
            sorted(config.symbol for config in maker_configs),
            ["BNBUSDT", "BTCUSDT", "DOGEUSDT", "ETHUSDT", "HYPEUSDT", "XRPUSDT"],
        )
        self.assertTrue(
            all(config.strategy.strategy_version == "scalp_passive_maker_ofi_v1" for config in maker_configs)
        )

    def test_scalp_suite_batch7_manifest_plans_4_by_6_hft_paper_bots(self):
        with open("configs/scalping/scalp_suite_batch7_24bot_manifest.json", encoding="utf-8") as f:
            manifest = json.load(f)

        signal_fleet = load_hft_scalp_fleet_config(manifest["event_signal_fleet_config"])
        maker_configs = [load_market_maker_config(path) for path in manifest["market_maker_configs"]]

        self.assertEqual(manifest["batch_id"], "scalp-suite-batch7-24bot-paper-v1")
        self.assertEqual(manifest["strategy_count_per_symbol"], 4)
        self.assertEqual(manifest["total_bots"], 24)
        self.assertEqual(manifest["event_signal_bots"], 18)
        self.assertEqual(manifest["inventory_maker_bots"], 6)
        self.assertEqual(len(signal_fleet.bots), 18)
        self.assertEqual(len(maker_configs), 6)
        self.assertEqual(len(signal_fleet.bots) + len(maker_configs), 24)
        self.assertFalse(signal_fleet.allow_live_orders)
        self.assertTrue(all(not config.execution.allow_live_orders for config in maker_configs))
        self.assertEqual(
            sorted(signal_fleet.symbols),
            [
                "BNB-USDT-SWAP",
                "BTC-USDT-SWAP",
                "DOGE-USDT-SWAP",
                "ETH-USDT-SWAP",
                "HYPE-USDT-SWAP",
                "XRP-USDT-SWAP",
            ],
        )
        self.assertEqual(set(Counter(bot.variant.symbol for bot in signal_fleet.bots).values()), {3})
        self.assertEqual(
            {bot.strategy_version for bot in signal_fleet.bots},
            {
                "hft_queue_imbalance_one_tick_v1",
                "hft_sweep_replenishment_reversion_v1",
                "hft_lead_lag_fair_value_v1",
            },
        )
        self.assertEqual(
            sorted(config.symbol for config in maker_configs),
            ["BNBUSDT", "BTCUSDT", "DOGEUSDT", "ETHUSDT", "HYPEUSDT", "XRPUSDT"],
        )
        self.assertTrue(
            all(config.strategy.strategy_version == "hft_inventory_aware_passive_mm_v1" for config in maker_configs)
        )
        self.assertTrue(
            all(
                config.strategy_tree.strategy_tree_path[:2] == ["scalping", "batch7_hft_scalp"]
                for config in maker_configs
            )
        )
        self.assertEqual(manifest["start_commands"], ["/Users/wl/projects/LiangH/scripts/install_scalp_batch7_launchagents.sh"])
        self.assertEqual(len(manifest["launchagent_plists"]), 7)
        for plist_path in manifest["launchagent_plists"]:
            plist = plistlib.loads(Path(plist_path).read_bytes())
            self.assertTrue(plist["RunAtLoad"])
            self.assertTrue(plist["KeepAlive"])
            self.assertTrue(plist["Label"].startswith("com.liangh.scalp.batch7."))


if __name__ == "__main__":
    unittest.main()

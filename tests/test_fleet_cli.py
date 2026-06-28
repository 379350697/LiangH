import unittest

from langlang_trader.config import MarketDataConfig
from langlang_trader.fleet import FleetConfig, load_fleet_config
from langlang_trader.fleet_cli import config_with_symbol_override


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


if __name__ == "__main__":
    unittest.main()

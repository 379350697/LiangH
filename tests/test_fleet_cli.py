import unittest

from langlang_trader.config import MarketDataConfig
from langlang_trader.fleet import FleetConfig
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


if __name__ == "__main__":
    unittest.main()

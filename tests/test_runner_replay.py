import os
import tempfile
import unittest

from langlang_trader.config import AppConfig, ExecutionConfig, MarketDataConfig, PaperConfig, RiskConfig
from langlang_trader.ledger import Ledger
from langlang_trader.market_data import StaticMarketData
from langlang_trader.models import Candle
from langlang_trader.runner import TradingRunner


class MutableMarketData(StaticMarketData):
    def __init__(self, candles_by_symbol, latest_prices=None):
        super().__init__(candles_by_symbol)
        self.latest_prices = latest_prices or {}

    def latest_price(self, symbol: str) -> float:
        if symbol in self.latest_prices:
            return self.latest_prices[symbol]
        return super().latest_price(symbol)


def candles(symbol="BTC-USDT-SWAP"):
    rows = []
    price = 100.0
    for idx in range(70):
        close = price * (1 + idx * 0.012)
        rows.append(
            Candle(
                symbol=symbol,
                bar="1D",
                ts=1_700_000_000_000 + idx * 86_400_000,
                open=close * 0.99,
                high=close * 1.02,
                low=close * 0.98,
                close=close,
                volume=1000 + idx,
            )
        )
    return rows


class RunnerReplayTest(unittest.TestCase):
    def test_runner_turns_candles_into_recorded_paper_trade(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = AppConfig(
                execution=ExecutionConfig(mode="paper", exchange="okx", executor="paper_okx"),
                paper=PaperConfig(initial_equity_usdt=50_000, fee_bps=5, slippage_bps=10),
                risk=RiskConfig(max_position_usdt=1_000, max_daily_loss_usdt=500, default_leverage=5),
                market_data=MarketDataConfig(symbols=["BTC-USDT-SWAP"]),
            )
            ledger = Ledger(os.path.join(tmp, "runner.sqlite3"))
            runner = TradingRunner(config=config, market_data=StaticMarketData({"BTC-USDT-SWAP": candles()}), ledger=ledger)

            cycle = runner.run_once()

            self.assertGreaterEqual(cycle["signals"], 1)
            self.assertEqual(cycle["orders"], cycle["fills"])
            self.assertGreaterEqual(len(ledger.list_rows("signals")), 1)
            self.assertGreaterEqual(len(ledger.list_rows("order_intents")), 1)
            self.assertGreaterEqual(len(ledger.list_rows("fills")), 1)
            self.assertGreaterEqual(len(ledger.list_rows("positions")), 1)

    def test_runner_closes_position_when_latest_price_crosses_stop_loss(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = AppConfig(
                execution=ExecutionConfig(mode="paper", exchange="okx", executor="paper_okx"),
                paper=PaperConfig(initial_equity_usdt=50_000, fee_bps=5, slippage_bps=10),
                risk=RiskConfig(max_position_usdt=1_000, max_daily_loss_usdt=500, default_leverage=5),
                market_data=MarketDataConfig(symbols=["BTC-USDT-SWAP"]),
            )
            market_data = MutableMarketData({"BTC-USDT-SWAP": candles()})
            ledger = Ledger(os.path.join(tmp, "runner.sqlite3"))
            runner = TradingRunner(config=config, market_data=market_data, ledger=ledger)

            first = runner.run_once()
            stop_loss = ledger.list_rows("order_intents")[0]["stop_loss"]
            market_data.latest_prices["BTC-USDT-SWAP"] = stop_loss * 0.99
            second = runner.run_once()

            self.assertEqual(first["fills"], 1)
            self.assertEqual(second["stop_exits"], 1)
            self.assertEqual(len(ledger.list_rows("positions")), 0)
            self.assertEqual(len(ledger.list_rows("fills")), 2)


if __name__ == "__main__":
    unittest.main()

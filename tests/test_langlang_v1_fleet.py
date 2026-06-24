import os
import tempfile
import unittest

from langlang_trader.config import ExecutionConfig, MarketDataConfig, PaperConfig, RiskConfig
from langlang_trader.fleet import BotConfig, FleetConfig, FleetRunner
from langlang_trader.ledger import Ledger
from langlang_trader.market_data import StaticMarketData
from langlang_trader.models import Candle
from langlang_trader.strategy import LangLangV1Variant


def candles(symbol, bar, count, step, start_ts, interval):
    rows = []
    for idx in range(count):
        close = 100.0 * (1 + idx * step)
        rows.append(
            Candle(
                symbol=symbol,
                bar=bar,
                ts=start_ts + idx * interval,
                open=close * 0.99,
                high=close * 1.02,
                low=close * 0.98,
                close=close,
                volume=1000 + idx * 10,
            )
        )
    return rows


class LangLangV1FleetTest(unittest.TestCase):
    def test_fleet_runner_records_v1_state_machine_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            symbol = "BTC-USDT-SWAP"
            variant = LangLangV1Variant(variant_id="v1-fleet", ret_20d_min=0.14)
            config = FleetConfig(
                run_id="run-v1",
                execution=ExecutionConfig(mode="paper", exchange="okx", executor="paper_okx"),
                paper=PaperConfig(initial_equity_usdt=50_000, fee_bps=5, slippage_bps=10),
                risk=RiskConfig(max_position_usdt=1_000, max_daily_loss_usdt=500, default_leverage=5),
                market_data=MarketDataConfig(symbols=[symbol], bars=["1D", "1H", "15m", "5m"]),
                ledger_path=os.path.join(tmp, "fleet.sqlite3"),
                strategy_version="rules_langlang_v1",
                bots=[BotConfig(bot_id="bot-v1", variant=variant)],
            )
            rows = (
                candles(symbol, "1D", 70, 0.012, 1_700_000_000_000, 86_400_000)
                + candles(symbol, "1H", 60, 0.002, 1_705_000_000_000, 3_600_000)
                + candles(symbol, "15m", 50, 0.0015, 1_705_000_000_000, 900_000)
                + candles(symbol, "5m", 40, 0.001, 1_705_000_000_000, 300_000)
            )
            ledger = Ledger(config.ledger_path)
            runner = FleetRunner(config=config, market_data=StaticMarketData({symbol: rows}), ledger=ledger)

            cycle = runner.run_once()

            self.assertEqual(cycle["signals"], 1)
            self.assertEqual(cycle["orders"], 1)
            signal = ledger.list_rows("signals")[0]
            self.assertEqual(signal["strategy_version"], "rules_langlang_v1")
            self.assertIn(signal["regime"], {"main_uptrend", "strong_pullback", "breakout_retest"})
            self.assertIsNotNone(signal["setup"])
            self.assertIn("decision_trace_json", signal)


if __name__ == "__main__":
    unittest.main()

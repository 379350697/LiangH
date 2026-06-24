import os
import tempfile
import unittest
from datetime import datetime, timezone

from langlang_trader.config import PaperConfig
from langlang_trader.execution.paper import PaperExecutor
from langlang_trader.ledger import Ledger
from langlang_trader.models import Candle, Position, Side
from langlang_trader.optimize import _ReplayPositionState, _apply_v1_exit_plan
from langlang_trader.strategy import LangLangV1Variant


class LangLangV1ReplayExitTest(unittest.TestCase):
    def test_partial_take_profit_then_trend_break_close(self):
        with tempfile.TemporaryDirectory() as tmp:
            symbol = "BTC-USDT-SWAP"
            prices = {symbol: 120.0}
            ledger = Ledger(os.path.join(tmp, "replay.sqlite3"))
            ledger.upsert_position(
                Position(
                    symbol=symbol,
                    side=Side.LONG,
                    qty=10.0,
                    avg_price=100.0,
                    leverage=3,
                )
            )
            executor = PaperExecutor(
                ledger=ledger,
                paper_config=PaperConfig(initial_equity_usdt=10_000, fee_bps=0, slippage_bps=0),
                price_provider=lambda requested: prices[requested],
            )
            state = _ReplayPositionState(
                symbol=symbol,
                side=Side.LONG,
                entry_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
                entry_price=100.0,
                remaining_qty=10.0,
                leverage=3,
                stop_loss=90.0,
                risk_per_unit=10.0,
                partial_take_profit_price=120.0,
                runner_take_profit_price=140.0,
                partial_exit_fraction=0.5,
            )
            active = {symbol: state}
            windows = []

            closed = _apply_v1_exit_plan(
                executor=executor,
                ledger=ledger,
                candle=Candle(symbol, "1D", 1, 118.0, 121.0, 117.0, 118.0, 1000),
                current_prices=prices,
                active_positions=active,
                position_windows=windows,
                current_day=datetime(2024, 1, 2, tzinfo=timezone.utc),
                features={"ma_20": 90.0, "h1_ret_24": 0.0, "m15_ret_8": 0.0},
                variant=LangLangV1Variant(),
            )

            self.assertFalse(closed)
            self.assertTrue(state.partial_taken)
            self.assertAlmostEqual(ledger.get_position(symbol).qty, 5.0)
            self.assertEqual(len(windows), 0)

            closed = _apply_v1_exit_plan(
                executor=executor,
                ledger=ledger,
                candle=Candle(symbol, "1D", 2, 88.0, 89.0, 86.0, 88.0, 1000),
                current_prices=prices,
                active_positions=active,
                position_windows=windows,
                current_day=datetime(2024, 1, 3, tzinfo=timezone.utc),
                features={"ma_20": 100.0, "h1_ret_24": -0.02, "m15_ret_8": -0.01},
                variant=LangLangV1Variant(),
            )

            self.assertTrue(closed)
            self.assertIsNone(ledger.get_position(symbol))
            self.assertEqual(len(windows), 1)
            self.assertEqual(windows[0]["symbol"], symbol)


if __name__ == "__main__":
    unittest.main()

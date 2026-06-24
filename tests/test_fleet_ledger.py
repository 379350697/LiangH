import os
import tempfile
import unittest

from langlang_trader.config import PaperConfig
from langlang_trader.execution.paper import PaperExecutor
from langlang_trader.ledger import Ledger
from langlang_trader.models import OrderIntent, Side


def intent(symbol="BTC-USDT-SWAP") -> OrderIntent:
    return OrderIntent(
        symbol=symbol,
        side=Side.LONG,
        order_type="market",
        qty=1.0,
        leverage=3,
        reduce_only=False,
        entry_reason="fleet_test",
        stop_loss=90.0,
        max_slippage_bps=10.0,
    )


class FleetLedgerIsolationTest(unittest.TestCase):
    def test_positions_orders_and_fills_are_isolated_by_run_and_bot(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Ledger(os.path.join(tmp, "fleet.sqlite3"))
            bot_a = base.scoped(run_id="run-1", bot_id="bot-a", variant_id="variant-a")
            bot_b = base.scoped(run_id="run-1", bot_id="bot-b", variant_id="variant-b")
            PaperExecutor(
                ledger=bot_a,
                paper_config=PaperConfig(initial_equity_usdt=10_000, fee_bps=5, slippage_bps=0),
                price_provider=lambda symbol: 100.0,
            ).place_order(intent())
            PaperExecutor(
                ledger=bot_b,
                paper_config=PaperConfig(initial_equity_usdt=10_000, fee_bps=5, slippage_bps=0),
                price_provider=lambda symbol: 100.0,
            ).place_order(intent())

            self.assertEqual(len(base.list_rows("orders")), 2)
            self.assertEqual(len(base.list_rows("fills")), 2)
            self.assertEqual(len(base.list_rows("positions")), 2)
            self.assertEqual({row["bot_id"] for row in base.list_rows("positions")}, {"bot-a", "bot-b"})
            self.assertEqual(len(bot_a.list_positions()), 1)
            self.assertEqual(len(bot_b.list_positions()), 1)
            self.assertEqual(bot_a.list_positions()[0].symbol, "BTC-USDT-SWAP")
            self.assertEqual(bot_b.list_positions()[0].symbol, "BTC-USDT-SWAP")


if __name__ == "__main__":
    unittest.main()

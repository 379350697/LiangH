import os
import tempfile
import unittest

from langlang_trader.execution.paper import PaperExecutor
from langlang_trader.ledger import Ledger
from langlang_trader.models import OrderIntent, Side
from langlang_trader.config import PaperConfig


class PaperLedgerTest(unittest.TestCase):
    def test_market_order_records_order_fill_position_and_equity(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Ledger(os.path.join(tmp, "paper.sqlite3"))
            executor = PaperExecutor(
                ledger=ledger,
                paper_config=PaperConfig(initial_equity_usdt=20_000, fee_bps=5, slippage_bps=10),
                price_provider=lambda symbol: 100.0,
            )

            result = executor.place_order(
                OrderIntent(
                    symbol="TEST-USDT-SWAP",
                    side=Side.LONG,
                    order_type="market",
                    qty=2.0,
                    leverage=3,
                    reduce_only=False,
                    entry_reason="ledger_test",
                    stop_loss=95.0,
                    max_slippage_bps=10.0,
                    strategy_version="rules_langlang_native_final",
                    regime="main_uptrend",
                    setup="starter_buy",
                    decision_trace={"entry_position_id": 1},
                    historical_match_score=0.42,
                )
            )

            self.assertEqual(result.status, "filled")
            orders = ledger.list_rows("orders")
            fills = ledger.list_rows("fills")
            positions = executor.get_positions()
            snapshots = ledger.list_rows("equity_snapshots")

            self.assertEqual(len(orders), 1)
            self.assertEqual(len(fills), 1)
            self.assertEqual(orders[0]["status"], "filled")
            self.assertEqual(orders[0]["strategy_version"], "rules_langlang_native_final")
            self.assertEqual(orders[0]["regime"], "main_uptrend")
            self.assertEqual(fills[0]["strategy_version"], "rules_langlang_native_final")
            self.assertEqual(fills[0]["setup"], "starter_buy")
            self.assertEqual(fills[0]["symbol"], "TEST-USDT-SWAP")
            self.assertAlmostEqual(fills[0]["price"], 100.1)
            self.assertEqual(len(positions), 1)
            self.assertEqual(positions[0].symbol, "TEST-USDT-SWAP")
            self.assertEqual(positions[0].strategy_version, "rules_langlang_native_final")
            self.assertEqual(positions[0].regime, "main_uptrend")
            self.assertAlmostEqual(positions[0].qty, 2.0)
            self.assertEqual(snapshots[-1]["strategy_version"], "rules_langlang_native_final")
            self.assertGreaterEqual(len(snapshots), 1)

    def test_executor_restores_cash_from_existing_fills(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Ledger(os.path.join(tmp, "paper.sqlite3"))
            config = PaperConfig(initial_equity_usdt=10_000, fee_bps=5, slippage_bps=10)
            first = PaperExecutor(
                ledger=ledger,
                paper_config=config,
                price_provider=lambda symbol: 100.0,
            )

            first.place_order(
                OrderIntent(
                    symbol="BTC-USDT-SWAP",
                    side=Side.LONG,
                    order_type="market",
                    qty=1.0,
                    leverage=3,
                    reduce_only=False,
                    entry_reason="restore_test",
                    stop_loss=95.0,
                    max_slippage_bps=10.0,
                )
            )

            restored = PaperExecutor(
                ledger=ledger,
                paper_config=config,
                price_provider=lambda symbol: 100.0,
            )

            self.assertAlmostEqual(restored.get_account().cash_usdt, first.get_account().cash_usdt)
            self.assertEqual(len(restored.get_positions()), 1)

    def test_account_snapshot_falls_back_to_position_average_when_mark_price_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Ledger(os.path.join(tmp, "paper.sqlite3"))
            config = PaperConfig(initial_equity_usdt=10_000, fee_bps=5, slippage_bps=0)
            first = PaperExecutor(
                ledger=ledger,
                paper_config=config,
                price_provider=lambda symbol: 100.0,
            )
            first.place_order(
                OrderIntent(
                    symbol="BIGTIME-USDT-SWAP",
                    side=Side.LONG,
                    order_type="market",
                    qty=1.0,
                    leverage=3,
                    reduce_only=False,
                    entry_reason="missing_mark_test",
                    stop_loss=95.0,
                    max_slippage_bps=10.0,
                )
            )

            restored = PaperExecutor(
                ledger=ledger,
                paper_config=config,
                price_provider=lambda symbol: (_ for _ in ()).throw(KeyError(symbol)),
            )

            account = restored.get_account()
            events = [
                event
                for event in ledger.list_rows("risk_events")
                if event["reason"] == "missing_mark_price_fallback"
            ]
            self.assertAlmostEqual(account.equity_usdt, restored.cash_usdt)
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["symbol"], "BIGTIME-USDT-SWAP")


if __name__ == "__main__":
    unittest.main()

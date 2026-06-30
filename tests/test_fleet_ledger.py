import json
import os
import tempfile
import unittest

from langlang_trader.config import PaperConfig
from langlang_trader.execution.paper import PaperExecutor
from langlang_trader.ledger import Ledger
from langlang_trader.models import EntrySetup, LangLangSignal, MarketRegime, OrderIntent, Side


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

    def test_trade_lifecycle_preserves_orthogonal_strategy_tree_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace = {
                "experiment_family": "orthogonal_v1",
                "entry_family": "low_position_wyckoff_long",
                "strategy_tree_variant_id": "orthogonal_v1_low_position_wyckoff_long_a",
                "strategy_tree_path": [
                    "langlang_01",
                    "langlang_plus_01",
                    "langlang_plus_01_loss",
                    "orthogonal_v1_low_position_wyckoff_long_a",
                ],
            }
            ledger = Ledger(os.path.join(tmp, "fleet.sqlite3")).scoped(
                run_id="orthogonal-run",
                bot_id="orthogonal-bot",
                variant_id="orthogonal_v1_low_position_wyckoff_long_a",
            )
            signal = LangLangSignal(
                symbol="BTC-USDT-SWAP",
                side=Side.LONG,
                strength=0.55,
                reason_codes=["orthogonal_low_position_wyckoff_long"],
                filter_codes=[],
                features={"experiment_family": "orthogonal_v1", "entry_family": "low_position_wyckoff_long"},
                invalidation_price=90.0,
                stop_loss=90.0,
                take_profit_hint=125.0,
                take_profit_plan={"partial_r": 1.25},
                hold_plan={"runner": False},
                strategy_version="rules_langlang_v1_3",
                regime=MarketRegime.PRE_MAIN_UPTREND,
                setup=EntrySetup.STARTER_BUY,
                decision_trace=trace,
            )
            signal_id = ledger.record_signal(signal, "rules_langlang_v1_3")
            order_intent = OrderIntent(
                symbol="BTC-USDT-SWAP",
                side=Side.LONG,
                order_type="market",
                qty=1.0,
                leverage=3,
                reduce_only=False,
                entry_reason="orthogonal_low_position_wyckoff_long",
                stop_loss=90.0,
                max_slippage_bps=10.0,
                strategy_version="rules_langlang_v1_3",
                regime=MarketRegime.PRE_MAIN_UPTREND.value,
                setup=EntrySetup.STARTER_BUY.value,
                decision_trace=trace,
            )
            ledger.record_order_intent(order_intent, signal_id=signal_id)

            PaperExecutor(
                ledger=ledger,
                paper_config=PaperConfig(initial_equity_usdt=10_000, fee_bps=5, slippage_bps=0),
                price_provider=lambda symbol: 100.0,
            ).place_order(order_intent)

            trade = ledger.list_rows("trade_lifecycle", run_id="orthogonal-run")[0]
            trade_trace = json.loads(trade["entry_decision_trace_json"])
            self.assertEqual(trade_trace["experiment_family"], "orthogonal_v1")
            self.assertEqual(trade_trace["entry_family"], "low_position_wyckoff_long")
            self.assertEqual(
                trade_trace["strategy_tree_variant_id"],
                "orthogonal_v1_low_position_wyckoff_long_a",
            )
            self.assertEqual(trade["entry_reason_summary"], "orthogonal_low_position_wyckoff_long")


if __name__ == "__main__":
    unittest.main()

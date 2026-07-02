import json
import os
import tempfile
import unittest

from langlang_trader.execution.paper import PaperExecutor
from langlang_trader.ledger import Ledger
from langlang_trader.models import OrderIntent, Side, Signal
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

    def test_trade_lifecycle_links_entry_signal_intent_fill_and_feature_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Ledger(os.path.join(tmp, "paper.sqlite3"))
            executor = PaperExecutor(
                ledger=ledger,
                paper_config=PaperConfig(initial_equity_usdt=20_000, fee_bps=0, slippage_bps=0),
                price_provider=lambda symbol: 100.0,
            )
            signal = Signal(
                symbol="TEST-USDT-SWAP",
                side=Side.LONG,
                strength=0.88,
                reason_codes=["golden_pit_reclaim", "wyckoff_spring_reclaim"],
                features={
                    "strong_pattern_tag": "golden_pit_reclaim",
                    "wyckoff_long_setup_tag": "spring_reclaim",
                    "btc_regime": "main_uptrend",
                },
                invalidation_price=95.0,
                take_profit_hint=112.0,
            )
            signal_id = ledger.record_signal(signal, strategy_version="rules_langlang_v1_3")
            intent = OrderIntent(
                symbol="TEST-USDT-SWAP",
                side=Side.LONG,
                order_type="market",
                qty=2.0,
                leverage=3,
                reduce_only=False,
                entry_reason="1_startup_long",
                stop_loss=95.0,
                max_slippage_bps=0.0,
                strategy_version="rules_langlang_v1_3",
                regime="main_uptrend",
                setup="starter_buy",
                decision_trace={"entry_position_id": "1_startup_long", "risk_unit": "W"},
            )
            intent_id = ledger.record_order_intent(intent, signal_id=signal_id)

            executor.place_order(intent)

            trades = ledger.list_rows("trade_lifecycle")
            events = ledger.list_rows("trade_events")
            self.assertEqual(len(trades), 1)
            self.assertEqual(trades[0]["entry_signal_id"], signal_id)
            self.assertEqual(trades[0]["entry_intent_id"], intent_id)
            self.assertEqual(trades[0]["status"], "open")
            self.assertEqual(json.loads(trades[0]["entry_reason_codes_json"]), signal.reason_codes)
            self.assertEqual(trades[0]["entry_reason_summary"], "1_startup_long")
            self.assertEqual(json.loads(trades[0]["entry_feature_snapshot_json"])["strong_pattern_tag"], "golden_pit_reclaim")
            self.assertEqual(events[0]["event_type"], "entry_fill")
            self.assertEqual(events[0]["trade_id"], trades[0]["trade_id"])

    def test_trade_lifecycle_closes_with_exit_reason_path_metrics_and_r_multiple(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Ledger(os.path.join(tmp, "paper.sqlite3"))
            mark = {"price": 100.0}
            executor = PaperExecutor(
                ledger=ledger,
                paper_config=PaperConfig(initial_equity_usdt=20_000, fee_bps=0, slippage_bps=0),
                price_provider=lambda symbol: mark["price"],
            )
            intent = OrderIntent(
                symbol="TEST-USDT-SWAP",
                side=Side.LONG,
                order_type="market",
                qty=1.0,
                leverage=3,
                reduce_only=False,
                entry_reason="small_divergence_absorb",
                stop_loss=95.0,
                max_slippage_bps=0.0,
                decision_trace={"reason_codes": ["small_divergence_absorb"]},
            )

            executor.place_order(intent)
            mark["price"] = 97.0
            executor.get_account()
            mark["price"] = 104.0
            executor.close_position("TEST-USDT-SWAP", reason="wyckoff_utad_exit")

            trade = ledger.list_rows("trade_lifecycle")[0]
            events = ledger.list_rows("trade_events")
            self.assertEqual(trade["status"], "closed")
            self.assertEqual(trade["exit_reason_summary"], "wyckoff_utad_exit")
            self.assertEqual(json.loads(trade["exit_reason_codes_json"]), ["wyckoff_utad_exit"])
            self.assertAlmostEqual(trade["gross_pnl_usdt"], 4.0)
            self.assertAlmostEqual(trade["realized_pnl_usdt"], 4.0)
            self.assertAlmostEqual(trade["mae_usdt"], -3.0)
            self.assertAlmostEqual(trade["mfe_usdt"], 4.0)
            self.assertAlmostEqual(trade["r_multiple"], 0.8)
            event_types = [event["event_type"] for event in events]
            self.assertEqual(event_types[0], "entry_fill")
            self.assertIn("path_mark", event_types)
            self.assertEqual(event_types[-1], "close_fill")

    def test_missing_mark_price_adds_data_quality_flag_without_fabricating_excursion(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Ledger(os.path.join(tmp, "paper.sqlite3"))
            config = PaperConfig(initial_equity_usdt=10_000, fee_bps=0, slippage_bps=0)
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
                    entry_reason="golden_pit_reclaim",
                    stop_loss=95.0,
                    max_slippage_bps=0.0,
                )
            )

            restored = PaperExecutor(
                ledger=ledger,
                paper_config=config,
                price_provider=lambda symbol: (_ for _ in ()).throw(KeyError(symbol)),
            )

            restored.get_account()

            trade = ledger.list_rows("trade_lifecycle")[0]
            self.assertEqual(json.loads(trade["data_quality_flags_json"]), ["mark_price_fallback"])
            self.assertEqual(trade["mae_usdt"], 0.0)
            self.assertEqual(trade["mfe_usdt"], 0.0)

    def test_partial_take_profit_stays_on_same_trade_and_final_close_accumulates_pnl(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Ledger(os.path.join(tmp, "paper.sqlite3"))
            mark = {"price": 100.0}
            executor = PaperExecutor(
                ledger=ledger,
                paper_config=PaperConfig(initial_equity_usdt=20_000, fee_bps=0, slippage_bps=0),
                price_provider=lambda symbol: mark["price"],
            )
            executor.place_order(
                OrderIntent(
                    symbol="TEST-USDT-SWAP",
                    side=Side.LONG,
                    order_type="market",
                    qty=2.0,
                    leverage=3,
                    reduce_only=False,
                    entry_reason="leader_platform_start",
                    stop_loss=95.0,
                    max_slippage_bps=0.0,
                )
            )

            mark["price"] = 106.0
            executor.place_order(
                OrderIntent(
                    symbol="TEST-USDT-SWAP",
                    side=Side.SHORT,
                    order_type="market",
                    qty=1.0,
                    leverage=3,
                    reduce_only=True,
                    entry_reason="close:partial_take_profit",
                    stop_loss=None,
                    max_slippage_bps=0.0,
                    exit_reason="partial_take_profit",
                )
            )
            mark["price"] = 108.0
            executor.close_position("TEST-USDT-SWAP", reason="take_profit_reached")

            trade = ledger.list_rows("trade_lifecycle")[0]
            events = ledger.list_rows("trade_events")
            self.assertEqual(trade["status"], "closed")
            self.assertAlmostEqual(trade["qty"], 2.0)
            self.assertAlmostEqual(trade["open_qty"], 0.0)
            self.assertAlmostEqual(trade["gross_pnl_usdt"], 14.0)
            self.assertAlmostEqual(trade["realized_pnl_usdt"], 14.0)
            self.assertEqual([event["event_type"] for event in events if event["event_type"] != "path_mark"], [
                "entry_fill",
                "partial_take_profit",
                "close_fill",
            ])

    def test_account_snapshot_uses_quote_fallback_before_position_average(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Ledger(os.path.join(tmp, "paper.sqlite3"))
            config = PaperConfig(initial_equity_usdt=10_000, fee_bps=0, slippage_bps=0)
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
                    entry_reason="quote_fallback_test",
                    stop_loss=95.0,
                    max_slippage_bps=10.0,
                )
            )

            def missing_mark(symbol: str) -> float:
                raise KeyError(symbol)

            restored = PaperExecutor(
                ledger=ledger,
                paper_config=config,
                price_provider=missing_mark,
                quote_fallback=lambda symbol: 112.0,
            )

            account = restored.get_account()
            events = ledger.list_rows("risk_events")
            self.assertAlmostEqual(account.equity_usdt, 10_012.0)
            self.assertEqual([event["reason"] for event in events], ["missing_mark_price_recovered"])
            self.assertEqual(events[0]["symbol"], "BIGTIME-USDT-SWAP")

    def test_non_reduce_only_reverse_fill_closes_old_lifecycle_and_opens_residual_side(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Ledger(os.path.join(tmp, "paper.sqlite3"))
            mark = {"price": 100.0}
            executor = PaperExecutor(
                ledger=ledger,
                paper_config=PaperConfig(initial_equity_usdt=20_000, fee_bps=0, slippage_bps=0),
                price_provider=lambda symbol: mark["price"],
            )
            executor.place_order(
                OrderIntent(
                    symbol="TEST-USDT-SWAP",
                    side=Side.LONG,
                    order_type="market",
                    qty=1.0,
                    leverage=3,
                    reduce_only=False,
                    entry_reason="initial_long",
                    stop_loss=95.0,
                    max_slippage_bps=0.0,
                )
            )

            mark["price"] = 105.0
            executor.place_order(
                OrderIntent(
                    symbol="TEST-USDT-SWAP",
                    side=Side.SHORT,
                    order_type="market",
                    qty=1.5,
                    leverage=3,
                    reduce_only=False,
                    entry_reason="reverse_short",
                    stop_loss=110.0,
                    max_slippage_bps=0.0,
                )
            )

            trades = ledger.list_rows("trade_lifecycle")
            closed_long = next(row for row in trades if row["side"] == "long")
            open_short = next(row for row in trades if row["side"] == "short")
            position = ledger.get_position("TEST-USDT-SWAP")
            self.assertEqual(closed_long["status"], "closed")
            self.assertAlmostEqual(closed_long["open_qty"], 0.0)
            self.assertEqual(open_short["status"], "open")
            self.assertAlmostEqual(open_short["open_qty"], 0.5)
            self.assertEqual(position.side, Side.SHORT)
            self.assertAlmostEqual(position.qty, 0.5)


if __name__ == "__main__":
    unittest.main()

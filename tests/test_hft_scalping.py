import json
import os
import tempfile
import unittest

from langlang_trader.hft_scalping import (
    HftScalpPaperRunner,
    HftScalpVariant,
    RulesLeadLagFairValueStrategy,
    RulesQueueImbalanceOneTickStrategy,
    RulesSweepReplenishmentReversionStrategy,
    load_hft_scalp_fleet_config,
)
from langlang_trader.ledger import Ledger
from liangh_trader.market_maker.models import BookTick, TradeTick


def _book(
    symbol: str = "BTCUSDT",
    *,
    bid: float = 100.00,
    ask: float = 100.02,
    bid_qty: float = 80.0,
    ask_qty: float = 10.0,
    now_ns: int = 1_000_000_000,
) -> BookTick:
    return BookTick(
        symbol=symbol,
        event_time_ms=1_700_000_000_000,
        receive_time_ns=now_ns,
        best_bid=bid,
        best_bid_qty=bid_qty,
        best_ask=ask,
        best_ask_qty=ask_qty,
        update_id=1,
    )


class HftScalpingStrategyTest(unittest.TestCase):
    def test_queue_imbalance_one_tick_signal_has_full_tp_sl_trace(self):
        variant = HftScalpVariant(
            variant_id="hft_queue_imbalance_btc_v1",
            symbol="BTC-USDT-SWAP",
            exchange_symbol="BTCUSDT",
            strategy_kind="queue_imbalance_one_tick",
            strategy_tree_parent_id="hft_queue_imbalance_one_tick_v1",
            strategy_tree_path=("scalping", "batch7_hft_scalp", "hft_queue_imbalance_one_tick", "hft_queue_imbalance_btc_v1"),
            min_queue_imbalance=0.60,
            take_profit_bps=10.0,
            stop_bps=2.0,
        )

        signal = RulesQueueImbalanceOneTickStrategy(variant).on_book(_book())

        self.assertIsNotNone(signal)
        assert signal is not None
        self.assertEqual(signal.side.value, "long")
        self.assertLess(signal.invalidation_price, signal.features["entry_price"])
        self.assertGreater(signal.take_profit_hint, signal.features["entry_price"])
        self.assertEqual(signal.decision_trace["exit_semantics"], "full_tp_sl")
        self.assertEqual(signal.features["strategy_tree_variant_id"], "hft_queue_imbalance_btc_v1")
        self.assertEqual(signal.decision_trace["take_profit_cost_floor_bps"], 10.0)

    def test_sweep_replenishment_reversion_enters_after_failed_replenishment(self):
        variant = HftScalpVariant(
            variant_id="hft_sweep_replenishment_btc_v1",
            symbol="BTC-USDT-SWAP",
            exchange_symbol="BTCUSDT",
            strategy_kind="sweep_replenishment_reversion",
            min_sweep_notional_usdt=10_000.0,
            replenishment_ratio=0.40,
        )
        strategy = RulesSweepReplenishmentReversionStrategy(variant)

        strategy.on_trade(
            TradeTick(
                symbol="BTCUSDT",
                event_time_ms=1_700_000_000_000,
                receive_time_ns=1_000_000_000,
                price=100.02,
                qty=150.0,
                is_buyer_maker=False,
                trade_id="sweep-buy",
            )
        )
        signal = strategy.on_book(_book(bid_qty=100.0, ask_qty=10.0, now_ns=1_050_000_000))

        self.assertIsNotNone(signal)
        assert signal is not None
        self.assertEqual(signal.side.value, "short")
        self.assertIn("sweep_replenishment_failed", signal.reason_codes)

    def test_lead_lag_fair_value_trades_lag_when_reference_moves_first(self):
        variant = HftScalpVariant(
            variant_id="hft_lead_lag_eth_v1",
            symbol="ETH-USDT-SWAP",
            exchange_symbol="ETHUSDT",
            strategy_kind="lead_lag_fair_value",
            lead_exchange_symbol="BTCUSDT",
            min_lead_move_bps=8.0,
            min_lag_divergence_bps=4.0,
        )
        strategy = RulesLeadLagFairValueStrategy(variant)

        strategy.on_book(_book("BTCUSDT", bid=100.00, ask=100.02, bid_qty=20.0, ask_qty=20.0, now_ns=1_000_000_000))
        strategy.on_book(_book("BTCUSDT", bid=101.00, ask=101.02, bid_qty=20.0, ask_qty=20.0, now_ns=1_010_000_000))
        signal = strategy.on_book(_book("ETHUSDT", bid=50.00, ask=50.02, bid_qty=20.0, ask_qty=20.0, now_ns=1_020_000_000))

        self.assertIsNotNone(signal)
        assert signal is not None
        self.assertEqual(signal.side.value, "long")
        self.assertIn("lead_lag_fair_value", signal.reason_codes)
        self.assertGreater(signal.features["lead_move_bps"], 8.0)

    def test_paper_runner_opens_and_closes_full_position_on_take_profit(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = os.path.join(tmp, "hft.sqlite3")
            variant = HftScalpVariant(
                variant_id="hft_queue_imbalance_btc_v1",
                symbol="BTC-USDT-SWAP",
                exchange_symbol="BTCUSDT",
                strategy_kind="queue_imbalance_one_tick",
                take_profit_bps=10.0,
                stop_bps=2.0,
                position_size_usdt=100.0,
            )
            runner = HftScalpPaperRunner(
                run_id="unit-hft-batch7",
                ledger=Ledger(ledger_path),
                bots=[("batch7_hft_queue_imbalance_btc_paper", variant, RulesQueueImbalanceOneTickStrategy.version)],
            )

            runner.on_book(_book(now_ns=1_000_000_000))
            runner.on_book(_book(bid=100.14, ask=100.16, now_ns=1_010_000_000))

            rows = Ledger(ledger_path).list_rows("trade_lifecycle", run_id="unit-hft-batch7")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "closed")
            self.assertEqual(rows[0]["open_qty"], 0.0)
            self.assertEqual(json.loads(rows[0]["exit_reason_codes_json"]), ["take_profit_exit"])
            closes = Ledger(ledger_path).list_rows("orders", run_id="unit-hft-batch7")
            reduce_only = [row for row in closes if row["reduce_only"] == 1]
            self.assertEqual(len(reduce_only), 1)
            self.assertEqual(reduce_only[0]["exit_reason"], "take_profit_exit")

    def test_paper_runner_after_restart_does_not_merge_new_hft_entry_into_stale_open_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = os.path.join(tmp, "hft.sqlite3")
            variant = HftScalpVariant(
                variant_id="hft_queue_imbalance_btc_v1",
                symbol="BTC-USDT-SWAP",
                exchange_symbol="BTCUSDT",
                strategy_kind="queue_imbalance_one_tick",
                take_profit_bps=10.0,
                stop_bps=2.0,
                position_size_usdt=100.0,
            )
            bot = ("batch7_hft_queue_imbalance_btc_paper", variant, RulesQueueImbalanceOneTickStrategy.version)
            first_runner = HftScalpPaperRunner(
                run_id="unit-hft-batch7",
                ledger=Ledger(ledger_path),
                bots=[bot],
            )
            first_runner.on_book(_book(now_ns=1_000_000_000))

            restarted_runner = HftScalpPaperRunner(
                run_id="unit-hft-batch7",
                ledger=Ledger(ledger_path),
                bots=[bot],
            )
            restarted_runner.on_book(_book(now_ns=2_000_000_000))
            restarted_runner.on_book(_book(bid=100.14, ask=100.16, now_ns=2_010_000_000))

            ledger = Ledger(ledger_path)
            lifecycle = ledger.list_rows("trade_lifecycle", run_id="unit-hft-batch7")
            self.assertEqual([row["status"] for row in lifecycle].count("closed"), 1)
            self.assertEqual([row["status"] for row in lifecycle].count("open"), 0)
            closed = [row for row in lifecycle if row["status"] == "closed"][0]
            self.assertEqual(closed["open_qty"], 0.0)
            self.assertEqual(json.loads(closed["exit_reason_codes_json"]), ["take_profit_exit"])

            trade_events = ledger.list_rows("trade_events", run_id="unit-hft-batch7")
            self.assertEqual([row["event_type"] for row in trade_events].count("entry_fill"), 1)
            self.assertEqual([row["event_type"] for row in trade_events].count("add_fill"), 0)
            self.assertEqual([row["event_type"] for row in trade_events].count("partial_take_profit"), 0)

    def test_paper_runner_after_restart_restores_open_position_and_does_not_duplicate_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = os.path.join(tmp, "hft.sqlite3")
            variant = HftScalpVariant(
                variant_id="hft_queue_imbalance_btc_v1",
                symbol="BTC-USDT-SWAP",
                exchange_symbol="BTCUSDT",
                strategy_kind="queue_imbalance_one_tick",
                take_profit_bps=10.0,
                stop_bps=2.0,
                position_size_usdt=100.0,
            )
            bot = ("batch7_hft_queue_imbalance_btc_paper", variant, RulesQueueImbalanceOneTickStrategy.version)
            first_runner = HftScalpPaperRunner(
                run_id="unit-hft-batch7",
                ledger=Ledger(ledger_path),
                bots=[bot],
            )
            first_runner.on_book(_book(now_ns=1_000_000_000))

            restarted_runner = HftScalpPaperRunner(
                run_id="unit-hft-batch7",
                ledger=Ledger(ledger_path),
                bots=[bot],
            )
            restarted_runner.on_book(_book(now_ns=2_000_000_000))

            lifecycle = Ledger(ledger_path).list_rows("trade_lifecycle", run_id="unit-hft-batch7")
            self.assertEqual([row["status"] for row in lifecycle].count("open"), 1)

    def test_config_rejects_event_signal_take_profit_below_round_trip_fee_floor(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "hft_config.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "run_id": "unit-hft-batch7",
                        "ledger_path": os.path.join(tmp, "hft.sqlite3"),
                        "execution": {"allow_live_orders": False},
                        "paper": {"initial_equity_usdt": 10_000.0, "fee_bps": 4.0, "slippage_bps": 2.0},
                        "symbols": ["BTC-USDT-SWAP"],
                        "exchange_symbols": ["BTCUSDT"],
                        "bots": [
                            {
                                "bot_id": "batch7_hft_queue_imbalance_btc_paper",
                                "strategy_version": RulesQueueImbalanceOneTickStrategy.version,
                                "variant": {
                                    "variant_id": "hft_queue_imbalance_btc_v1",
                                    "symbol": "BTC-USDT-SWAP",
                                    "exchange_symbol": "BTCUSDT",
                                    "strategy_kind": "queue_imbalance_one_tick",
                                    "take_profit_bps": 3.5,
                                    "stop_bps": 2.5,
                                },
                            }
                        ],
                    },
                    handle,
                )

            with self.assertRaisesRegex(ValueError, "take_profit_bps.*10.0"):
                load_hft_scalp_fleet_config(path)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path

from liangh_trader.market_maker.binance_execution_ws import BinanceWsApiRequestBuilder
from liangh_trader.market_maker.binance_rest_recovery import BinanceRestRecoveryClient, RecoveryContextError
from liangh_trader.market_maker.binance_user_stream import parse_order_trade_update
from liangh_trader.market_maker.binance_ws import BinanceUsdmWebSocketMarketData, LocalOrderBook
from liangh_trader.market_maker.cli import _event_dispatch_lag_ms, build_parser
from liangh_trader.market_maker.config import MarketMakerConfigError, load_market_maker_config
from liangh_trader.market_maker.exchange_interfaces import MarketSignalState, RateLimitBudget
from liangh_trader.market_maker.hybrid_runtime import HybridMarketMakerRuntime
from liangh_trader.market_maker.ledger import MarketMakerLedger
from liangh_trader.market_maker.live_executor import LiveExecutorSafetyError, assert_live_orders_enabled
from liangh_trader.market_maker.models import BookTick, InventoryState, OrderTruthEvent, QuoteIntent, TopBookTick, TradeTick
from liangh_trader.market_maker.paper_executor import MarketMakerPaperExecutor
from liangh_trader.market_maker.runner import MarketMakerRunner
from liangh_trader.market_maker.strategy import (
    InventoryAwarePassiveMakerStrategy,
    OfiInventorySkewMakerStrategy,
    ReferencePassiveMakerStrategy,
)


def write_config(tmp: str, **overrides) -> str:
    config = {
        "run_id": "mm-test-run",
        "bot_id": "mm-test-bot",
        "mode": "paper",
        "venue": "binance_usdm",
        "symbol": "BTCUSDT",
        "allowed_symbols": ["BTCUSDT", "ETHUSDT"],
        "ledger_path": os.path.join(tmp, "mm.sqlite3"),
        "execution": {"allow_live_orders": False},
        "paper": {"initial_quote_usdt": 10_000.0, "maker_fee_bps": 2.0, "taker_fee_bps": 5.0},
        "risk": {
            "max_inventory_base": 0.02,
            "max_notional_usdt": 500.0,
            "stale_feed_ms": 1500,
            "max_loop_lag_ms": 200,
            "max_spread_bps": 50.0,
        },
        "strategy": {
            "strategy_version": "market_maker_v1",
            "variant_id": "mm_v1_reference_passive_btcusdt",
            "quote_size_usdt": 100.0,
            "quote_spread_bps": 4.0,
            "order_ttl_ms": 1000,
            "quote_interval_ms": 250,
        },
        "strategy_tree": {
            "strategy_tree_variant_id": "mm_v1_reference_passive_btcusdt",
            "strategy_tree_parent_id": "market_maker_v1_root",
            "strategy_tree_path": ["market_making", "market_maker_v1", "reference_passive"],
        },
    }
    config.update(overrides)
    path = Path(tmp) / "config.json"
    path.write_text(__import__("json").dumps(config), encoding="utf-8")
    return str(path)


class MarketMakerConfigTest(unittest.TestCase):
    def test_config_requires_single_allowed_symbol(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_config(tmp, symbols=["BTCUSDT", "ETHUSDT"])
            with self.assertRaisesRegex(MarketMakerConfigError, "single symbol"):
                load_market_maker_config(path)

            path = write_config(tmp, symbol="NOTTOPUSDT")
            with self.assertRaisesRegex(MarketMakerConfigError, "allowed_symbols"):
                load_market_maker_config(path)

    def test_live_orders_require_triple_explicit_enablement(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_config(tmp, mode="live", execution={"allow_live_orders": True})
            config = load_market_maker_config(path)
            with self.assertRaises(LiveExecutorSafetyError):
                assert_live_orders_enabled(config, env={})
            self.assertTrue(
                assert_live_orders_enabled(
                    config,
                    env={"LIANGH_MARKET_MAKER_LIVE_ORDERS": "1"},
                )
            )

    def test_hybrid_execution_signal_and_limit_defaults_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_market_maker_config(write_config(tmp))

        self.assertEqual(config.execution.primary_gateway, "paper")
        self.assertTrue(config.signals.use_book_ticker)
        self.assertTrue(config.signals.use_trade_flow)
        self.assertLess(config.limits.max_order_ops_per_10s, 300)
        self.assertLess(config.limits.max_order_ops_per_minute, 1200)

    def test_binance_ws_primary_gateway_is_allowed_as_dry_run_in_paper(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_market_maker_config(
                write_config(
                    tmp,
                    execution={"allow_live_orders": False, "primary_gateway": "binance_ws_api"},
                )
            )

        self.assertEqual(config.mode, "paper")
        self.assertEqual(config.execution.primary_gateway, "binance_ws_api")


class MarketMakerCliTimingTest(unittest.TestCase):
    def test_dispatch_lag_uses_event_receive_time_not_message_gap(self):
        event = BookTick(
            symbol="BTCUSDT",
            event_time_ms=1_000,
            receive_time_ns=10_000_000,
            best_bid=99.0,
            best_bid_qty=5.0,
            best_ask=101.0,
            best_ask_qty=5.0,
            update_id=1,
        )

        self.assertEqual(_event_dispatch_lag_ms(event, now_ns=10_500_000), 0.5)
        self.assertEqual(_event_dispatch_lag_ms(event, now_ns=9_000_000), 0.0)


class InventoryAwareMakerStrategyTest(unittest.TestCase):
    def test_inventory_aware_hft_maker_only_quotes_inventory_reducing_side_when_adverse(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_config(
                tmp,
                strategy={
                    "strategy_version": "hft_inventory_aware_passive_mm_v1",
                    "variant_id": "hft_inventory_mm_btc_v1",
                    "quote_size_usdt": 100.0,
                    "quote_spread_bps": 4.0,
                    "order_ttl_ms": 500,
                    "quote_interval_ms": 100,
                    "min_quote_edge_bps": 1.0,
                    "min_ofi_abs": 0.20,
                    "inventory_stop_bps": 20.0,
                    "adverse_ofi_ticks": 2,
                    "max_inventory_hold_ms": 10_000,
                },
                strategy_tree={
                    "strategy_tree_variant_id": "hft_inventory_mm_btc_v1",
                    "strategy_tree_parent_id": "hft_inventory_aware_passive_mm_v1",
                    "strategy_tree_path": [
                        "scalping",
                        "batch7_hft_scalp",
                        "hft_inventory_aware_passive_mm",
                        "hft_inventory_mm_btc_v1",
                    ],
                },
            )
            config = load_market_maker_config(path)
            strategy = InventoryAwarePassiveMakerStrategy(config)
            book = BookTick(
                symbol="BTCUSDT",
                event_time_ms=1_700_000_000_000,
                receive_time_ns=1_000_000,
                best_bid=99.99,
                best_bid_qty=5.0,
                best_ask=100.01,
                best_ask_qty=50.0,
                update_id=1,
            )
            inventory = InventoryState(symbol="BTCUSDT", base_qty=0.018, avg_price=100.0)

            quotes = strategy.generate_quotes(book, inventory, now_ns=2_000_000)

            self.assertEqual([quote.side for quote in quotes], ["sell"])
            self.assertEqual(quotes[0].strategy_tree_path[:2], ["scalping", "batch7_hft_scalp"])


class BinanceOrderBookTest(unittest.TestCase):
    def test_buffered_depth_snapshot_bridge_promotes_book_hot(self):
        book = LocalOrderBook("ETHUSDT")
        buffered = book.apply_depth_event(
            {
                "E": 1_010,
                "U": 101,
                "u": 102,
                "pu": 100,
                "b": [["100.50", "1.0"], ["100.00", "0"]],
                "a": [["101.00", "2.0"]],
            },
            receive_time_ns=3_000_000,
        )

        self.assertTrue(buffered.stale)
        self.assertEqual(book.status, "bootstrapping")
        self.assertEqual(book.buffered_depth_event_count, 1)

        bridged = book.apply_snapshot(
            last_update_id=100,
            bids=[["100.00", "2.0"]],
            asks=[["101.00", "3.0"]],
            event_time_ms=1_000,
            receive_time_ns=4_000_000,
        )

        self.assertFalse(bridged.stale)
        self.assertEqual(bridged.book_status, "hot")
        self.assertEqual(bridged.source, "l2_depth")
        self.assertEqual(book.status, "hot")
        self.assertEqual(book.last_update_id, 102)
        self.assertEqual(bridged.best_bid, 100.5)
        self.assertEqual(bridged.best_ask, 101.0)

    def test_snapshot_boundary_miss_keeps_book_rebuilding(self):
        book = LocalOrderBook("ETHUSDT")
        book.apply_depth_event(
            {"E": 1_010, "U": 110, "u": 120, "pu": 109, "b": [["100.50", "1.0"]], "a": []},
            receive_time_ns=3_000_000,
        )

        tick = book.apply_snapshot(
            last_update_id=100,
            bids=[["100.00", "2.0"]],
            asks=[["101.00", "3.0"]],
            event_time_ms=1_000,
            receive_time_ns=4_000_000,
        )

        self.assertTrue(tick.stale)
        self.assertEqual(tick.book_status, "rebuilding")
        self.assertEqual(book.status, "rebuilding")
        self.assertIn("snapshot_boundary", book.fault_reason)

    def test_depth_sequence_gap_marks_book_stale(self):
        book = LocalOrderBook("BTCUSDT")
        book.apply_snapshot(
            last_update_id=100,
            bids=[["100.0", "2.0"]],
            asks=[["101.0", "3.0"]],
            event_time_ms=1_000,
            receive_time_ns=2_000_000,
        )

        tick = book.apply_depth_event(
            {
                "E": 1_010,
                "U": 101,
                "u": 102,
                "pu": 100,
                "b": [["100.5", "1.0"], ["100.0", "0"]],
                "a": [["101.0", "2.0"]],
            },
            receive_time_ns=3_000_000,
        )

        self.assertFalse(tick.sequence_gap)
        self.assertFalse(tick.stale)
        self.assertEqual(tick.best_bid, 100.5)
        self.assertEqual(tick.best_ask, 101.0)

        gap = book.apply_depth_event(
            {"E": 1_020, "U": 103, "u": 104, "pu": 999, "b": [], "a": []},
            receive_time_ns=4_000_000,
        )

        self.assertTrue(gap.sequence_gap)
        self.assertTrue(gap.stale)
        self.assertTrue(book.stale)
        self.assertEqual(book.status, "rebuilding")
        self.assertEqual(book.sequence_gap_count, 1)

    def test_book_ticker_does_not_mutate_l2_depth_state(self):
        book = LocalOrderBook("ETHUSDT")
        book.apply_depth_event(
            {"E": 1_010, "U": 110, "u": 120, "pu": 109, "b": [["100.50", "1.0"]], "a": []},
            receive_time_ns=3_000_000,
        )
        self.assertEqual(book.status, "bootstrapping")
        self.assertIsNone(book.last_update_id)

        top = book.apply_book_ticker(
            {"E": 1_020, "u": 9_999, "b": "100.75", "B": "2.0", "a": "100.85", "A": "2.5"},
            receive_time_ns=4_000_000,
        )

        self.assertIsInstance(top, TopBookTick)
        self.assertEqual(top.source, "book_ticker")
        self.assertEqual(top.best_bid, 100.75)
        self.assertEqual(book.status, "bootstrapping")
        self.assertIsNone(book.last_update_id)
        self.assertTrue(book.stale)


class BinanceMarketDataReconnectTest(unittest.TestCase):
    def test_stream_reconnects_after_connection_reset(self):
        class FakeWebSocket:
            def __init__(self, attempt: int) -> None:
                self.attempt = attempt
                self.emitted = False

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.attempt == 1:
                    raise ConnectionResetError("fixture reset")
                if self.emitted:
                    raise StopAsyncIteration
                self.emitted = True
                return '{"data":{"e":"trade","E":1000,"p":"100.0","q":"0.5","m":false,"t":"t-1"}}'

        class FakeWebSocketsModule:
            def __init__(self) -> None:
                self.attempts = 0

            def connect(self, *args, **kwargs):
                self.attempts += 1
                return FakeWebSocket(self.attempts)

        fake_websockets = FakeWebSocketsModule()
        previous = sys.modules.get("websockets")
        sys.modules["websockets"] = fake_websockets
        try:
            market_data = BinanceUsdmWebSocketMarketData("BTCUSDT")
            market_data.reconnect_initial_backoff_s = 0.0

            async def collect_one():
                async for event in market_data.stream():
                    return event
                return None

            event = asyncio.run(asyncio.wait_for(collect_one(), timeout=1.0))
        finally:
            if previous is None:
                sys.modules.pop("websockets", None)
            else:
                sys.modules["websockets"] = previous

        self.assertIsInstance(event, TradeTick)
        self.assertEqual(fake_websockets.attempts, 2)
        self.assertEqual(market_data.connection_error_count, 1)


class MarketMakerStrategyTest(unittest.TestCase):
    def test_reference_strategy_skews_quotes_when_inventory_is_near_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_market_maker_config(write_config(tmp))
        strategy = ReferencePassiveMakerStrategy(config)
        book = BookTick(
            symbol="BTCUSDT",
            event_time_ms=1_000,
            receive_time_ns=1_000_000,
            best_bid=10_000.0,
            best_bid_qty=5.0,
            best_ask=10_001.0,
            best_ask_qty=5.0,
            update_id=10,
        )

        flat_quotes = strategy.generate_quotes(book, InventoryState(symbol="BTCUSDT"), now_ns=2_000_000)
        self.assertEqual([quote.side for quote in flat_quotes], ["buy", "sell"])
        self.assertLessEqual(flat_quotes[0].price, book.best_bid)
        self.assertGreaterEqual(flat_quotes[1].price, book.best_ask)
        self.assertEqual(flat_quotes[0].strategy_tree_variant_id, "mm_v1_reference_passive_btcusdt")

        long_inventory = InventoryState(symbol="BTCUSDT", base_qty=0.019)
        long_quotes = strategy.generate_quotes(book, long_inventory, now_ns=2_000_000)
        self.assertEqual([quote.side for quote in long_quotes], ["sell"])

        short_inventory = InventoryState(symbol="BTCUSDT", base_qty=-0.019)
        short_quotes = strategy.generate_quotes(book, short_inventory, now_ns=2_000_000)
        self.assertEqual([quote.side for quote in short_quotes], ["buy"])

    def test_strategies_cap_inventory_increasing_quote_qty_near_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_market_maker_config(
                write_config(
                    tmp,
                    risk={
                        "max_inventory_base": 1.0,
                        "max_notional_usdt": 500.0,
                        "stale_feed_ms": 1500,
                        "max_loop_lag_ms": 200,
                        "max_spread_bps": 50.0,
                    },
                    strategy={
                        "strategy_version": "scalp_passive_maker_ofi_v1",
                        "variant_id": "scalp_maker_ofi_btc_v1",
                        "quote_size_usdt": 100.0,
                        "quote_spread_bps": 4.0,
                        "order_ttl_ms": 1000,
                        "quote_interval_ms": 250,
                        "min_quote_edge_bps": 0.0,
                        "min_ofi_abs": 0.2,
                    },
                )
            )
        book = BookTick(
            symbol="BTCUSDT",
            event_time_ms=1_000,
            receive_time_ns=1_000_000,
            best_bid=99.99,
            best_bid_qty=5.0,
            best_ask=100.01,
            best_ask_qty=5.0,
            update_id=10,
        )
        inventory = InventoryState(symbol="BTCUSDT", base_qty=0.75)

        for strategy in (ReferencePassiveMakerStrategy(config), OfiInventorySkewMakerStrategy(config)):
            quotes = strategy.generate_quotes(book, inventory, now_ns=2_000_000)
            by_side = {quote.side: quote for quote in quotes}
            self.assertLessEqual(by_side["buy"].qty, 0.25 + 1e-12)
            self.assertGreater(by_side["sell"].qty, 0.9)

    def test_ofi_inventory_skew_strategy_blocks_adverse_side_and_keeps_tree_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_market_maker_config(
                write_config(
                    tmp,
                    strategy={
                        "strategy_version": "scalp_passive_maker_ofi_v1",
                        "variant_id": "scalp_maker_ofi_btc_v1",
                        "quote_size_usdt": 100.0,
                        "quote_spread_bps": 4.0,
                        "order_ttl_ms": 1000,
                        "quote_interval_ms": 250,
                        "min_quote_edge_bps": 1.0,
                        "min_ofi_abs": 0.2,
                    },
                    strategy_tree={
                        "strategy_tree_variant_id": "scalp_maker_ofi_btc_v1",
                        "strategy_tree_parent_id": "scalp_passive_maker_ofi_v1",
                        "strategy_tree_path": ["scalping", "scalp_passive_maker_ofi_v1", "scalp_maker_ofi_btc_v1"],
                    },
                )
            )
        strategy = OfiInventorySkewMakerStrategy(config)
        book = BookTick(
            symbol="BTCUSDT",
            event_time_ms=1_000,
            receive_time_ns=1_000_000,
            best_bid=10_000.0,
            best_bid_qty=2.0,
            best_ask=10_004.0,
            best_ask_qty=10.0,
            update_id=10,
        )

        quotes = strategy.generate_quotes(book, InventoryState(symbol="BTCUSDT"), now_ns=2_000_000)

        self.assertEqual([quote.side for quote in quotes], ["sell"])
        self.assertEqual(quotes[0].strategy_version, "scalp_passive_maker_ofi_v1")
        self.assertEqual(quotes[0].strategy_tree_variant_id, "scalp_maker_ofi_btc_v1")


class MarketMakerPaperExecutorTest(unittest.TestCase):
    def test_post_only_limit_order_fills_from_trade_tick_and_records_strategy_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_market_maker_config(write_config(tmp))
            ledger = MarketMakerLedger(config.ledger_path, context=config.ledger_context)
            executor = MarketMakerPaperExecutor(config=config, ledger=ledger)
            book = BookTick(
                symbol="BTCUSDT",
                event_time_ms=1_000,
                receive_time_ns=1_000_000,
                best_bid=99.0,
                best_bid_qty=5.0,
                best_ask=101.0,
                best_ask_qty=5.0,
                update_id=1,
            )
            quote = QuoteIntent(
                symbol="BTCUSDT",
                side="buy",
                price=98.5,
                qty=0.5,
                created_at_ns=1_100_000,
                ttl_ms=1000,
                post_only=True,
                strategy_version="market_maker_v1",
                strategy_tree_variant_id="mm_v1_reference_passive_btcusdt",
                strategy_tree_parent_id="market_maker_v1_root",
                strategy_tree_path=["market_making", "market_maker_v1", "reference_passive"],
            )

            accepted = executor.place_quotes([quote], book=book)
            self.assertEqual(len(accepted), 1)
            fills = executor.on_trade(
                TradeTick(
                    symbol="BTCUSDT",
                    event_time_ms=1_200,
                    receive_time_ns=1_200_000,
                    price=98.0,
                    qty=1.0,
                    is_buyer_maker=False,
                    trade_id="t1",
                )
            )

            self.assertEqual(len(fills), 1)
            self.assertEqual(fills[0].liquidity, "maker")
            self.assertGreater(executor.inventory.base_qty, 0.0)
            fill_rows = ledger.list_rows("mm_fills")
            self.assertEqual(fill_rows[0]["strategy_tree_variant_id"], "mm_v1_reference_passive_btcusdt")
            self.assertEqual(fill_rows[0]["liquidity"], "maker")

    def test_post_only_crossing_quote_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_market_maker_config(write_config(tmp))
            ledger = MarketMakerLedger(config.ledger_path, context=config.ledger_context)
            executor = MarketMakerPaperExecutor(config=config, ledger=ledger)
            book = BookTick(
                symbol="BTCUSDT",
                event_time_ms=1_000,
                receive_time_ns=1_000_000,
                best_bid=99.0,
                best_bid_qty=5.0,
                best_ask=101.0,
                best_ask_qty=5.0,
                update_id=1,
            )
            crossing = QuoteIntent(
                symbol="BTCUSDT",
                side="buy",
                price=101.0,
                qty=0.1,
                created_at_ns=1_100_000,
                ttl_ms=1000,
                post_only=True,
                strategy_version="market_maker_v1",
                strategy_tree_variant_id="mm_v1_reference_passive_btcusdt",
                strategy_tree_parent_id="market_maker_v1_root",
                strategy_tree_path=["market_making", "market_maker_v1", "reference_passive"],
            )

            self.assertEqual(executor.place_quotes([crossing], book=book), [])
            self.assertEqual(ledger.list_rows("mm_orders")[0]["status"], "rejected")

    def test_flatten_inventory_closes_position_with_taker_stop_fill(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_market_maker_config(write_config(tmp))
            ledger = MarketMakerLedger(config.ledger_path, context=config.ledger_context)
            executor = MarketMakerPaperExecutor(config=config, ledger=ledger)
            executor.inventory.base_qty = 0.25
            executor.inventory.avg_price = 100.0
            book = BookTick(
                symbol="BTCUSDT",
                event_time_ms=1_000,
                receive_time_ns=1_000_000,
                best_bid=99.5,
                best_bid_qty=5.0,
                best_ask=100.5,
                best_ask_qty=5.0,
                update_id=1,
            )

            fills = executor.flatten_inventory(book, now_ns=2_000_000, reason="inventory_stop_loss")

            self.assertEqual(len(fills), 1)
            self.assertEqual(fills[0].side, "sell")
            self.assertEqual(fills[0].liquidity, "taker_stop")
            self.assertAlmostEqual(executor.inventory.base_qty, 0.0)
            fill_rows = ledger.list_rows("mm_fills")
            self.assertEqual(fill_rows[-1]["liquidity"], "taker_stop")


class MarketMakerRunnerRiskTest(unittest.TestCase):
    def test_stale_feed_and_loop_lag_cancel_pending_quotes(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_market_maker_config(write_config(tmp))
            ledger = MarketMakerLedger(config.ledger_path, context=config.ledger_context)
            executor = MarketMakerPaperExecutor(config=config, ledger=ledger)
            runner = MarketMakerRunner(config=config, ledger=ledger, executor=executor)
            book = BookTick(
                symbol="BTCUSDT",
                event_time_ms=1_000,
                receive_time_ns=1_000_000,
                best_bid=99.0,
                best_bid_qty=5.0,
                best_ask=101.0,
                best_ask_qty=5.0,
                update_id=1,
            )
            strategy = ReferencePassiveMakerStrategy(config)
            executor.place_quotes(strategy.generate_quotes(book, executor.inventory, now_ns=1_000_000), book=book)
            self.assertEqual(len(executor.open_orders()), 2)

            allowed = runner.evaluate_risk(
                book=book,
                now_ns=3_000_000_000,
                last_event_receive_ns=1_000_000,
                loop_lag_ms=250.0,
            )

            self.assertFalse(allowed)
            self.assertEqual(executor.open_orders(), [])
            reasons = [row["reason"] for row in ledger.list_rows("mm_risk_events")]
            self.assertIn("stale_feed", reasons)
            self.assertIn("loop_lag_exceeded", reasons)

    def test_inventory_cap_halts_quotes(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_market_maker_config(write_config(tmp))
            ledger = MarketMakerLedger(config.ledger_path, context=config.ledger_context)
            executor = MarketMakerPaperExecutor(config=config, ledger=ledger)
            runner = MarketMakerRunner(config=config, ledger=ledger, executor=executor)
            executor.inventory.base_qty = 0.021
            book = BookTick(
                symbol="BTCUSDT",
                event_time_ms=1_000,
                receive_time_ns=1_000_000,
                best_bid=100.0,
                best_bid_qty=5.0,
                best_ask=100.01,
                best_ask_qty=5.0,
                update_id=1,
            )

            allowed = runner.evaluate_risk(
                book=book,
                now_ns=1_100_000,
                last_event_receive_ns=1_000_000,
                loop_lag_ms=1.0,
            )

            self.assertFalse(allowed)
            reasons = [row["reason"] for row in ledger.list_rows("mm_risk_events")]
            self.assertIn("inventory_cap_exceeded", reasons)

    def test_inventory_cap_force_flattens_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_market_maker_config(write_config(tmp))
            ledger = MarketMakerLedger(config.ledger_path, context=config.ledger_context)
            executor = MarketMakerPaperExecutor(config=config, ledger=ledger)
            runner = MarketMakerRunner(config=config, ledger=ledger, executor=executor)
            executor.inventory.base_qty = 0.021
            executor.inventory.avg_price = 100.0
            book = BookTick(
                symbol="BTCUSDT",
                event_time_ms=1_000,
                receive_time_ns=1_000_000,
                best_bid=99.9,
                best_bid_qty=5.0,
                best_ask=100.01,
                best_ask_qty=5.0,
                update_id=1,
                source="l2_depth",
                book_status="hot",
            )

            allowed = runner.evaluate_risk(
                book=book,
                now_ns=1_100_000,
                last_event_receive_ns=1_000_000,
                loop_lag_ms=1.0,
            )

            self.assertFalse(allowed)
            self.assertAlmostEqual(executor.inventory.base_qty, 0.0)
            fill_rows = ledger.list_rows("mm_fills")
            self.assertEqual(fill_rows[-1]["liquidity"], "taker_stop")
            self.assertEqual(fill_rows[-1]["trade_id"], "inventory_cap_exceeded")
            inventory_rows = ledger.list_rows("mm_inventory_snapshots")
            self.assertEqual(inventory_rows[-1]["reason"], "inventory_cap_exceeded")

    def test_runner_blocks_rebuilding_depth_book_and_records_latency(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_market_maker_config(write_config(tmp))
            ledger = MarketMakerLedger(config.ledger_path, context=config.ledger_context)
            executor = MarketMakerPaperExecutor(config=config, ledger=ledger)
            runner = MarketMakerRunner(config=config, ledger=ledger, executor=executor)
            hot_book = BookTick(
                symbol="BTCUSDT",
                event_time_ms=1_000,
                receive_time_ns=1_000_000,
                best_bid=99.99,
                best_bid_qty=5.0,
                best_ask=100.01,
                best_ask_qty=5.0,
                update_id=1,
                source="l2_depth",
                book_status="hot",
            )
            runner.on_book(hot_book, now_ns=2_000_000, loop_lag_ms=1.0)
            self.assertEqual(len(executor.open_orders()), 2)

            rebuilding = BookTick(
                symbol="BTCUSDT",
                event_time_ms=2_000,
                receive_time_ns=2_000_000,
                best_bid=99.99,
                best_bid_qty=5.0,
                best_ask=100.01,
                best_ask_qty=5.0,
                update_id=2,
                stale=True,
                source="l2_depth",
                book_status="rebuilding",
                resync_count=1,
                sequence_gap_count=1,
            )
            runner.on_book(rebuilding, now_ns=3_000_000, loop_lag_ms=1.0)

            self.assertEqual(executor.open_orders(), [])
            reasons = [row["reason"] for row in ledger.list_rows("mm_risk_events")]
            self.assertIn("book_not_hot", reasons)
            latency_rows = ledger.list_rows("mm_latency_events")
            self.assertTrue(latency_rows)
            payload = __import__("json").loads(latency_rows[-1]["payload_json"])
            self.assertEqual(payload["book_status"], "rebuilding")
            self.assertEqual(payload["sequence_gap_count"], 1)

    def test_runner_ignores_top_book_ticker_for_quotes(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_market_maker_config(write_config(tmp))
            ledger = MarketMakerLedger(config.ledger_path, context=config.ledger_context)
            executor = MarketMakerPaperExecutor(config=config, ledger=ledger)
            runner = MarketMakerRunner(config=config, ledger=ledger, executor=executor)

            runner.on_book(
                TopBookTick(
                    symbol="BTCUSDT",
                    event_time_ms=1_000,
                    receive_time_ns=1_000_000,
                    best_bid=99.0,
                    best_bid_qty=5.0,
                    best_ask=101.0,
                    best_ask_qty=5.0,
                    update_id=1,
                ),
                now_ns=2_000_000,
                loop_lag_ms=1.0,
            )

            self.assertEqual(executor.open_orders(), [])
            reasons = [row["reason"] for row in ledger.list_rows("mm_risk_events")]
            self.assertIn("top_book_not_tradeable", reasons)


class FakeRecoveryGateway:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def list_open_orders(self, symbol: str, context: str) -> list[object]:
        self.calls.append(f"list:{symbol}:{context}")
        return []

    def cancel_all_open_orders(self, symbol: str, context: str) -> dict[str, object]:
        self.calls.append(f"cancel:{symbol}:{context}")
        return {"symbol": symbol}


class HybridRuntimeTest(unittest.TestCase):
    def test_top_book_and_trade_update_signals_without_generating_quotes(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_market_maker_config(write_config(tmp))
            ledger = MarketMakerLedger(config.ledger_path, context=config.ledger_context)
            runtime = HybridMarketMakerRuntime(config=config, ledger=ledger)

            runtime.on_market_event(
                TopBookTick(
                    symbol="BTCUSDT",
                    event_time_ms=1_000,
                    receive_time_ns=1_000_000,
                    best_bid=99.0,
                    best_bid_qty=5.0,
                    best_ask=101.0,
                    best_ask_qty=5.0,
                    update_id=1,
                ),
                now_ns=2_000_000,
            )
            runtime.on_market_event(
                TradeTick(
                    symbol="BTCUSDT",
                    event_time_ms=1_001,
                    receive_time_ns=1_100_000,
                    price=100.0,
                    qty=0.5,
                    is_buyer_maker=True,
                    trade_id="trade-1",
                ),
                now_ns=2_100_000,
            )

            self.assertEqual(ledger.list_rows("mm_quotes"), [])
            self.assertEqual(ledger.list_rows("mm_orders"), [])
            self.assertIsNotNone(runtime.signal_state.last_top_book)
            self.assertIsNotNone(runtime.signal_state.last_trade)
            latency_names = [row["name"] for row in ledger.list_rows("mm_latency_events")]
            self.assertIn("market_data_book", latency_names)
            self.assertIn("market_data_trade", latency_names)

    def test_hot_l2_generates_paper_quotes_and_execution_request_records_strategy_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_market_maker_config(write_config(tmp))
            ledger = MarketMakerLedger(config.ledger_path, context=config.ledger_context)
            runtime = HybridMarketMakerRuntime(config=config, ledger=ledger)
            book = BookTick(
                symbol="BTCUSDT",
                event_time_ms=1_000,
                receive_time_ns=1_000_000,
                best_bid=99.99,
                best_bid_qty=5.0,
                best_ask=100.01,
                best_ask_qty=5.0,
                update_id=1,
                source="l2_depth",
                book_status="hot",
            )

            runtime.on_market_event(book, now_ns=2_000_000)

            self.assertEqual(len(ledger.list_rows("mm_quotes")), 2)
            request_rows = ledger.list_rows("mm_execution_requests")
            self.assertEqual(len(request_rows), 1)
            self.assertEqual(request_rows[0]["gateway"], "paper")
            self.assertEqual(request_rows[0]["method"], "quote_batch")
            self.assertEqual(request_rows[0]["strategy_tree_variant_id"], "mm_v1_reference_passive_btcusdt")

    def test_inventory_cap_force_flattens_paper_gateway(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_market_maker_config(write_config(tmp))
            ledger = MarketMakerLedger(config.ledger_path, context=config.ledger_context)
            runtime = HybridMarketMakerRuntime(config=config, ledger=ledger)
            runtime.execution_gateway.inventory.base_qty = -0.021
            runtime.execution_gateway.inventory.avg_price = 100.0
            book = BookTick(
                symbol="BTCUSDT",
                event_time_ms=1_000,
                receive_time_ns=1_000_000,
                best_bid=99.99,
                best_bid_qty=5.0,
                best_ask=100.01,
                best_ask_qty=5.0,
                update_id=1,
                source="l2_depth",
                book_status="hot",
            )

            runtime.on_market_event(book, now_ns=2_000_000)

            self.assertAlmostEqual(runtime.execution_gateway.inventory.base_qty, 0.0)
            fill_rows = ledger.list_rows("mm_fills")
            self.assertEqual(fill_rows[-1]["side"], "buy")
            self.assertEqual(fill_rows[-1]["liquidity"], "taker_stop")
            self.assertEqual(fill_rows[-1]["trade_id"], "inventory_cap_exceeded")

    def test_rate_limit_budget_blocks_quote_cycle_without_calling_rest_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_market_maker_config(write_config(tmp))
            ledger = MarketMakerLedger(config.ledger_path, context=config.ledger_context)
            recovery = FakeRecoveryGateway()
            runtime = HybridMarketMakerRuntime(
                config=config,
                ledger=ledger,
                recovery_gateway=recovery,
                rate_limit_budget=RateLimitBudget(max_order_ops_per_minute=1, max_order_ops_per_10s=1),
            )
            runtime.rate_limit_budget.record_order_op(cost=1, now_ns=1_000_000)
            book = BookTick(
                symbol="BTCUSDT",
                event_time_ms=1_000,
                receive_time_ns=1_000_000,
                best_bid=99.99,
                best_bid_qty=5.0,
                best_ask=100.01,
                best_ask_qty=5.0,
                update_id=1,
                source="l2_depth",
                book_status="hot",
            )

            runtime.on_market_event(book, now_ns=2_000_000)

            self.assertEqual(ledger.list_rows("mm_quotes"), [])
            self.assertEqual(recovery.calls, [])
            reasons = [row["reason"] for row in ledger.list_rows("mm_risk_events")]
            self.assertIn("rate_limit_backoff", reasons)

    def test_signal_spread_shock_cancels_existing_orders_but_does_not_quote_from_top_book(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_market_maker_config(write_config(tmp))
            ledger = MarketMakerLedger(config.ledger_path, context=config.ledger_context)
            runtime = HybridMarketMakerRuntime(config=config, ledger=ledger)
            hot_book = BookTick(
                symbol="BTCUSDT",
                event_time_ms=1_000,
                receive_time_ns=1_000_000,
                best_bid=99.99,
                best_bid_qty=5.0,
                best_ask=100.01,
                best_ask_qty=5.0,
                update_id=1,
                source="l2_depth",
                book_status="hot",
            )
            runtime.on_market_event(hot_book, now_ns=2_000_000)
            self.assertEqual(len(runtime.open_orders()), 2)

            runtime.on_market_event(
                TopBookTick(
                    symbol="BTCUSDT",
                    event_time_ms=1_001,
                    receive_time_ns=1_100_000,
                    best_bid=90.0,
                    best_bid_qty=5.0,
                    best_ask=110.0,
                    best_ask_qty=5.0,
                    update_id=2,
                ),
                now_ns=3_000_000,
            )

            self.assertEqual(runtime.open_orders(), [])
            self.assertEqual(len(ledger.list_rows("mm_quotes")), 2)
            reasons = [row["reason"] for row in ledger.list_rows("mm_risk_events")]
            self.assertIn("spread_shock", reasons)

    def test_user_data_truth_event_records_and_overrides_local_order_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_market_maker_config(write_config(tmp))
            ledger = MarketMakerLedger(config.ledger_path, context=config.ledger_context)
            runtime = HybridMarketMakerRuntime(config=config, ledger=ledger)
            hot_book = BookTick(
                symbol="BTCUSDT",
                event_time_ms=1_000,
                receive_time_ns=1_000_000,
                best_bid=99.99,
                best_bid_qty=5.0,
                best_ask=100.01,
                best_ask_qty=5.0,
                update_id=1,
                source="l2_depth",
                book_status="hot",
            )
            runtime.on_market_event(hot_book, now_ns=2_000_000)
            order = runtime.open_orders()[0]

            runtime.on_order_truth_event(
                OrderTruthEvent(
                    symbol="BTCUSDT",
                    order_id=order.order_id,
                    client_order_id=order.order_id,
                    event_type="ORDER_TRADE_UPDATE",
                    order_status="CANCELED",
                    execution_type="CANCELED",
                    side=order.side.upper(),
                    price=order.price,
                    qty=order.qty,
                    filled_qty=0.0,
                    last_fill_qty=0.0,
                    last_fill_price=0.0,
                    event_time_ms=1_001,
                    transaction_time_ms=1_001,
                    receive_time_ns=3_000_000,
                    truth_source="user_data",
                    strategy_version=config.strategy.strategy_version,
                    strategy_tree_variant_id=config.strategy_tree.strategy_tree_variant_id,
                    strategy_tree_parent_id=config.strategy_tree.strategy_tree_parent_id,
                    strategy_tree_path=list(config.strategy_tree.strategy_tree_path),
                )
            )

            self.assertEqual(ledger.list_rows("mm_order_truth_events")[0]["truth_source"], "user_data")
            self.assertNotIn(order.order_id, [open_order.order_id for open_order in runtime.open_orders()])
            order_rows = ledger.list_rows("mm_orders")
            self.assertEqual(order_rows[-1]["status"], "canceled")
            self.assertEqual(order_rows[-1]["reason"], "user_data_truth")


class ExchangeInterfaceTest(unittest.TestCase):
    def test_rate_limit_budget_uses_binance_order_rate_limit_snapshots(self):
        budget = RateLimitBudget(max_order_ops_per_minute=1000, max_order_ops_per_10s=240)

        budget.sync_from_binance_rate_limits(
            [
                {"rateLimitType": "ORDERS", "interval": "MINUTE", "intervalNum": 1, "limit": 1200, "count": 1199},
                {"rateLimitType": "ORDERS", "interval": "SECOND", "intervalNum": 10, "limit": 300, "count": 299},
            ]
        )

        self.assertFalse(budget.can_submit(cost=2, now_ns=10_000_000))
        self.assertTrue(budget.can_submit(cost=1, now_ns=10_000_000))

    def test_market_signal_state_marks_spread_shock_from_top_book(self):
        state = MarketSignalState(max_spread_bps=50.0)

        state.update_top_book(
            TopBookTick(
                symbol="BTCUSDT",
                event_time_ms=1_000,
                receive_time_ns=1_000_000,
                best_bid=90.0,
                best_bid_qty=1.0,
                best_ask=110.0,
                best_ask_qty=1.0,
                update_id=1,
            )
        )

        self.assertTrue(state.spread_shock)
        self.assertTrue(state.cancel_urgency)


class BinanceWsExecutionTest(unittest.TestCase):
    def test_ws_order_requests_are_signed_and_decimal_serialized(self):
        builder = BinanceWsApiRequestBuilder(api_key="key", api_secret="secret")

        request = builder.build_order_place_request(
            symbol="ETHUSDT",
            side="buy",
            price=100.1200,
            quantity=0.001,
            client_order_id="cid-1",
            timestamp_ms=1_717_171_717_000,
            request_id="req-1",
        )

        self.assertEqual(request["id"], "req-1")
        self.assertEqual(request["method"], "order.place")
        params = request["params"]
        self.assertEqual(params["apiKey"], "key")
        self.assertEqual(params["symbol"], "ETHUSDT")
        self.assertEqual(params["side"], "BUY")
        self.assertEqual(params["type"], "LIMIT")
        self.assertEqual(params["timeInForce"], "GTX")
        self.assertEqual(params["price"], "100.12")
        self.assertEqual(params["quantity"], "0.001")
        self.assertEqual(
            params["signature"],
            builder.sign_params({key: value for key, value in params.items() if key != "signature"}),
        )

    def test_ws_modify_and_cancel_requests_use_expected_methods(self):
        builder = BinanceWsApiRequestBuilder(api_key="key", api_secret="secret")

        modify = builder.build_order_modify_request(
            symbol="ETHUSDT",
            order_id="123",
            side="sell",
            price=101.5,
            quantity=0.002,
            timestamp_ms=1_717_171_717_000,
            request_id="modify-1",
        )
        cancel = builder.build_order_cancel_request(
            symbol="ETHUSDT",
            order_id="123",
            timestamp_ms=1_717_171_717_001,
            request_id="cancel-1",
        )

        self.assertEqual(modify["method"], "order.modify")
        self.assertEqual(modify["params"]["side"], "SELL")
        self.assertEqual(modify["params"]["price"], "101.5")
        self.assertEqual(cancel["method"], "order.cancel")
        self.assertEqual(cancel["params"]["orderId"], "123")


class BinanceUserStreamTest(unittest.TestCase):
    def test_order_trade_update_maps_to_user_data_truth_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_market_maker_config(write_config(tmp))
            ledger = MarketMakerLedger(config.ledger_path, context=config.ledger_context)
            event = parse_order_trade_update(
                {
                    "e": "ORDER_TRADE_UPDATE",
                    "E": 1_717_171_717_111,
                    "o": {
                        "s": "BTCUSDT",
                        "c": "client-1",
                        "S": "BUY",
                        "i": 12345,
                        "x": "TRADE",
                        "X": "PARTIALLY_FILLED",
                        "p": "100.12",
                        "q": "0.010",
                        "z": "0.004",
                        "l": "0.004",
                        "L": "100.12",
                        "T": 1_717_171_717_100,
                    },
                },
                receive_time_ns=9_000_000,
                context=config.ledger_context,
            )

            self.assertIsNotNone(event)
            assert event is not None
            ledger.record_order_truth_event(event)

            rows = ledger.list_rows("mm_order_truth_events")
            self.assertEqual(rows[0]["truth_source"], "user_data")
            self.assertEqual(rows[0]["execution_type"], "TRADE")
            self.assertEqual(rows[0]["order_status"], "PARTIALLY_FILLED")
            self.assertEqual(rows[0]["strategy_tree_variant_id"], "mm_v1_reference_passive_btcusdt")


class BinanceRestRecoveryTest(unittest.TestCase):
    def test_rest_recovery_rejects_quote_loop_context(self):
        client = BinanceRestRecoveryClient(api_key="key", api_secret="secret")

        with self.assertRaises(RecoveryContextError):
            client.build_open_orders_request(symbol="BTCUSDT", context="quote_loop", timestamp_ms=1_000)

        request = client.build_open_orders_request(symbol="BTCUSDT", context="startup", timestamp_ms=1_000)
        self.assertEqual(request["method"], "GET")
        self.assertEqual(request["path"], "/fapi/v1/openOrders")
        self.assertEqual(request["params"]["symbol"], "BTCUSDT")


class MarketMakerConfigArtifactTest(unittest.TestCase):
    def test_default_btc_config_and_strategy_tree_load(self):
        config = load_market_maker_config("configs/market_maker/market_maker_v1_btcusdt.json")
        self.assertEqual(config.symbol, "BTCUSDT")
        self.assertEqual(config.mode, "paper")
        self.assertEqual(config.strategy_tree.strategy_tree_variant_id, "mm_v1_reference_passive_btcusdt")
        tree_path = Path("configs/strategy_library/market_maker_strategy_tree.json")
        self.assertTrue(tree_path.exists())

    def test_default_eth_config_and_market_data_smoke_flag_load(self):
        config = load_market_maker_config("configs/market_maker/market_maker_v1_ethusdt.json")
        self.assertEqual(config.symbol, "ETHUSDT")
        self.assertEqual(config.mode, "paper")
        self.assertIn("ETHUSDT", config.allowed_symbols)

        args = build_parser().parse_args(
            [
                "--config",
                "configs/market_maker/market_maker_v1_ethusdt.json",
                "--duration-seconds",
                "300",
                "--market-data-smoke",
                "--hybrid-runtime",
            ]
        )
        self.assertTrue(args.market_data_smoke)
        self.assertTrue(args.hybrid_runtime)

    def test_default_configs_include_binance_mainstream_top_cap_futures_whitelist(self):
        expected_symbols = {
            "BTCUSDT",
            "ETHUSDT",
            "BNBUSDT",
            "XRPUSDT",
            "SOLUSDT",
            "TRXUSDT",
            "HYPEUSDT",
            "DOGEUSDT",
            "ZECUSDT",
            "XLMUSDT",
            "XMRUSDT",
            "ADAUSDT",
            "LINKUSDT",
            "BCHUSDT",
            "LTCUSDT",
            "HBARUSDT",
            "AVAXUSDT",
            "SUIUSDT",
        }

        for path in (
            "configs/market_maker/market_maker_v1_btcusdt.json",
            "configs/market_maker/market_maker_v1_ethusdt.json",
        ):
            with self.subTest(path=path):
                config = load_market_maker_config(path)
                self.assertTrue(expected_symbols.issubset(set(config.allowed_symbols)))
                self.assertNotIn("CCUSDT", config.allowed_symbols)
                self.assertNotIn("LABUSDT", config.allowed_symbols)


if __name__ == "__main__":
    unittest.main()

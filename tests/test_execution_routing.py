import os
import tempfile
import unittest

from langlang_trader.config import PaperConfig
from langlang_trader.execution.paper import BinancePaperExecutor, MultiExchangePaperExecutor, OkxPaperExecutor
from langlang_trader.execution.routing import ExecutionRouter
from langlang_trader.ledger import Ledger
from langlang_trader.models import OrderIntent, OrderResult, Side
from langlang_trader.universe import OkxBinanceUniverseProvider


def intent(symbol):
    return OrderIntent(
        symbol=symbol,
        side=Side.LONG,
        order_type="market",
        qty=1.0,
        leverage=3,
        reduce_only=False,
        entry_reason="routing_test",
        stop_loss=95.0,
        max_slippage_bps=10.0,
    )


def combined_snapshot():
    okx_payload = {
        "code": "0",
        "data": [
            {
                "instId": "BTC-USDT-SWAP",
                "instType": "SWAP",
                "state": "live",
                "settleCcy": "USDT",
                "baseCcy": "BTC",
                "quoteCcy": "USDT",
            },
            {
                "instId": "SHARED-USDT-SWAP",
                "instType": "SWAP",
                "state": "live",
                "settleCcy": "USDT",
                "baseCcy": "SHARED",
                "quoteCcy": "USDT",
            },
            {
                "instId": "OKXONLY-USDT-SWAP",
                "instType": "SWAP",
                "state": "live",
                "settleCcy": "USDT",
                "baseCcy": "OKXONLY",
                "quoteCcy": "USDT",
            },
        ],
    }
    binance_payload = {
        "symbols": [
            {
                "symbol": "BTCUSDT",
                "contractType": "PERPETUAL",
                "status": "TRADING",
                "baseAsset": "BTC",
                "quoteAsset": "USDT",
            },
            {
                "symbol": "SHAREDUSDT",
                "contractType": "PERPETUAL",
                "status": "TRADING",
                "baseAsset": "SHARED",
                "quoteAsset": "USDT",
            },
            {
                "symbol": "BINONLYUSDT",
                "contractType": "PERPETUAL",
                "status": "TRADING",
                "baseAsset": "BINONLY",
                "quoteAsset": "USDT",
            },
        ]
    }
    return OkxBinanceUniverseProvider.snapshot_from_payloads(okx_payload, binance_payload)


class ExecutionRoutingTest(unittest.TestCase):
    def test_router_uses_binance_for_shared_symbols_and_exchange_native_symbol_mapping(self):
        router = ExecutionRouter(combined_snapshot())

        shared = router.route(intent("SHARED-USDT-SWAP"))
        okx_only = router.route(intent("OKXONLY-USDT-SWAP"))
        binance_only = router.route(intent("BINONLY-USDT-SWAP"))
        unavailable = router.route(intent("MISSING-USDT-SWAP"))

        self.assertEqual(shared.exchange, "binance")
        self.assertEqual(shared.exchange_symbol, "SHAREDUSDT")
        self.assertEqual(shared.route_reason, "shared_binance_preferred")
        self.assertEqual(okx_only.exchange, "okx")
        self.assertEqual(okx_only.exchange_symbol, "OKXONLY-USDT-SWAP")
        self.assertEqual(okx_only.route_reason, "okx_only")
        self.assertEqual(binance_only.exchange, "binance")
        self.assertEqual(binance_only.exchange_symbol, "BINONLYUSDT")
        self.assertEqual(binance_only.route_reason, "binance_only")
        self.assertIsNone(unavailable)
        self.assertEqual(router.rejection_reason(intent("MISSING-USDT-SWAP")), "symbol_not_executable_on_configured_exchanges")

    def test_okx_and_binance_paper_executors_share_contract_but_isolate_ledger_by_exchange(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Ledger(os.path.join(tmp, "multi.sqlite3"), run_id="r1", bot_id="bot1")
            paper = PaperConfig(initial_equity_usdt=10_000, fee_bps=5, slippage_bps=10)
            okx = OkxPaperExecutor(
                ledger=ledger,
                paper_config=paper,
                price_provider=lambda symbol: 100.0,
            )
            binance = BinancePaperExecutor(
                ledger=ledger,
                paper_config=paper,
                price_provider=lambda symbol: 100.0,
            )

            okx_result = okx.place_order(intent("SHARED-USDT-SWAP"))
            binance_result = binance.place_order(intent("SHARED-USDT-SWAP"))

            self.assertIsInstance(okx_result, OrderResult)
            self.assertIsInstance(binance_result, OrderResult)
            self.assertEqual(okx_result.status, "filled")
            self.assertEqual(binance_result.status, "filled")
            orders = ledger.list_rows("orders", run_id="r1", bot_id="bot1")
            fills = ledger.list_rows("fills", run_id="r1", bot_id="bot1")
            positions = ledger.list_rows("positions", run_id="r1", bot_id="bot1")
            self.assertEqual([row["exchange"] for row in orders], ["okx", "binance"])
            self.assertEqual([row["exchange"] for row in fills], ["okx", "binance"])
            self.assertEqual({row["exchange"] for row in positions}, {"okx", "binance"})
            self.assertEqual(len(ledger.list_positions(exchange="okx")), 1)
            self.assertEqual(len(ledger.list_positions(exchange="binance")), 1)
            self.assertIn('"paper_exchange": "binance"', fills[1]["raw_payload_json"])
            self.assertIn('"exchange_symbol": "SHAREDUSDT"', orders[1]["raw_payload_json"])

    def test_multi_exchange_paper_executor_keeps_one_bot_level_cash_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Ledger(os.path.join(tmp, "multi.sqlite3"), run_id="r1", bot_id="bot1")
            paper = PaperConfig(initial_equity_usdt=10_000, fee_bps=5, slippage_bps=10)
            executor = MultiExchangePaperExecutor(
                ledger=ledger,
                paper_config=paper,
                price_provider=lambda symbol: 100.0,
                router=ExecutionRouter(combined_snapshot()),
            )

            account_before = executor.get_account()
            result = executor.place_order(intent("BINONLY-USDT-SWAP"))
            account_after = executor.get_account()

            self.assertEqual(result.status, "filled")
            self.assertEqual(account_before.cash_usdt, 10_000)
            self.assertLess(account_after.cash_usdt, 10_000)
            self.assertGreater(account_after.cash_usdt, 9_999)
            self.assertLess(account_after.equity_usdt, 10_000)


if __name__ == "__main__":
    unittest.main()

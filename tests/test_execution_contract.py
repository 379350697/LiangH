import os
import tempfile
import unittest

from langlang_trader.config import AppConfig, ExecutionConfig, PaperConfig, RiskConfig
from langlang_trader.execution.live_okx import OkxLiveExecutor
from langlang_trader.execution.paper import PaperExecutor
from langlang_trader.ledger import Ledger
from langlang_trader.models import OrderIntent, OrderResult, Side


class MockOkxTransport:
    def __init__(self):
        self.calls = []

    def request(self, method, path, body=None, auth=False):
        self.calls.append((method, path, body, auth))
        return {
            "code": "0",
            "data": [
                {
                    "ordId": "okx-123",
                    "sCode": "0",
                    "sMsg": "",
                }
            ],
        }


def make_intent() -> OrderIntent:
    return OrderIntent(
        symbol="BTC-USDT-SWAP",
        side=Side.LONG,
        order_type="market",
        qty=0.01,
        leverage=5,
        reduce_only=False,
        entry_reason="unit_test",
        stop_loss=60000.0,
        max_slippage_bps=10.0,
    )


class ExecutionContractTest(unittest.TestCase):
    def test_paper_and_live_executors_return_same_order_result_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Ledger(os.path.join(tmp, "paper.sqlite3"))
            paper = PaperExecutor(
                ledger=ledger,
                paper_config=PaperConfig(initial_equity_usdt=10_000, fee_bps=5, slippage_bps=10),
                price_provider=lambda symbol: 62_000.0,
            )
            paper_result = paper.place_order(make_intent())

            transport = MockOkxTransport()
            live = OkxLiveExecutor(
                config=ExecutionConfig(
                    mode="live",
                    exchange="okx",
                    executor="live_okx",
                    allow_live_orders=True,
                ),
                transport=transport,
            )
            live_result = live.place_order(make_intent())

            self.assertIsInstance(paper_result, OrderResult)
            self.assertIsInstance(live_result, OrderResult)
            self.assertEqual(paper_result.status, "filled")
            self.assertEqual(live_result.status, "accepted")
            self.assertEqual(live_result.exchange_order_id, "okx-123")
            self.assertEqual(transport.calls[0][1], "/api/v5/trade/order")

    def test_live_executor_refuses_without_explicit_safety_switch(self):
        transport = MockOkxTransport()
        live = OkxLiveExecutor(
            config=ExecutionConfig(mode="live", exchange="okx", executor="live_okx", allow_live_orders=False),
            transport=transport,
        )

        with self.assertRaises(PermissionError):
            live.place_order(make_intent())

        self.assertEqual(transport.calls, [])


if __name__ == "__main__":
    unittest.main()

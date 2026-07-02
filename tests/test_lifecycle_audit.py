import os
import tempfile
import unittest

from langlang_trader.execution.paper import PaperExecutor
from langlang_trader.ledger import Ledger
from langlang_trader.lifecycle_audit import audit_maker_ledger, audit_signal_ledger, repair_signal_ledger
from langlang_trader.models import OrderIntent, Side
from langlang_trader.config import PaperConfig
from liangh_trader.market_maker.ledger import MarketMakerLedger
from liangh_trader.market_maker.models import LimitOrderState
from tests.test_market_maker_scaffold import write_config
from liangh_trader.market_maker.config import load_market_maker_config


class LifecycleAuditTest(unittest.TestCase):
    def test_signal_audit_fails_on_duplicate_open_lifecycle_and_repair_closes_stale_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "fleet.sqlite3")
            ledger = Ledger(path).scoped(run_id="audit-run", bot_id="audit-bot", variant_id="audit-var")
            executor = PaperExecutor(
                ledger=ledger,
                paper_config=PaperConfig(initial_equity_usdt=10_000, fee_bps=0, slippage_bps=0),
                price_provider=lambda symbol: 100.0,
            )
            executor.place_order(
                OrderIntent(
                    symbol="TEST-USDT-SWAP",
                    side=Side.LONG,
                    order_type="market",
                    qty=1.0,
                    leverage=3,
                    reduce_only=False,
                    entry_reason="current_long",
                    stop_loss=95.0,
                    max_slippage_bps=0.0,
                )
            )
            ledger.record_trade_fill(
                intent=OrderIntent(
                    symbol="TEST-USDT-SWAP",
                    side=Side.SHORT,
                    order_type="market",
                    qty=1.0,
                    leverage=3,
                    reduce_only=False,
                    entry_reason="stale_short",
                    stop_loss=105.0,
                    max_slippage_bps=0.0,
                    decision_trace={"exit_semantics": "full_tp_sl"},
                ),
                order_id=9,
                fill_id=9,
                price=100.0,
                fee=0.0,
            )

            before = audit_signal_ledger(path)
            dry_run = repair_signal_ledger(path, dry_run=True)
            applied = repair_signal_ledger(path, dry_run=False)
            after = audit_signal_ledger(path)

            self.assertGreater(before.critical_count, 0)
            self.assertGreater(dry_run.actions, 0)
            self.assertEqual(applied.actions, dry_run.actions)
            self.assertEqual(after.critical_count, 0)

    def test_maker_audit_detects_reused_order_id_but_uses_latest_order_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_market_maker_config(write_config(tmp))
            ledger = MarketMakerLedger(config.ledger_path, context=config.ledger_context)
            ledger.record_order_state(
                LimitOrderState(
                    order_id="mm-1",
                    quote_id="q-1",
                    symbol="BTCUSDT",
                    side="buy",
                    price=100.0,
                    qty=0.1,
                    remaining_qty=0.1,
                    status="open",
                    post_only=True,
                    created_at_ns=1_000,
                    updated_at_ns=1_000,
                    expires_at_ns=10_000_000_000_000,
                    strategy_version="market_maker_v1",
                    strategy_tree_variant_id="mm_v1_reference_passive_btcusdt",
                    strategy_tree_parent_id="market_maker_v1_root",
                    strategy_tree_path=["market_making"],
                )
            )
            ledger.record_order_state(
                LimitOrderState(
                    order_id="mm-1",
                    quote_id="q-1",
                    symbol="BTCUSDT",
                    side="buy",
                    price=100.0,
                    qty=0.1,
                    remaining_qty=0.0,
                    status="canceled",
                    post_only=True,
                    created_at_ns=1_000,
                    updated_at_ns=2_000,
                    expires_at_ns=10_000_000_000_000,
                    strategy_version="market_maker_v1",
                    strategy_tree_variant_id="mm_v1_reference_passive_btcusdt",
                    strategy_tree_parent_id="market_maker_v1_root",
                    strategy_tree_path=["market_making"],
                )
            )
            ledger.record_order_state(
                LimitOrderState(
                    order_id="mm-1",
                    quote_id="q-2",
                    symbol="BTCUSDT",
                    side="sell",
                    price=101.0,
                    qty=0.1,
                    remaining_qty=0.1,
                    status="open",
                    post_only=True,
                    created_at_ns=3_000,
                    updated_at_ns=3_000,
                    expires_at_ns=10_000_000_000_000,
                    strategy_version="market_maker_v1",
                    strategy_tree_variant_id="mm_v1_reference_passive_btcusdt",
                    strategy_tree_parent_id="market_maker_v1_root",
                    strategy_tree_path=["market_making"],
                )
            )

            report = audit_maker_ledger(config.ledger_path)

            self.assertIn("maker_order_id_reused", [finding.code for finding in report.findings])
            self.assertNotIn("maker_open_order_remaining_qty_invalid", [finding.code for finding in report.findings])


if __name__ == "__main__":
    unittest.main()

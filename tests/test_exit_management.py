import unittest

from langlang_trader.exit_management import (
    ExitActionType,
    ExitManagementContext,
    ExitManagementEngine,
)
from langlang_trader.models import Position, Side


def position(side=Side.LONG, qty=10.0, avg_price=100.0):
    return Position(
        symbol="TEST-USDT-SWAP",
        side=side,
        qty=qty,
        avg_price=avg_price,
        leverage=3,
        exchange="okx",
        strategy_version="rules_langlang_v1_3",
        regime="main_uptrend",
        setup="starter_buy",
    )


def context(**overrides):
    data = {
        "position": position(),
        "latest_price": 100.0,
        "entry_price": 100.0,
        "initial_stop_loss": 90.0,
        "current_stop_loss": 90.0,
        "initial_risk_usdt": 100.0,
        "mfe_usdt": 0.0,
        "partial_taken": False,
        "take_profit_plan": {"partial_r": 2.0, "partial_exit_fraction": 0.5, "runner_r": 4.0},
        "exit_profile": "partial_tp_trailing",
        "features": {},
        "fee_bps": 5.0,
        "slippage_bps": 5.0,
    }
    data.update(overrides)
    return ExitManagementContext(**data)


class ExitManagementEngineTest(unittest.TestCase):
    def test_one_r_moves_stop_to_fee_buffered_breakeven_without_closing(self):
        decision = ExitManagementEngine().evaluate(context(latest_price=110.1))

        self.assertEqual(decision.action, ExitActionType.MOVE_STOP)
        self.assertGreater(decision.new_stop_loss, 100.0)
        self.assertIsNone(decision.reduce_qty)
        self.assertIn("breakeven_stop_moved", decision.reason_codes)

    def test_two_r_partially_takes_profit_without_full_close(self):
        decision = ExitManagementEngine().evaluate(context(latest_price=120.1))

        self.assertEqual(decision.action, ExitActionType.PARTIAL_TAKE_PROFIT)
        self.assertAlmostEqual(decision.reduce_qty, 5.0)
        self.assertIsNotNone(decision.new_stop_loss)
        self.assertGreater(decision.new_stop_loss, 100.0)
        self.assertIn("partial_take_profit", decision.reason_codes)
        self.assertIn("breakeven_stop_moved", decision.reason_codes)

    def test_partial_profile_closes_runner_at_second_take_profit(self):
        decision = ExitManagementEngine().evaluate(
            context(
                latest_price=140.1,
                partial_taken=True,
                mfe_usdt=390.0,
                take_profit_plan={"partial_r": 2.0, "partial_exit_fraction": 0.5, "runner_r": 4.0},
            )
        )

        self.assertEqual(decision.action, ExitActionType.CLOSE_POSITION)
        self.assertEqual(decision.exit_reason, "runner_take_profit_exit")
        self.assertIn("runner_take_profit_exit", decision.reason_codes)

    def test_full_tp_profile_closes_entire_position_at_take_profit_without_partial(self):
        decision = ExitManagementEngine().evaluate(
            context(
                latest_price=112.1,
                exit_profile="full_tp_stop_loss",
                take_profit_plan={"take_profit_r": 1.2, "partial_r": 2.0, "partial_exit_fraction": 0.5},
            )
        )

        self.assertEqual(decision.action, ExitActionType.CLOSE_POSITION)
        self.assertEqual(decision.exit_reason, "take_profit_exit")
        self.assertIsNone(decision.reduce_qty)
        self.assertIn("take_profit_exit", decision.reason_codes)

    def test_mfe_trailing_waits_until_giveback_exceeds_wide_threshold(self):
        engine = ExitManagementEngine()

        held = engine.evaluate(context(latest_price=125.0, current_stop_loss=100.1, mfe_usdt=400.0, partial_taken=True))
        closed = engine.evaluate(context(latest_price=119.0, current_stop_loss=100.1, mfe_usdt=400.0, partial_taken=True))

        self.assertEqual(held.action, ExitActionType.HOLD)
        self.assertEqual(closed.action, ExitActionType.CLOSE_POSITION)
        self.assertIn("mfe_trailing_exit", closed.reason_codes)

    def test_wyckoff_or_pattern_risk_only_tightens_runner_exit(self):
        decision = ExitManagementEngine().evaluate(
            context(
                latest_price=125.0,
                current_stop_loss=100.1,
                mfe_usdt=400.0,
                partial_taken=True,
                features={"wyckoff_risk_score": 0.72, "risk_pattern_tag": "five_wave_late_risk"},
            )
        )

        self.assertEqual(decision.action, ExitActionType.CLOSE_POSITION)
        self.assertIn("wyckoff_exit_tightened", decision.reason_codes)
        self.assertIn("mfe_trailing_exit", decision.reason_codes)

    def test_missing_realtime_price_holds_with_data_quality_flag(self):
        decision = ExitManagementEngine().evaluate(context(latest_price=None))

        self.assertEqual(decision.action, ExitActionType.HOLD)
        self.assertIn("missing_realtime_exit_price", decision.data_quality_flags)


if __name__ == "__main__":
    unittest.main()

import unittest

from langlang_trader.config import RiskConfig
from langlang_trader.models import AccountSnapshot, LangLangSignal, MarketRegime, EntrySetup, Position, Side
from langlang_trader.position_sizing import LangLangPositionSizer
from langlang_trader.risk import RiskEngine


class RiskEnginePositionSizingTest(unittest.TestCase):
    def test_risk_engine_uses_langlang_position_sizer_for_qty_leverage_and_trace(self):
        signal = LangLangSignal(
            symbol="SOL-USDT-SWAP",
            side=Side.LONG,
            strength=0.8,
            reason_codes=["starter_buy"],
            filter_codes=[],
            features={"position_size_multiplier": 1.0, "stop_loss_cluster_24h": 0},
            invalidation_price=95.0,
            stop_loss=95.0,
            take_profit_hint=120.0,
            take_profit_plan={},
            hold_plan={},
            strategy_version="rules_langlang_native_final",
            regime=MarketRegime.PRE_MAIN_UPTREND,
            setup=EntrySetup.STARTER_BUY,
            decision_trace={
                "entry_position_id": "1_startup_long",
                "market_season": "summer",
                "position_size_multiplier": 1.0,
            },
        )
        config = RiskConfig(
            position_sizing_mode="langlang_w_unit",
            active_capital_fraction=0.30,
            max_position_usdt=5_000,
            max_total_position_usdt=25_000,
            alt_leverage=5,
            reference_leverage=10,
            max_daily_loss_usdt=500,
        )
        engine = RiskEngine(config, position_sizer=LangLangPositionSizer(config, initial_equity_usdt=10_000))

        intent = engine.intent_from_signal(
            signal=signal,
            account=AccountSnapshot(equity_usdt=10_000, cash_usdt=10_000, margin_used_usdt=0),
            latest_price=100.0,
            open_positions=[],
        )

        self.assertIsNotNone(intent)
        self.assertEqual(intent.leverage, 5)
        self.assertAlmostEqual(intent.qty, 50.0)
        self.assertAlmostEqual(intent.decision_trace["risk_unit_w_usdt"], 1_000.0)
        self.assertAlmostEqual(intent.decision_trace["position_margin_usdt"], 1_000.0)
        self.assertAlmostEqual(intent.decision_trace["position_notional_usdt"], 5_000.0)

    def test_risk_engine_records_rejection_reason(self):
        config = RiskConfig(max_open_positions=1)
        engine = RiskEngine(config)
        blocked = engine.intent_from_signal(
            signal=LangLangSignal(
                symbol="SOL-USDT-SWAP",
                side=Side.LONG,
                strength=0.8,
                reason_codes=["starter_buy"],
                filter_codes=[],
                features={},
                invalidation_price=95.0,
                stop_loss=95.0,
                take_profit_hint=120.0,
                take_profit_plan={},
                hold_plan={},
                strategy_version="rules_langlang_native_final",
                regime=MarketRegime.PRE_MAIN_UPTREND,
                setup=EntrySetup.STARTER_BUY,
                decision_trace={},
            ),
            account=AccountSnapshot(equity_usdt=10_000, cash_usdt=10_000, margin_used_usdt=0),
            latest_price=100.0,
            open_positions=[Position(symbol="BTC-USDT-SWAP", side=Side.LONG, qty=1, avg_price=100, leverage=5)],
        )

        self.assertIsNone(blocked)
        self.assertEqual(engine.last_rejection_reason, "max_open_positions")
        self.assertEqual(engine.last_rejection_trace["open_count"], 1)


if __name__ == "__main__":
    unittest.main()

import unittest

from langlang_trader.config import RiskConfig
from langlang_trader.models import AccountSnapshot, LangLangSignal, MarketRegime, EntrySetup, Side
from langlang_trader.position_sizing import LangLangPositionSizer


def signal(
    *,
    symbol="WAVE-USDT-SWAP",
    side=Side.LONG,
    entry_position_id="1_startup_long",
    market_season="summer",
    setup=EntrySetup.STARTER_BUY,
    features=None,
    decision_trace=None,
):
    merged_features = {"position_size_multiplier": 1.0, "stop_loss_cluster_24h": 0}
    if features:
        merged_features.update(features)
    merged_trace = {
        "entry_position_id": entry_position_id,
        "market_season": market_season,
        "position_size_multiplier": merged_features["position_size_multiplier"],
    }
    if decision_trace:
        merged_trace.update(decision_trace)
    return LangLangSignal(
        symbol=symbol,
        side=side,
        strength=0.8,
        reason_codes=["unit_test"],
        filter_codes=[],
        features=merged_features,
        invalidation_price=95.0,
        stop_loss=95.0,
        take_profit_hint=120.0,
        take_profit_plan={},
        hold_plan={},
        strategy_version="rules_langlang_native_final",
        regime=MarketRegime.PRE_MAIN_UPTREND,
        setup=setup,
        decision_trace=merged_trace,
    )


class LangLangPositionSizerTest(unittest.TestCase):
    def test_document_entry_positions_map_to_w_unit_margins_and_leverage(self):
        sizer = LangLangPositionSizer(
            RiskConfig(
                position_sizing_mode="langlang_w_unit",
                active_capital_fraction=0.30,
                max_position_usdt=5_000,
                max_total_position_usdt=25_000,
                alt_leverage=5,
                reference_leverage=10,
            ),
            initial_equity_usdt=10_000,
        )
        account = AccountSnapshot(equity_usdt=10_000, cash_usdt=10_000, margin_used_usdt=0)

        cases = [
            ("1_startup_long", Side.LONG, "W", 1_000.0, 5, 5_000.0),
            ("4_second_wave_long", Side.LONG, "W", 1_000.0, 5, 5_000.0),
            ("2_small_divergence_low_pullback", Side.LONG, "0.6W", 600.0, 5, 3_000.0),
            ("6_box_rebound_long", Side.LONG, "0.25W", 250.0, 5, 1_250.0),
            ("3_first_large_divergence_top_short", Side.SHORT, "0.15W", 150.0, 5, 750.0),
            ("short_waterfall_continuation", Side.SHORT, "0.45W", 450.0, 5, 2_250.0),
        ]

        for entry_position_id, side, risk_unit, margin, leverage, notional in cases:
            with self.subTest(entry_position_id=entry_position_id):
                decision = sizer.size(
                    signal=signal(entry_position_id=entry_position_id, side=side),
                    account=account,
                    open_positions=[],
                    latest_price=100.0,
                )
                self.assertIsNotNone(decision)
                self.assertEqual(decision.risk_unit, risk_unit)
                self.assertAlmostEqual(decision.margin_usdt, margin)
                self.assertEqual(decision.leverage, leverage)
                self.assertAlmostEqual(decision.notional_usdt, notional)

    def test_market_season_stop_cluster_reference_leverage_and_step_capital(self):
        sizer = LangLangPositionSizer(
            RiskConfig(
                position_sizing_mode="langlang_w_unit",
                active_capital_fraction=0.30,
                max_position_usdt=20_000,
                max_total_position_usdt=100_000,
                alt_leverage=5,
                reference_leverage=10,
            ),
            initial_equity_usdt=10_000,
        )

        spring = sizer.size(
            signal=signal(symbol="BTC-USDT-SWAP", market_season="spring"),
            account=AccountSnapshot(equity_usdt=10_000, cash_usdt=10_000, margin_used_usdt=0),
            open_positions=[],
            latest_price=100.0,
        )
        self.assertIsNotNone(spring)
        self.assertEqual(spring.leverage, 10)
        self.assertAlmostEqual(spring.margin_usdt, 700.0)
        self.assertAlmostEqual(spring.notional_usdt, 7_000.0)

        stepped = sizer.size(
            signal=signal(market_season="summer"),
            account=AccountSnapshot(equity_usdt=20_500, cash_usdt=20_500, margin_used_usdt=0),
            open_positions=[],
            latest_price=100.0,
        )
        self.assertIsNotNone(stepped)
        self.assertEqual(stepped.capital_step_level, 2)
        self.assertAlmostEqual(stepped.risk_unit_w_usdt, 2_000.0)

        cooled = sizer.size(
            signal=signal(features={"stop_loss_cluster_24h": 2}),
            account=AccountSnapshot(equity_usdt=10_000, cash_usdt=10_000, margin_used_usdt=0),
            open_positions=[],
            latest_price=100.0,
        )
        self.assertIsNotNone(cooled)
        self.assertAlmostEqual(cooled.margin_usdt, 350.0)
        self.assertIn("stop_loss_cluster_reduce", cooled.reason_codes)

        rejected = sizer.size(
            signal=signal(features={"stop_loss_cluster_24h": 3}),
            account=AccountSnapshot(equity_usdt=10_000, cash_usdt=10_000, margin_used_usdt=0),
            open_positions=[],
            latest_price=100.0,
        )
        self.assertIsNone(rejected)


if __name__ == "__main__":
    unittest.main()

import unittest

from langlang_trader.models import Candle, OrderBook, OrderBookLevel, Side
from langlang_trader.scalping import (
    detect_five_bar_fractal,
    EntryMode,
    FiveBarScalpConfig,
    FiveBarScalpStrategy,
    RulesFiveBarScalpStrategy,
    ScalpTrade,
    ScalpingVariant,
    OrderFlowWindow,
    summarize_scalp_trades,
)


def candle(idx, open_, high, low, close, *, volume=1000):
    return Candle(
        symbol="BTC-USDT-SWAP",
        bar="5s",
        ts=1_700_000_000_000 + idx * 5_000,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


def book(*, bid=100.0, ask=100.03, bid_qty=20.0, ask_qty=10.0):
    return OrderBook(
        symbol="BTC-USDT-SWAP",
        ts=1_700_000_030_000,
        bids=[OrderBookLevel(bid, bid_qty)],
        asks=[OrderBookLevel(ask, ask_qty)],
    )


class FiveBarFractalScalpStrategyTest(unittest.TestCase):
    def test_rejects_bullish_fractal_when_left_lows_do_not_step_down(self):
        fractal = detect_five_bar_fractal(
            [
                candle(0, 100.0, 100.4, 99.4, 100.0),
                candle(1, 100.0, 100.3, 99.6, 99.8),
                candle(2, 99.8, 100.0, 98.0, 98.7),
                candle(3, 98.7, 99.7, 98.4, 99.5),
                candle(4, 99.5, 100.1, 98.8, 99.9),
            ]
        )

        self.assertIsNone(fractal)

    def test_rejects_bearish_fractal_when_left_highs_do_not_step_up(self):
        fractal = detect_five_bar_fractal(
            [
                candle(0, 100.0, 100.6, 99.7, 100.2),
                candle(1, 100.2, 100.4, 99.8, 100.1),
                candle(2, 100.1, 102.0, 100.0, 101.5),
                candle(3, 101.5, 101.5, 100.0, 100.4),
                candle(4, 100.4, 101.0, 99.7, 100.0),
            ]
        )

        self.assertIsNone(fractal)

    def test_bullish_pullback_fractal_waits_for_breakout_with_order_flow_confirmation(self):
        strategy = FiveBarScalpStrategy(FiveBarScalpConfig(total_cost_bps=4.0, stop_buffer_bps=2.0, max_stop_bps=300.0))
        candles = [
            candle(0, 100.0, 100.5, 99.6, 100.2),
            candle(1, 100.2, 100.3, 98.8, 99.1),
            candle(2, 99.1, 99.4, 97.8, 98.6),
            candle(3, 98.6, 99.8, 98.3, 99.5),
            candle(4, 99.5, 100.1, 98.9, 99.9),
        ]
        order_flow = OrderFlowWindow(
            buy_volume=1800,
            sell_volume=900,
            bid_depth_change=0.18,
            ask_depth_change=-0.12,
            latest_spread_bps=3.0,
        )

        decision = strategy.evaluate(
            symbol="BTC-USDT-SWAP",
            scalp_candles=candles,
            trend_candles_by_bar={
                "1m": [candle(10, 99, 101, 98, 100), candle(11, 100, 102, 99, 101.5)],
                "3m": [candle(12, 98, 100, 97, 99), candle(13, 99, 102, 98, 101)],
                "5m": [candle(14, 97, 100, 96, 98), candle(15, 98, 102, 97, 101)],
            },
            order_flow=order_flow,
            order_book=book(bid=99.98, ask=100.01, bid_qty=24.0, ask_qty=9.0),
        )

        self.assertEqual(decision.action, "enter")
        self.assertIsNotNone(decision.signal)
        self.assertEqual(decision.signal.side, Side.LONG)
        self.assertEqual(decision.signal.entry_trigger, 100.1)
        self.assertLess(decision.signal.stop_loss, 97.8)
        self.assertGreaterEqual(decision.signal.take_profit, 100.1 + 3 * strategy.config.total_cost_bps / 10_000 * 100.1)
        self.assertIn("bullish_5_bar_fractal", decision.signal.reason_codes)
        self.assertIn("order_flow_reclaim_confirmed", decision.signal.reason_codes)
        self.assertEqual(decision.signal.entry_mode, EntryMode.BREAKOUT.value)
        self.assertEqual(
            decision.signal.features["stop_loss"]["policy"],
            "fractal_extreme_buffer_with_min_max_risk",
        )
        self.assertGreaterEqual(decision.signal.features["stop_loss"]["risk_bps"], strategy.config.min_stop_bps)
        self.assertLessEqual(decision.signal.features["stop_loss"]["risk_bps"], strategy.config.max_stop_bps)

    def test_bearish_pullback_fractal_waits_for_breakdown_with_order_flow_confirmation(self):
        strategy = FiveBarScalpStrategy(FiveBarScalpConfig(total_cost_bps=4.0, stop_buffer_bps=2.0, max_stop_bps=300.0))
        candles = [
            candle(0, 100.0, 100.4, 99.6, 99.8),
            candle(1, 99.8, 101.0, 99.6, 100.7),
            candle(2, 100.7, 102.2, 100.3, 101.5),
            candle(3, 101.5, 101.7, 100.0, 100.4),
            candle(4, 100.4, 101.1, 99.7, 100.0),
        ]
        order_flow = OrderFlowWindow(
            buy_volume=700,
            sell_volume=1500,
            bid_depth_change=-0.16,
            ask_depth_change=0.14,
            latest_spread_bps=3.0,
        )

        decision = strategy.evaluate(
            symbol="BTC-USDT-SWAP",
            scalp_candles=candles,
            trend_candles_by_bar={
                "1m": [candle(10, 101, 102, 99, 100), candle(11, 100, 101, 98, 98.6)],
                "3m": [candle(12, 103, 104, 100, 102), candle(13, 102, 103, 97, 99)],
                "5m": [candle(14, 104, 105, 101, 103), candle(15, 103, 104, 98, 100)],
            },
            order_flow=order_flow,
            order_book=book(bid=99.98, ask=100.01, bid_qty=8.0, ask_qty=22.0),
        )

        self.assertEqual(decision.action, "enter")
        self.assertIsNotNone(decision.signal)
        self.assertEqual(decision.signal.side, Side.SHORT)
        self.assertEqual(decision.signal.entry_trigger, 99.7)
        self.assertGreater(decision.signal.stop_loss, 102.2)
        self.assertIn("bearish_5_bar_fractal", decision.signal.reason_codes)
        self.assertIn("order_flow_breakdown_confirmed", decision.signal.reason_codes)

    def test_rejects_fractal_when_higher_timeframes_disagree(self):
        strategy = FiveBarScalpStrategy()
        decision = strategy.evaluate(
            symbol="BTC-USDT-SWAP",
            scalp_candles=[
                candle(0, 100.0, 100.5, 99.6, 100.2),
                candle(1, 100.2, 100.3, 98.8, 99.1),
                candle(2, 99.1, 99.4, 97.8, 98.6),
                candle(3, 98.6, 99.8, 98.3, 99.5),
                candle(4, 99.5, 100.1, 98.9, 99.9),
            ],
            trend_candles_by_bar={"1m": [candle(10, 101, 102, 99, 100), candle(11, 100, 101, 97, 98)]},
            order_flow=OrderFlowWindow(1500, 700, 0.2, -0.1, 2.0),
            order_book=book(),
        )

        self.assertEqual(decision.action, "skip")
        self.assertIn("trend_not_aligned", decision.filter_codes)

    def test_rejects_when_order_flow_confirmation_is_missing(self):
        strategy = FiveBarScalpStrategy()
        decision = strategy.evaluate(
            symbol="BTC-USDT-SWAP",
            scalp_candles=[
                candle(0, 100.0, 100.5, 99.6, 100.2),
                candle(1, 100.2, 100.3, 98.8, 99.1),
                candle(2, 99.1, 99.4, 97.8, 98.6),
                candle(3, 98.6, 99.8, 98.3, 99.5),
                candle(4, 99.5, 100.1, 98.9, 99.9),
            ],
            trend_candles_by_bar={"1m": [candle(10, 99, 101, 98, 100), candle(11, 100, 102, 99, 101)]},
            order_flow=OrderFlowWindow(900, 1100, -0.05, 0.06, 2.0),
            order_book=book(),
        )

        self.assertEqual(decision.action, "skip")
        self.assertIn("order_flow_not_confirmed", decision.filter_codes)

    def test_can_disable_order_flow_for_ablation_but_keeps_cost_and_spread_filters(self):
        strategy = FiveBarScalpStrategy(
            FiveBarScalpConfig(
                require_order_flow=False,
                max_spread_bps=5.0,
                min_range_cost_multiple=3.0,
                max_stop_bps=300.0,
            )
        )
        candles = [
            candle(0, 100.0, 100.5, 99.6, 100.2),
            candle(1, 100.2, 100.3, 98.8, 99.1),
            candle(2, 99.1, 99.4, 97.8, 98.6),
            candle(3, 98.6, 99.8, 98.3, 99.5),
            candle(4, 99.5, 100.1, 98.9, 99.9),
        ]

        decision = strategy.evaluate(
            symbol="BTC-USDT-SWAP",
            scalp_candles=candles,
            trend_candles_by_bar={"1m": [candle(10, 99, 101, 98, 100), candle(11, 100, 102, 99, 101)]},
            order_flow=OrderFlowWindow(900, 1100, -0.05, 0.06, 3.0),
            order_book=book(),
        )

        self.assertEqual(decision.action, "enter")
        self.assertIn("order_flow_ablation_mode", decision.signal.reason_codes)

    def test_weak_order_flow_allows_imbalance_confirmation_without_order_book(self):
        strategy = FiveBarScalpStrategy(
            FiveBarScalpConfig(
                order_flow_mode="weak",
                max_stop_bps=300.0,
                min_order_flow_imbalance=0.20,
            )
        )

        decision = strategy.evaluate(
            symbol="BTC-USDT-SWAP",
            scalp_candles=[
                candle(0, 100.0, 100.5, 99.6, 100.2),
                candle(1, 100.2, 100.3, 98.8, 99.1),
                candle(2, 99.1, 99.4, 97.8, 98.6),
                candle(3, 98.6, 99.8, 98.3, 99.5),
                candle(4, 99.5, 100.1, 98.9, 99.9),
            ],
            trend_candles_by_bar={"1m": [candle(10, 99, 101, 98, 100), candle(11, 100, 102, 99, 101)]},
            order_flow=OrderFlowWindow(1800, 900, 0.0, 0.0, 2.0),
            order_book=None,
        )

        self.assertEqual(decision.action, "enter")
        self.assertEqual(decision.signal.features["order_flow"]["tier"], "weak")
        self.assertIn("order_flow_weak_confirmed", decision.signal.reason_codes)

    def test_fractal_confirm_entry_mode_enters_on_last_close_instead_of_breakout(self):
        strategy = FiveBarScalpStrategy(
            FiveBarScalpConfig(
                entry_mode=EntryMode.FRACTAL_CONFIRM,
                order_flow_mode="weak",
                max_stop_bps=300.0,
            )
        )
        candles = [
            candle(0, 100.0, 100.5, 99.6, 100.2),
            candle(1, 100.2, 100.3, 98.8, 99.1),
            candle(2, 99.1, 99.4, 97.8, 98.6),
            candle(3, 98.6, 99.8, 98.3, 99.5),
            candle(4, 99.5, 100.1, 98.9, 99.9),
        ]

        decision = strategy.evaluate(
            symbol="BTC-USDT-SWAP",
            scalp_candles=candles,
            trend_candles_by_bar={"1m": [candle(10, 99, 101, 98, 100), candle(11, 100, 102, 99, 101)]},
            order_flow=OrderFlowWindow(1800, 900, 0.0, 0.0, 2.0),
            order_book=None,
        )

        self.assertEqual(decision.action, "enter")
        self.assertEqual(decision.signal.entry_trigger, 99.9)
        self.assertNotEqual(decision.signal.entry_trigger, 100.1)
        self.assertIn("entry_mode_fractal_confirm", decision.signal.reason_codes)

    def test_rejects_when_structure_stop_is_too_wide_for_scalping(self):
        strategy = FiveBarScalpStrategy(FiveBarScalpConfig(max_stop_bps=10.0))

        decision = strategy.evaluate(
            symbol="BTC-USDT-SWAP",
            scalp_candles=[
                candle(0, 100.0, 100.5, 99.6, 100.2),
                candle(1, 100.2, 100.3, 98.8, 99.1),
                candle(2, 99.1, 99.4, 97.8, 98.6),
                candle(3, 98.6, 99.8, 98.3, 99.5),
                candle(4, 99.5, 100.1, 98.9, 99.9),
            ],
            trend_candles_by_bar={"1m": [candle(10, 99, 101, 98, 100), candle(11, 100, 102, 99, 101)]},
            order_flow=OrderFlowWindow(1800, 900, 0.18, -0.12, 3.0),
            order_book=book(bid=99.98, ask=100.01, bid_qty=24.0, ask_qty=9.0),
        )

        self.assertEqual(decision.action, "skip")
        self.assertIn("stop_distance_too_wide", decision.filter_codes)

    def test_rules_adapter_returns_paper_signal_with_stop_loss(self):
        strategy = RulesFiveBarScalpStrategy(
            ScalpingVariant(
                variant_id="scalp_BTC_5s",
                symbol="BTC-USDT-SWAP",
                scalp_bar="5s",
                min_order_flow_imbalance=0.05,
                max_stop_bps=300.0,
            )
        )
        candles = [
            candle(0, 100.0, 100.5, 99.6, 100.2, volume=800),
            candle(1, 100.2, 100.3, 98.8, 99.1, volume=900),
            candle(2, 99.1, 99.4, 97.8, 98.6, volume=1200),
            candle(3, 98.6, 99.8, 98.3, 99.5, volume=1800),
            candle(4, 99.5, 100.1, 98.9, 99.9, volume=2000),
        ]

        signal = strategy.generate_from_market_data(
            symbol="BTC-USDT-SWAP",
            candles_by_bar={
                "5s": candles,
                "1m": [candle(10, 99, 101, 98, 100), candle(11, 100, 102, 99, 101.5)],
            },
            order_book=book(bid=99.98, ask=100.01, bid_qty=24.0, ask_qty=9.0),
        )

        self.assertIsNotNone(signal)
        self.assertLess(signal.invalidation_price, signal.features["entry_trigger"])
        self.assertGreater(signal.take_profit_hint, signal.features["entry_trigger"])
        self.assertEqual(signal.features["scalp_bar"], "5s")


class ScalpBacktestSummaryTest(unittest.TestCase):
    def test_summary_deducts_full_costs_and_reports_profit_factor_and_drawdown(self):
        trades = [
            ScalpTrade(
                symbol="BTC-USDT-SWAP",
                side=Side.LONG,
                entry_price=100.0,
                exit_price=100.4,
                fee_bps=4.0,
                spread_bps=2.0,
                slippage_bps=1.0,
                bucket="trend",
            ),
            ScalpTrade(
                symbol="BTC-USDT-SWAP",
                side=Side.SHORT,
                entry_price=100.0,
                exit_price=100.3,
                fee_bps=4.0,
                spread_bps=2.0,
                slippage_bps=1.0,
                bucket="trend",
            ),
            ScalpTrade(
                symbol="ETH-USDT-SWAP",
                side=Side.SHORT,
                entry_price=100.0,
                exit_price=99.6,
                fee_bps=4.0,
                spread_bps=2.0,
                slippage_bps=1.0,
                bucket="range",
            ),
        ]

        summary = summarize_scalp_trades(trades)

        self.assertEqual(summary["trade_count"], 3)
        self.assertAlmostEqual(summary["net_expectancy_bps"], (33.0 - 37.0 + 33.0) / 3)
        self.assertAlmostEqual(summary["profit_factor"], 66.0 / 37.0)
        self.assertAlmostEqual(summary["max_drawdown_bps"], 37.0)
        self.assertEqual(summary["by_symbol"]["BTC-USDT-SWAP"]["trade_count"], 2)
        self.assertAlmostEqual(summary["by_bucket"]["trend"]["net_expectancy_bps"], -2.0)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import os
import tempfile
import unittest

from langlang_trader.config import ExecutionConfig, MarketDataConfig, PaperConfig, RiskConfig, UniverseConfig
from langlang_trader.fleet import BotConfig, FleetConfig, FleetRunner
from langlang_trader.ledger import Ledger
from langlang_trader.market_data import StaticMarketData
from langlang_trader.micro_scalping import (
    MicroScalpVariant,
    RulesFundingBasisShadowStrategy,
    RulesOfiMicropriceScalpStrategy,
    RulesVolatilityBreakoutScalpStrategy,
    RulesVwapMeanReversionScalpStrategy,
)
from langlang_trader.models import Candle, OrderBook, OrderBookLevel, Ticker


def _book(symbol: str = "BTC-USDT-SWAP", *, bid_qty: float = 40.0, ask_qty: float = 5.0) -> OrderBook:
    return OrderBook(
        symbol=symbol,
        ts=1_700_000_050_000,
        bids=[OrderBookLevel(price=100.00, qty=bid_qty)],
        asks=[OrderBookLevel(price=100.04, qty=ask_qty)],
    )


def _candles(symbol: str = "BTC-USDT-SWAP", bar: str = "5s", closes: list[float] | None = None) -> list[Candle]:
    closes = closes or [100.00, 100.01, 100.02, 100.03, 100.04, 100.05, 100.06, 100.08]
    rows: list[Candle] = []
    for idx, close in enumerate(closes):
        open_ = close - 0.02
        rows.append(
            Candle(
                symbol=symbol,
                bar=bar,
                ts=1_700_000_000_000 + idx * 5_000,
                open=open_,
                high=max(open_, close) + 0.02,
                low=min(open_, close) - 0.02,
                close=close,
                volume=1_000.0 + idx * 50,
            )
        )
    return rows


class MicroScalpingStrategyTest(unittest.TestCase):
    def test_ofi_microprice_signal_has_tree_trace_and_hard_stop(self):
        variant = MicroScalpVariant(
            variant_id="scalp_ofi_micro_btc_5s_v1",
            symbol="BTC-USDT-SWAP",
            strategy_kind="ofi_microprice_directional",
            min_ofi=0.35,
            min_microprice_edge_bps=0.5,
            stop_bps=10.0,
            strategy_tree_parent_id="scalp_ofi_microprice_directional_v1",
            strategy_tree_path=("scalping", "scalp_ofi_microprice_directional_v1", "scalp_ofi_micro_btc_5s_v1"),
        )

        signal = RulesOfiMicropriceScalpStrategy(variant).generate_from_market_data(
            symbol="BTC-USDT-SWAP",
            candles_by_bar={"5s": _candles()},
            order_book=_book(),
        )

        self.assertIsNotNone(signal)
        assert signal is not None
        self.assertEqual(signal.side.value, "long")
        self.assertLess(signal.invalidation_price, 100.08)
        self.assertGreater(signal.take_profit_hint, 100.08)
        self.assertEqual(signal.features["strategy_tree_variant_id"], "scalp_ofi_micro_btc_5s_v1")
        self.assertEqual(signal.features["time_stop_bars"], 10)
        self.assertIn("strict_hard_stop_bps", signal.features["risk"])

    def test_vwap_mean_reversion_shorts_extended_move_with_stop(self):
        closes = [100.0, 100.02, 99.98, 100.01, 100.00, 100.03, 100.04, 101.0]
        variant = MicroScalpVariant(
            variant_id="scalp_vwap_mr_btc_15s_v1",
            symbol="BTC-USDT-SWAP",
            strategy_kind="vwap_mean_reversion",
            bar="15s",
            vwap_deviation_bps=20.0,
            stop_bps=14.0,
        )

        signal = RulesVwapMeanReversionScalpStrategy(variant).generate_from_market_data(
            symbol="BTC-USDT-SWAP",
            candles_by_bar={"15s": _candles(bar="15s", closes=closes)},
            order_book=_book(bid_qty=5.0, ask_qty=40.0),
        )

        self.assertIsNotNone(signal)
        assert signal is not None
        self.assertEqual(signal.side.value, "short")
        self.assertGreater(signal.invalidation_price, 101.0)
        self.assertLess(signal.take_profit_hint, 101.0)
        self.assertEqual(signal.features["strategy_kind"], "vwap_mean_reversion")

    def test_vwap_mean_reversion_rejects_strong_pre_signal_trend(self):
        closes = [100.0, 100.12, 100.24, 100.36, 100.48, 100.60, 100.72, 101.20]
        variant = MicroScalpVariant(
            variant_id="scalp_vwap_mr_btc_15s_v1",
            symbol="BTC-USDT-SWAP",
            strategy_kind="vwap_mean_reversion",
            bar="15s",
            vwap_deviation_bps=20.0,
            stop_bps=14.0,
        )

        signal = RulesVwapMeanReversionScalpStrategy(variant).generate_from_market_data(
            symbol="BTC-USDT-SWAP",
            candles_by_bar={"15s": _candles(bar="15s", closes=closes)},
            order_book=_book(bid_qty=5.0, ask_qty=40.0),
        )

        self.assertIsNone(signal)

    def test_volatility_breakout_enters_after_compression_and_volume_expansion(self):
        closes = [100.00, 100.01, 100.00, 100.02, 99.99, 100.01, 100.00, 100.45]
        rows = _candles(closes=closes)
        rows[-1] = Candle(
            rows[-1].symbol,
            rows[-1].bar,
            rows[-1].ts,
            100.04,
            100.50,
            100.03,
            100.45,
            4_000.0,
        )
        variant = MicroScalpVariant(
            variant_id="scalp_vol_breakout_btc_5s_v1",
            symbol="BTC-USDT-SWAP",
            strategy_kind="volatility_breakout",
            breakout_lookback_bars=6,
            min_volume_ratio=1.5,
            stop_bps=16.0,
        )

        signal = RulesVolatilityBreakoutScalpStrategy(variant).generate_from_market_data(
            symbol="BTC-USDT-SWAP",
            candles_by_bar={"5s": rows},
            order_book=_book(),
        )

        self.assertIsNotNone(signal)
        assert signal is not None
        self.assertEqual(signal.side.value, "long")
        self.assertLess(signal.invalidation_price, 100.45)
        self.assertEqual(signal.features["breakout_lookback_bars"], 6)
        self.assertLessEqual(
            signal.features["previous_range_bps"],
            signal.features["compression_threshold_bps"],
        )

    def test_volatility_breakout_rejects_when_previous_window_is_not_compressed(self):
        rows = [
            Candle("BTC-USDT-SWAP", "5s", 1_700_000_000_000 + idx * 5_000, 100.0, 101.0, 99.0, 100.0, 1_000.0)
            for idx in range(7)
        ]
        rows.append(
            Candle(
                "BTC-USDT-SWAP",
                "5s",
                1_700_000_035_000,
                101.05,
                101.50,
                101.00,
                101.40,
                4_000.0,
            )
        )
        variant = MicroScalpVariant(
            variant_id="scalp_vol_breakout_btc_5s_v1",
            symbol="BTC-USDT-SWAP",
            strategy_kind="volatility_breakout",
            breakout_lookback_bars=6,
            min_volume_ratio=1.5,
            stop_bps=16.0,
        )

        signal = RulesVolatilityBreakoutScalpStrategy(variant).generate_from_market_data(
            symbol="BTC-USDT-SWAP",
            candles_by_bar={"5s": rows},
            order_book=_book(),
        )

        self.assertIsNone(signal)

    def test_funding_basis_shadow_strategy_records_pair_plan_without_single_leg_signal(self):
        variant = MicroScalpVariant(
            variant_id="scalp_funding_basis_btc_v1",
            symbol="BTC-USDT-SWAP",
            strategy_kind="funding_basis_delta_neutral",
            min_basis_bps=8.0,
            pair_stop_bps=12.0,
        )

        pair = RulesFundingBasisShadowStrategy(variant).generate_shadow_pair_from_market_data(
            symbol="BTC-USDT-SWAP",
            market_metrics={
                "mark_price": 101.0,
                "index_price": 100.0,
                "basis_bps": 100.0,
                "funding_rate_last": 0.0004,
            },
        )
        signal = RulesFundingBasisShadowStrategy(variant).generate_from_market_data(
            symbol="BTC-USDT-SWAP",
            candles_by_bar={"5s": _candles()},
            order_book=_book(),
            market_metrics={"basis_bps": 100.0, "funding_rate_last": 0.0004},
        )

        self.assertIsNotNone(pair)
        assert pair is not None
        self.assertEqual(pair.perp_side, "short")
        self.assertEqual(pair.hedge_side, "long_spot_or_inverse")
        self.assertGreater(pair.stop_basis_bps, pair.basis_bps)
        self.assertLess(pair.take_profit_basis_bps, pair.basis_bps)
        self.assertIsNone(signal)


class MicroScalpingFleetTest(unittest.TestCase):
    def test_micro_scalp_bot_records_stop_loss_and_tree_trace(self):
        class MicroMarketData(StaticMarketData):
            def latest_price(self, symbol: str) -> float:
                return 100.08

            def get_ticker(self, symbol: str) -> Ticker:
                return Ticker(symbol=symbol, ts=1_700_000_060_000, last=100.08, bid=100.07, ask=100.09)

            def get_order_book(self, symbol: str, depth: int = 20) -> OrderBook:
                return _book(symbol)

        with tempfile.TemporaryDirectory() as tmp:
            variant = MicroScalpVariant(
                variant_id="scalp_ofi_micro_btc_5s_v1",
                symbol="BTC-USDT-SWAP",
                strategy_kind="ofi_microprice_directional",
                min_ofi=0.35,
                min_microprice_edge_bps=0.5,
            )
            config = FleetConfig(
                run_id="unit-micro-scalp-paper",
                strategy_version="scalp_ofi_microprice_directional_v1",
                execution=ExecutionConfig(mode="paper", exchange="binance", executor="paper_binance"),
                paper=PaperConfig(initial_equity_usdt=10_000, fee_bps=0, slippage_bps=0),
                risk=RiskConfig(max_position_usdt=100, max_open_positions=1, max_open_symbols=1),
                market_data=MarketDataConfig(symbols=["BTC-USDT-SWAP"], bars=["5s"], candle_limit=20),
                universe=UniverseConfig(mode="static", reference_symbols=[], snapshot_path=""),
                ledger_path=os.path.join(tmp, "fleet.sqlite3"),
                bots=[BotConfig(bot_id="micro-btc-5s", variant=variant, strategy_version="scalp_ofi_microprice_directional_v1")],
            )
            ledger = Ledger(config.ledger_path)

            cycle = FleetRunner(config=config, market_data=MicroMarketData({"BTC-USDT-SWAP": _candles()}), ledger=ledger).run_once()

            self.assertEqual(cycle["signals"], 1)
            self.assertEqual(cycle["intents"], 1)
            intents = ledger.list_rows("order_intents")
            self.assertGreater(intents[0]["stop_loss"], 0)
            trace = json.loads(intents[0]["decision_trace_json"])
            self.assertEqual(trace["strategy_tree_variant_id"], "scalp_ofi_micro_btc_5s_v1")
            self.assertEqual(trace["time_stop_bars"], 10)

    def test_funding_basis_bot_writes_shadow_pair_event_without_order_intent(self):
        class BasisMarketData(StaticMarketData):
            def latest_price(self, symbol: str) -> float:
                return 101.0

            def get_market_metrics(self, symbol: str) -> dict[str, float]:
                return {
                    "mark_price": 101.0,
                    "index_price": 100.0,
                    "basis_bps": 100.0,
                    "funding_rate_last": 0.0004,
                }

        with tempfile.TemporaryDirectory() as tmp:
            variant = MicroScalpVariant(
                variant_id="scalp_funding_basis_btc_v1",
                symbol="BTC-USDT-SWAP",
                strategy_kind="funding_basis_delta_neutral",
                min_basis_bps=8.0,
            )
            config = FleetConfig(
                run_id="unit-funding-basis-paper",
                strategy_version="scalp_funding_basis_delta_neutral_v1",
                execution=ExecutionConfig(mode="paper", exchange="binance", executor="paper_binance"),
                market_data=MarketDataConfig(symbols=["BTC-USDT-SWAP"], bars=["5s"], candle_limit=20),
                universe=UniverseConfig(mode="static", reference_symbols=[], snapshot_path=""),
                ledger_path=os.path.join(tmp, "fleet.sqlite3"),
                bots=[BotConfig(bot_id="basis-btc", variant=variant, strategy_version="scalp_funding_basis_delta_neutral_v1")],
            )
            ledger = Ledger(config.ledger_path)

            cycle = FleetRunner(config=config, market_data=BasisMarketData({"BTC-USDT-SWAP": _candles()}), ledger=ledger).run_once()

            self.assertEqual(cycle["shadow_pair_events"], 1)
            self.assertEqual(ledger.list_rows("order_intents"), [])
            events = ledger.list_rows("shadow_pair_events")
            self.assertEqual(events[0]["variant_id"], "scalp_funding_basis_btc_v1")
            self.assertEqual(events[0]["perp_side"], "short")


if __name__ == "__main__":
    unittest.main()

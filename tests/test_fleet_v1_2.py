import json
import os
import tempfile
import unittest

from langlang_trader.config import (
    ExecutionConfig,
    MarketDataConfig,
    PaperConfig,
    RiskConfig,
    SymbolSelectionConfig,
    UniverseConfig,
)
from langlang_trader.fleet import BotConfig, FleetConfig, FleetRunner
from langlang_trader.ledger import Ledger
from langlang_trader.market_data import StaticMarketData
from langlang_trader.models import Candle
from langlang_trader.strategy import LangLangV1_1Variant
from langlang_trader.universe import StaticUniverseProvider, UniverseSnapshot, UniverseSymbol


DAY_MS = 86_400_000


def daily(symbol, step, count=90):
    rows = []
    price = 100.0
    for idx in range(count):
        price *= 1 + step
        rows.append(
            Candle(
                symbol=symbol,
                bar="1D",
                ts=1_700_000_000_000 + idx * DAY_MS,
                open=price * 0.99,
                high=price * 1.02,
                low=price * 0.98,
                close=price,
                volume=1000.0 + idx * 20,
            )
        )
    return rows


def intraday(symbol, bar, step, count, interval):
    rows = []
    price = 100.0
    for idx in range(count):
        price *= 1 + step
        rows.append(
            Candle(
                symbol=symbol,
                bar=bar,
                ts=1_707_000_000_000 + idx * interval,
                open=price * 0.99,
                high=price * 1.02,
                low=price * 0.98,
                close=price,
                volume=1000.0 + idx,
            )
        )
    return rows


def all_bars(symbol, daily_step, intraday_step):
    return (
        daily(symbol, daily_step)
        + intraday(symbol, "1H", intraday_step, 80, 3_600_000)
        + intraday(symbol, "15m", intraday_step, 80, 900_000)
        + intraday(symbol, "5m", intraday_step, 80, 300_000)
    )


class FleetV12Test(unittest.TestCase):
    def test_all_market_dual_board_keeps_long_and_short_bots_on_their_own_boards(self):
        with tempfile.TemporaryDirectory() as tmp:
            long_symbol = "TURBO-USDT-SWAP"
            short_symbol = "WATER-USDT-SWAP"
            config = FleetConfig(
                run_id="fleet-v1-2",
                execution=ExecutionConfig(mode="paper", exchange="okx", executor="paper_okx"),
                paper=PaperConfig(initial_equity_usdt=50_000, fee_bps=5, slippage_bps=10),
                risk=RiskConfig(max_position_usdt=1_000, max_daily_loss_usdt=500, default_leverage=5),
                market_data=MarketDataConfig(symbols=["IGNORED-USDT-SWAP"], bars=["1D", "1H", "15m", "5m"]),
                selection=SymbolSelectionConfig(enabled=True, style="dual_board", long_top_n=1, short_top_n=1),
                universe=UniverseConfig(
                    mode="okx_all_usdt_swap",
                    snapshot_path=os.path.join(tmp, "universe_snapshot.json"),
                ),
                ledger_path=os.path.join(tmp, "fleet.sqlite3"),
                strategy_version="rules_langlang_v1_2",
                bots=[
                    BotConfig(
                        bot_id="long-bot",
                        variant=LangLangV1_1Variant(
                            variant_id="llv1_1_long_unit",
                            allowed_side="long",
                            exploratory=True,
                            ret_20d_min=0.10,
                            ret_60d_min=0.20,
                        ),
                    ),
                    BotConfig(
                        bot_id="short-bot",
                        variant=LangLangV1_1Variant(
                            variant_id="llv1_1_short_unit",
                            allowed_side="short",
                            exploratory=True,
                            short_ret_20d_max=-0.08,
                            short_ret_60d_max=-0.12,
                        ),
                    ),
                ],
            )
            market_data = StaticMarketData(
                {
                    "BTC-USDT-SWAP": all_bars("BTC-USDT-SWAP", 0.003, 0.0005),
                    "ETH-USDT-SWAP": all_bars("ETH-USDT-SWAP", 0.002, 0.0004),
                    long_symbol: all_bars(long_symbol, 0.018, 0.0020),
                    short_symbol: all_bars(short_symbol, -0.012, -0.0020),
                }
            )
            ledger = Ledger(config.ledger_path)
            runner = FleetRunner(
                config=config,
                market_data=market_data,
                ledger=ledger,
                universe_provider=StaticUniverseProvider(
                    symbols=[long_symbol, short_symbol],
                    reference_symbols=["BTC-USDT-SWAP", "ETH-USDT-SWAP"],
                ),
            )

            cycle = runner.run_once()

            self.assertEqual(cycle["symbols"], 4)
            self.assertEqual(cycle["selected_symbols"], 2)
            self.assertEqual(cycle["orders"], 2)
            self.assertTrue(os.path.exists(config.universe.snapshot_path))
            long_orders = ledger.list_rows("orders", run_id="fleet-v1-2", bot_id="long-bot")
            short_orders = ledger.list_rows("orders", run_id="fleet-v1-2", bot_id="short-bot")
            self.assertEqual({row["symbol"] for row in long_orders}, {long_symbol})
            self.assertEqual({row["symbol"] for row in short_orders}, {short_symbol})

    def test_combined_observed_universe_can_trade_binance_only_symbols_with_binance_paper_executor(self):
        class CombinedProvider:
            def list_symbols(self):
                return UniverseSnapshot(
                    mode="okx_binance_usdt_swap_observe",
                    generated_at="2024-03-12T00:00:00+00:00",
                    symbols=["SLOW-USDT-SWAP"],
                    reference_symbols=["BTC-USDT-SWAP", "ETH-USDT-SWAP"],
                    observed_symbols=[
                        "BTC-USDT-SWAP",
                        "ETH-USDT-SWAP",
                        "SLOW-USDT-SWAP",
                        "NEWCOIN-USDT-SWAP",
                    ],
                    rows=[
                        UniverseSymbol(
                            symbol="SLOW-USDT-SWAP",
                            base_ccy="SLOW",
                            quote_ccy="USDT",
                            inst_type="SWAP",
                            state="live",
                            is_reference=False,
                            tradable=True,
                            filter_reason="",
                            raw_payload={},
                            source_exchange="okx",
                            exchange_symbol="SLOW-USDT-SWAP",
                            execution_symbol="SLOW-USDT-SWAP",
                        ),
                        UniverseSymbol(
                            symbol="NEWCOIN-USDT-SWAP",
                            base_ccy="NEWCOIN",
                            quote_ccy="USDT",
                            inst_type="PERPETUAL",
                            state="TRADING",
                            is_reference=False,
                            tradable=False,
                            filter_reason="binance_observed_only_not_okx_executable",
                            raw_payload={},
                            source_exchange="binance",
                            exchange_symbol="NEWCOINUSDT",
                            observed_only=True,
                        ),
                    ],
                    raw_payload={"summary": {"binance_only_observed_count": 1}},
                )

        with tempfile.TemporaryDirectory() as tmp:
            config = FleetConfig(
                run_id="fleet-v1-2-binance-observe",
                execution=ExecutionConfig(mode="paper", exchange="multi", executor="paper_multi"),
                market_data=MarketDataConfig(symbols=[], bars=["1D", "1H", "15m", "5m"]),
                selection=SymbolSelectionConfig(enabled=True, style="dual_board", long_top_n=1, short_top_n=0),
                universe=UniverseConfig(
                    mode="okx_binance_usdt_swap_observe",
                    provider="okx_binance",
                    snapshot_path=os.path.join(tmp, "universe_snapshot.json"),
                ),
                ledger_path=os.path.join(tmp, "fleet.sqlite3"),
                strategy_version="rules_langlang_v1_2",
                bots=[
                    BotConfig(
                        bot_id="long-bot",
                        variant=LangLangV1_1Variant(
                            variant_id="llv1_1_long_unit",
                            allowed_side="long",
                            exploratory=True,
                            ret_20d_min=0.10,
                            ret_60d_min=0.20,
                        ),
                    )
                ],
            )
            market_data = StaticMarketData(
                {
                    "BTC-USDT-SWAP": all_bars("BTC-USDT-SWAP", 0.003, 0.0005),
                    "ETH-USDT-SWAP": all_bars("ETH-USDT-SWAP", 0.002, 0.0004),
                    "SLOW-USDT-SWAP": all_bars("SLOW-USDT-SWAP", 0.004, 0.0005),
                    "NEWCOIN-USDT-SWAP": all_bars("NEWCOIN-USDT-SWAP", 0.020, 0.0025),
                }
            )
            ledger = Ledger(config.ledger_path)
            runner = FleetRunner(
                config=config,
                market_data=market_data,
                ledger=ledger,
                universe_provider=CombinedProvider(),
            )

            cycle = runner.run_once()

            self.assertEqual(cycle["symbols"], 4)
            self.assertEqual(cycle["selected_symbols"], 1)
            self.assertEqual(cycle["orders"], 1)
            self.assertEqual(cycle["fills"], 1)
            selection_events = ledger.list_rows("risk_events")
            selection_event = next(row for row in selection_events if row["reason"] == "symbol_selection")
            self.assertIn("NEWCOIN-USDT-SWAP", selection_event["payload_json"])
            selection_payload = json.loads(selection_event["payload_json"])
            self.assertIn("NEWCOIN-USDT-SWAP", selection_payload["executable_selected_symbols"])
            self.assertIn("NEWCOIN-USDT-SWAP", selection_payload["routable_selected_symbols"])
            self.assertNotIn("NEWCOIN-USDT-SWAP", selection_payload["okx_executable_selected_symbols"])
            orders = ledger.list_rows("orders", run_id="fleet-v1-2-binance-observe", bot_id="long-bot")
            fills = ledger.list_rows("fills", run_id="fleet-v1-2-binance-observe", bot_id="long-bot")
            positions = ledger.list_rows("positions", run_id="fleet-v1-2-binance-observe", bot_id="long-bot")
            self.assertEqual([row["symbol"] for row in orders], ["NEWCOIN-USDT-SWAP"])
            self.assertEqual([row["exchange"] for row in orders], ["binance"])
            self.assertEqual([row["exchange"] for row in fills], ["binance"])
            self.assertEqual([row["exchange"] for row in positions], ["binance"])

    def test_paper_multi_requires_exchange_aware_universe_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = FleetConfig(
                run_id="fleet-v1-2-misconfigured",
                execution=ExecutionConfig(mode="paper", exchange="multi", executor="paper_multi"),
                market_data=MarketDataConfig(symbols=["BTC-USDT-SWAP"], bars=["1D", "1H", "15m", "5m"]),
                universe=UniverseConfig(mode="static", snapshot_path=os.path.join(tmp, "universe_snapshot.json")),
                ledger_path=os.path.join(tmp, "fleet.sqlite3"),
                strategy_version="rules_langlang_v1_2",
                bots=[
                    BotConfig(
                        bot_id="long-bot",
                        variant=LangLangV1_1Variant(variant_id="llv1_1_long_unit", allowed_side="long", exploratory=True),
                    )
                ],
            )

            with self.assertRaisesRegex(PermissionError, "paper_multi requires an exchange-aware universe"):
                FleetRunner(
                    config=config,
                    market_data=StaticMarketData({"BTC-USDT-SWAP": all_bars("BTC-USDT-SWAP", 0.003, 0.0005)}),
                    ledger=Ledger(config.ledger_path),
                )


if __name__ == "__main__":
    unittest.main()

import csv
import os
import tempfile
import unittest

from langlang_trader.config import (
    ExecutionConfig,
    MarketDataConfig,
    PaperConfig,
    RiskConfig,
    SymbolSelectionConfig,
)
from langlang_trader.features import DailyFeatureBuilder
from langlang_trader.fleet import BotConfig, FleetConfig, FleetRunner
from langlang_trader.ledger import Ledger
from langlang_trader.market_data import StaticMarketData
from langlang_trader.models import Candle
from langlang_trader.strategy import StrategyVariant
from langlang_trader.symbol_selection import HistoricalSymbolSelectionAnalyzer, SymbolSelector


START_TS = 1_650_672_000_000
DAY_MS = 86_400_000


def trend_candles(symbol: str, step: float, *, volume_step: float = 0.0, count: int = 80) -> list[Candle]:
    rows: list[Candle] = []
    price = 100.0
    for idx in range(count):
        price *= 1 + step
        volume = 1000.0 * (1 + volume_step * idx)
        rows.append(
            Candle(
                symbol=symbol,
                bar="1D",
                ts=START_TS + idx * DAY_MS,
                open=price * 0.99,
                high=price * 1.02,
                low=price * 0.98,
                close=price,
                volume=volume,
            )
        )
    return rows


def write_cache_csv(path: str, rows: list[Candle]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ts", "open", "high", "low", "close", "volume"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "ts": row.ts,
                    "open": row.open,
                    "high": row.high,
                    "low": row.low,
                    "close": row.close,
                    "volume": row.volume,
                }
            )


class SymbolSelectionTest(unittest.TestCase):
    def test_selector_identifies_long_strength_and_short_weakness(self):
        builder = DailyFeatureBuilder()
        snapshots = {
            "STRONG-USDT-SWAP": builder.build("STRONG-USDT-SWAP", trend_candles("STRONG-USDT-SWAP", 0.018, volume_step=0.01)),
            "WEAK-USDT-SWAP": builder.build("WEAK-USDT-SWAP", trend_candles("WEAK-USDT-SWAP", -0.012, volume_step=0.01)),
            "FLAT-USDT-SWAP": builder.build("FLAT-USDT-SWAP", trend_candles("FLAT-USDT-SWAP", 0.001)),
        }

        ranked = SymbolSelector().rank({key: value for key, value in snapshots.items() if value}, top_n=2)
        by_symbol = {row.symbol: row for row in ranked}

        self.assertTrue(by_symbol["STRONG-USDT-SWAP"].selected)
        self.assertTrue(by_symbol["WEAK-USDT-SWAP"].selected)
        self.assertEqual(by_symbol["STRONG-USDT-SWAP"].selection_bias, "long")
        self.assertEqual(by_symbol["WEAK-USDT-SWAP"].selection_bias, "short")
        self.assertIn("relative_strength_top_quartile", by_symbol["STRONG-USDT-SWAP"].reason_codes)
        self.assertIn("relative_weakness_bottom_quartile", by_symbol["WEAK-USDT-SWAP"].reason_codes)

    def test_historical_analyzer_writes_trade_selection_reasons_from_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = os.path.join(tmp, "kline_cache")
            daily_dir = os.path.join(cache_dir, "1D")
            out_dir = os.path.join(tmp, "selection")
            os.makedirs(daily_dir)
            strong = trend_candles("STRONG-USDT-SWAP", 0.018, volume_step=0.01)
            weak = trend_candles("WEAK-USDT-SWAP", -0.012, volume_step=0.01)
            flat = trend_candles("FLAT-USDT-SWAP", 0.001)
            write_cache_csv(os.path.join(daily_dir, "STRONG-USDT-SWAP.csv"), strong)
            write_cache_csv(os.path.join(daily_dir, "WEAK-USDT-SWAP.csv"), weak)
            write_cache_csv(os.path.join(daily_dir, "FLAT-USDT-SWAP.csv"), flat)

            trades_path = os.path.join(tmp, "standard_trades.csv")
            with open(trades_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["trade_id", "symbol", "side", "entry_time", "pnl_usdt", "return_rate"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "trade_id": "t-strong",
                        "symbol": "STRONG-USDT-SWAP",
                        "side": "long",
                        "entry_time": "2022-07-25T00:00:00+00:00",
                        "pnl_usdt": "120",
                        "return_rate": "0.12",
                    }
                )
                writer.writerow(
                    {
                        "trade_id": "t-weak",
                        "symbol": "WEAK-USDT-SWAP",
                        "side": "short",
                        "entry_time": "2022-07-25T00:00:00+00:00",
                        "pnl_usdt": "80",
                        "return_rate": "0.08",
                    }
                )

            result = HistoricalSymbolSelectionAnalyzer(cache_dir).run(trades_path, out_dir)

            self.assertEqual(result["trades"], 2)
            self.assertTrue(os.path.exists(os.path.join(out_dir, "symbol_selection_features.csv")))
            self.assertTrue(os.path.exists(os.path.join(out_dir, "symbol_selection_report.md")))
            with open(os.path.join(out_dir, "symbol_selection_features.csv"), encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            reasons_by_trade = {row["trade_id"]: row["reason_codes"] for row in rows}
            self.assertIn("relative_strength_top_quartile", reasons_by_trade["t-strong"])
            self.assertIn("relative_weakness_bottom_quartile", reasons_by_trade["t-weak"])

    def test_fleet_selection_gate_trades_only_selected_symbols(self):
        with tempfile.TemporaryDirectory() as tmp:
            strong = "STRONG-USDT-SWAP"
            medium = "MEDIUM-USDT-SWAP"
            variant = StrategyVariant("loose", 0.05, 0.10, 0.40, 0.25, 0.005)
            config = FleetConfig(
                run_id="selection-run",
                execution=ExecutionConfig(mode="paper", exchange="okx", executor="paper_okx"),
                paper=PaperConfig(initial_equity_usdt=50_000, fee_bps=5, slippage_bps=10),
                risk=RiskConfig(max_position_usdt=1_000, max_daily_loss_usdt=500, default_leverage=5),
                market_data=MarketDataConfig(symbols=[strong, medium]),
                selection=SymbolSelectionConfig(enabled=True, top_n=1),
                ledger_path=os.path.join(tmp, "fleet.sqlite3"),
                bots=[BotConfig(bot_id="bot-1", variant=variant)],
            )
            ledger = Ledger(config.ledger_path)
            runner = FleetRunner(
                config=config,
                market_data=StaticMarketData(
                    {
                        strong: trend_candles(strong, 0.018, volume_step=0.01),
                        medium: trend_candles(medium, 0.008),
                    }
                ),
                ledger=ledger,
            )

            cycle = runner.run_once()

            self.assertEqual(cycle["selected_symbols"], 1)
            self.assertEqual(cycle["selection_skips"], 1)
            self.assertEqual(cycle["orders"], 1)
            orders = ledger.list_rows("orders", run_id="selection-run", bot_id="bot-1")
            self.assertEqual({row["symbol"] for row in orders}, {strong})


if __name__ == "__main__":
    unittest.main()

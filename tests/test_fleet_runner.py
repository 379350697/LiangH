import os
import tempfile
import threading
import time
import unittest

from langlang_trader.config import ExecutionConfig, MarketDataConfig, PaperConfig, RiskConfig, SymbolSelectionConfig, UniverseConfig
from langlang_trader.features import FeatureSnapshot
from langlang_trader.fleet import BotConfig, FleetConfig, FleetRunner, _selection_states_for_profiles, load_fleet_config
from langlang_trader.ledger import Ledger
from langlang_trader.market_data import StaticMarketData
from langlang_trader.models import Candle
from langlang_trader.strategy import LangLangEnhancedVariant, LangLangNativeVariant, StrategyVariant


def candles(symbol="BTC-USDT-SWAP", bar="1D", count=70, step_ms=86_400_000, slope=0.012):
    rows = []
    price = 100.0
    for idx in range(count):
        close = price * (1 + idx * slope)
        rows.append(
            Candle(
                symbol=symbol,
                bar=bar,
                ts=1_700_000_000_000 + idx * step_ms,
                open=close * 0.99,
                high=close * 1.02,
                low=close * 0.98,
                close=close,
                volume=1000 + idx,
            )
        )
    return rows


def multi_timeframe_candles(symbol="BTC-USDT-SWAP"):
    rows = []
    rows.extend(candles(symbol=symbol, bar="1D", count=70, step_ms=86_400_000, slope=0.02))
    rows.extend(candles(symbol=symbol, bar="1H", count=80, step_ms=3_600_000, slope=0.003))
    rows.extend(candles(symbol=symbol, bar="15m", count=90, step_ms=900_000, slope=0.0015))
    rows.extend(candles(symbol=symbol, bar="5m", count=70, step_ms=300_000, slope=0.001))
    rows.extend(candles(symbol=symbol, bar="1m", count=130, step_ms=60_000, slope=0.0005))
    return rows


def feature_snapshot(symbol: str, **features):
    base = {
        "ret_3d": 0.0,
        "ret_7d": 0.0,
        "ret_20d": 0.0,
        "ret_60d": 0.0,
        "pos_20d": 0.85,
        "pullback_from_20d_high": -0.05,
        "vol_ratio_20d": 1.3,
        "upside_space_pct": 0.30,
        "ma_5": 1.0,
        "ma_20": 1.0,
        "latest_close": 2.0,
    }
    base.update(features)
    return FeatureSnapshot(symbol=symbol, bar="1D", last_ts=1_700_000_000_000, features=base, created_at="2024-01-01T00:00:00Z")


class FleetRunnerTest(unittest.TestCase):
    def test_market_data_fetch_can_run_with_configured_workers(self):
        class SlowMarketData(StaticMarketData):
            def __init__(self, candles_by_symbol):
                super().__init__(candles_by_symbol)
                self.active = 0
                self.max_active = 0
                self.lock = threading.Lock()

            def get_candles(self, symbol: str, bar: str = "1D", limit: int = 120):
                with self.lock:
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                try:
                    time.sleep(0.02)
                    return super().get_candles(symbol, bar=bar, limit=limit)
                finally:
                    with self.lock:
                        self.active -= 1

        with tempfile.TemporaryDirectory() as tmp:
            symbols = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP", "DOGE-USDT-SWAP"]
            config = FleetConfig(
                run_id="unit-run",
                execution=ExecutionConfig(mode="paper", exchange="okx", executor="paper_okx"),
                paper=PaperConfig(initial_equity_usdt=50_000, fee_bps=5, slippage_bps=10),
                risk=RiskConfig(max_position_usdt=1_000, max_daily_loss_usdt=500, default_leverage=5),
                market_data=MarketDataConfig(symbols=symbols, max_fetch_workers=4),
                ledger_path=os.path.join(tmp, "fleet.sqlite3"),
                bots=[BotConfig(bot_id="bot-1", variant=StrategyVariant("fleet-loose", 0.12, 0.32, 0.45, 0.18, 0.005))],
            )
            market_data = SlowMarketData({symbol: candles(symbol) for symbol in symbols})
            runner = FleetRunner(config=config, market_data=market_data, ledger=Ledger(config.ledger_path))

            cycle = runner.run_once()

            self.assertEqual(cycle["market_data_errors"], 0)
            self.assertGreaterEqual(market_data.max_active, 2)

    def test_latest_price_failure_falls_back_to_latest_candle_close(self):
        class CandleOnlyMarketData(StaticMarketData):
            def latest_price(self, symbol: str) -> float:
                raise RuntimeError("ticker unavailable")

        with tempfile.TemporaryDirectory() as tmp:
            variant = StrategyVariant("fleet-loose", 0.12, 0.32, 0.45, 0.18, 0.005)
            config = FleetConfig(
                run_id="unit-latest-fallback-run",
                execution=ExecutionConfig(mode="paper", exchange="okx", executor="paper_okx"),
                paper=PaperConfig(initial_equity_usdt=50_000, fee_bps=5, slippage_bps=10),
                risk=RiskConfig(max_position_usdt=1_000, max_daily_loss_usdt=500, default_leverage=5),
                market_data=MarketDataConfig(symbols=["BTC-USDT-SWAP"]),
                ledger_path=os.path.join(tmp, "fleet.sqlite3"),
                bots=[BotConfig(bot_id="bot-1", variant=variant)],
            )
            ledger = Ledger(config.ledger_path)
            runner = FleetRunner(
                config=config,
                market_data=CandleOnlyMarketData({"BTC-USDT-SWAP": candles()}),
                ledger=ledger,
            )

            cycle = runner.run_once()

            self.assertEqual(cycle["market_data_errors"], 0)
            self.assertEqual(cycle["fills"], 1)
            events = [
                row
                for row in ledger.list_rows("risk_events", run_id="unit-latest-fallback-run")
                if row["reason"] == "latest_price_fallback_to_candle_close"
            ]
            self.assertEqual(len(events), 1)

    def test_three_bots_share_market_data_but_record_independent_trades(self):
        with tempfile.TemporaryDirectory() as tmp:
            variant = StrategyVariant("fleet-loose", 0.12, 0.32, 0.45, 0.18, 0.005)
            config = FleetConfig(
                run_id="unit-run",
                execution=ExecutionConfig(mode="paper", exchange="okx", executor="paper_okx"),
                paper=PaperConfig(initial_equity_usdt=50_000, fee_bps=5, slippage_bps=10),
                risk=RiskConfig(max_position_usdt=1_000, max_daily_loss_usdt=500, default_leverage=5),
                market_data=MarketDataConfig(symbols=["BTC-USDT-SWAP"]),
                ledger_path=os.path.join(tmp, "fleet.sqlite3"),
                bots=[
                    BotConfig(bot_id="bot-1", variant=variant),
                    BotConfig(bot_id="bot-2", variant=variant),
                    BotConfig(bot_id="bot-3", variant=variant),
                ],
            )
            ledger = Ledger(config.ledger_path)
            runner = FleetRunner(
                config=config,
                market_data=StaticMarketData({"BTC-USDT-SWAP": candles()}),
                ledger=ledger,
            )

            cycle = runner.run_once()

            self.assertEqual(cycle["bots"], 3)
            self.assertEqual(cycle["signals"], 3)
            self.assertEqual(cycle["orders"], 3)
            self.assertEqual(cycle["fills"], 3)
            self.assertEqual({row["bot_id"] for row in ledger.list_rows("positions")}, {"bot-1", "bot-2", "bot-3"})

    def test_second_tick_does_not_open_duplicate_position_for_same_bot_symbol(self):
        with tempfile.TemporaryDirectory() as tmp:
            variant = StrategyVariant("fleet-loose", 0.12, 0.32, 0.45, 0.18, 0.005)
            config = FleetConfig(
                run_id="unit-run",
                execution=ExecutionConfig(mode="paper", exchange="okx", executor="paper_okx"),
                paper=PaperConfig(initial_equity_usdt=50_000, fee_bps=5, slippage_bps=10),
                risk=RiskConfig(max_position_usdt=1_000, max_daily_loss_usdt=500, default_leverage=5),
                market_data=MarketDataConfig(symbols=["BTC-USDT-SWAP"]),
                ledger_path=os.path.join(tmp, "fleet.sqlite3"),
                bots=[BotConfig(bot_id="bot-1", variant=variant)],
            )
            ledger = Ledger(config.ledger_path)
            runner = FleetRunner(
                config=config,
                market_data=StaticMarketData({"BTC-USDT-SWAP": candles()}),
                ledger=ledger,
            )

            first = runner.run_once()
            second = runner.run_once()

            self.assertEqual(first["orders"], 1)
            self.assertEqual(second["orders"], 0)
            self.assertEqual(second["risk_rejections"], 1)
            bot_ledger = ledger.scoped(run_id="unit-run", bot_id="bot-1", variant_id=variant.variant_id)
            self.assertEqual(len(bot_ledger.list_rows("orders", run_id="unit-run", bot_id="bot-1")), 1)
            self.assertEqual(len(bot_ledger.list_rows("fills", run_id="unit-run", bot_id="bot-1")), 1)

    def test_bot_level_strategy_versions_run_together_and_record_separate_versions(self):
        with tempfile.TemporaryDirectory() as tmp:
            native = LangLangNativeVariant(
                variant_id="langlang_01",
                allowed_side="long",
                exploratory=True,
                ret_20d_min=0.10,
                ret_60d_min=0.30,
                min_upside_space_pct=0.0,
                min_vol_ratio_20d=0.0,
            )
            enhanced = LangLangEnhancedVariant(
                variant_id="langlang_plus_01",
                allowed_side="long",
                exploratory=True,
                ret_20d_min=0.10,
                ret_60d_min=0.30,
                min_upside_space_pct=0.0,
                min_vol_ratio_20d=0.0,
                max_turnover_rank_24h=999,
            )
            config = FleetConfig(
                run_id="unit-mixed-run",
                strategy_version="rules_langlang_native_final",
                execution=ExecutionConfig(mode="paper", exchange="okx", executor="paper_okx"),
                paper=PaperConfig(initial_equity_usdt=50_000, fee_bps=5, slippage_bps=10),
                risk=RiskConfig(max_position_usdt=1_000, max_daily_loss_usdt=500, default_leverage=5),
                market_data=MarketDataConfig(symbols=["BTC-USDT-SWAP"]),
                ledger_path=os.path.join(tmp, "fleet.sqlite3"),
                bots=[
                    BotConfig(bot_id="bot-native", variant=native, strategy_version="rules_langlang_native_final"),
                    BotConfig(bot_id="bot-plus", variant=enhanced, strategy_version="rules_langlang_enhanced_final"),
                ],
            )
            ledger = Ledger(config.ledger_path)
            runner = FleetRunner(
                config=config,
                market_data=StaticMarketData({"BTC-USDT-SWAP": multi_timeframe_candles()}),
                ledger=ledger,
            )

            cycle = runner.run_once()

            self.assertEqual(cycle["bots"], 2)
            self.assertEqual(cycle["signals"], 2)
            self.assertEqual(cycle["fills"], 2)
            rows = ledger.list_rows("signals", run_id="unit-mixed-run")
            versions_by_bot = {row["bot_id"]: row["strategy_version"] for row in rows}
            self.assertEqual(versions_by_bot["bot-native"], "rules_langlang_native_final")
            self.assertEqual(versions_by_bot["bot-plus"], "rules_langlang_enhanced_final")
            for table in ("order_intents", "orders", "fills", "positions"):
                with self.subTest(table=table):
                    table_rows = ledger.list_rows(table, run_id="unit-mixed-run")
                    self.assertEqual(
                        {row["bot_id"]: row["strategy_version"] for row in table_rows},
                        versions_by_bot,
                    )
            trade_snapshots = [
                row
                for row in ledger.list_rows("equity_snapshots", run_id="unit-mixed-run")
                if row["strategy_version"] in versions_by_bot.values()
            ]
            self.assertEqual({row["bot_id"] for row in trade_snapshots}, {"bot-native", "bot-plus"})

    def test_fleet_config_loads_langlang_10bot_persistent_strategy_forest(self):
        config = load_fleet_config(
            "output/fleet/langlang_strategy_forest/selected_fleet_config_langlang_10bot.json"
        )

        self.assertEqual(config.run_id, "langlang-paper-main-v1")
        self.assertEqual(config.ledger_path, "output/fleet/langlang_strategy_forest/fleet.sqlite3")
        self.assertEqual(config.execution.executor, "paper_multi")
        self.assertFalse(config.execution.allow_live_orders)
        self.assertEqual(len(config.bots), 10)
        self.assertEqual(
            {bot.bot_id for bot in config.bots},
            {
                "bot_langlang_01",
                "bot_langlang_plus_01",
                "bot_langlang_01_select",
                "bot_langlang_01_entry",
                "bot_langlang_01_exit",
                "bot_langlang_01_risk",
                "bot_langlang_plus_01_select",
                "bot_langlang_plus_01_entry",
                "bot_langlang_plus_01_exit",
                "bot_langlang_plus_01_loss",
            },
        )
        versions = {bot.strategy_version for bot in config.bots}
        self.assertEqual(versions, {"rules_langlang_native_final", "rules_langlang_enhanced_final"})
        profiles_by_bot = {bot.bot_id: bot.selection_profile for bot in config.bots}
        self.assertEqual(
            profiles_by_bot,
            {
                "bot_langlang_01": "langlang_01",
                "bot_langlang_plus_01": "langlang_plus_01",
                "bot_langlang_01_select": "langlang_01_select",
                "bot_langlang_01_entry": "langlang_01_entry",
                "bot_langlang_01_exit": "langlang_01_exit",
                "bot_langlang_01_risk": "langlang_01_risk",
                "bot_langlang_plus_01_select": "langlang_plus_01_select",
                "bot_langlang_plus_01_entry": "langlang_plus_01_entry",
                "bot_langlang_plus_01_exit": "langlang_plus_01_exit",
                "bot_langlang_plus_01_loss": "langlang_plus_01_loss",
            },
        )
        self.assertEqual(len(set(profiles_by_bot.values())), 10)

    def test_dual_board_selection_is_computed_per_bot_profile(self):
        snapshots = {
            "BTC-USDT-SWAP": feature_snapshot("BTC-USDT-SWAP"),
            "ETH-USDT-SWAP": feature_snapshot("ETH-USDT-SWAP"),
            "ALPHA-USDT-SWAP": feature_snapshot(
                "ALPHA-USDT-SWAP",
                ret_3d=0.12,
                ret_7d=0.25,
                ret_20d=0.60,
                ret_60d=1.20,
                vol_ratio_20d=1.25,
                turnover_rank=199,
                turnover_rank_top_n=200,
                oi_change_3d=-0.50,
                funding_rate_last=0.02,
            ),
            "BETA-USDT-SWAP": feature_snapshot(
                "BETA-USDT-SWAP",
                ret_3d=0.11,
                ret_7d=0.24,
                ret_20d=0.58,
                ret_60d=1.10,
                vol_ratio_20d=4.00,
                turnover_rank=1,
                turnover_rank_top_n=200,
                oi_change_3d=0.50,
                funding_rate_last=0.001,
            ),
            "GAMMA-USDT-SWAP": feature_snapshot(
                "GAMMA-USDT-SWAP",
                ret_3d=0.05,
                ret_7d=0.10,
                ret_20d=0.30,
                ret_60d=0.60,
                vol_ratio_20d=1.20,
                turnover_rank=100,
                turnover_rank_top_n=200,
                oi_change_3d=0.0,
                funding_rate_last=0.0,
            ),
        }
        states = _selection_states_for_profiles(
            SymbolSelectionConfig(
                enabled=True,
                style="dual_board",
                scoring_profile="enhanced",
                long_top_n=1,
                short_top_n=0,
            ),
            snapshots,
            reference_symbols={"BTC-USDT-SWAP", "ETH-USDT-SWAP"},
            profiles={"native", "enhanced"},
        )

        self.assertEqual(states["native"]["long_selected_symbols"], {"ALPHA-USDT-SWAP"})
        self.assertEqual(states["enhanced"]["long_selected_symbols"], {"BETA-USDT-SWAP"})
        self.assertIn("funding_overheated", states["enhanced"]["selection_results_by_side"]["long"]["ALPHA-USDT-SWAP"].filter_codes)

    def test_fresh_runner_restores_existing_paper_position_with_same_run_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            variant = StrategyVariant("fleet-loose", 0.12, 0.32, 0.45, 0.18, 0.005)
            config = FleetConfig(
                run_id="unit-persistent-run",
                execution=ExecutionConfig(mode="paper", exchange="okx", executor="paper_okx"),
                paper=PaperConfig(initial_equity_usdt=50_000, fee_bps=5, slippage_bps=10),
                risk=RiskConfig(max_position_usdt=1_000, max_daily_loss_usdt=500, default_leverage=5),
                market_data=MarketDataConfig(symbols=["BTC-USDT-SWAP"]),
                ledger_path=os.path.join(tmp, "fleet.sqlite3"),
                bots=[BotConfig(bot_id="bot-1", variant=variant)],
            )
            market_data = StaticMarketData({"BTC-USDT-SWAP": candles()})
            first_runner = FleetRunner(config=config, market_data=market_data, ledger=Ledger(config.ledger_path))

            first = first_runner.run_once()
            second_runner = FleetRunner(config=config, market_data=market_data, ledger=Ledger(config.ledger_path))
            second = second_runner.run_once()

            self.assertEqual(first["orders"], 1)
            self.assertEqual(second["orders"], 0)
            self.assertEqual(second["risk_rejections"], 1)
            ledger = Ledger(config.ledger_path)
            self.assertEqual(len(ledger.list_rows("orders", run_id="unit-persistent-run", bot_id="bot-1")), 1)
            self.assertEqual(len(ledger.list_rows("positions", run_id="unit-persistent-run", bot_id="bot-1")), 1)

    def test_each_bot_records_persistent_account_snapshot_even_when_it_skips(self):
        with tempfile.TemporaryDirectory() as tmp:
            loose = LangLangNativeVariant(
                variant_id="langlang_01",
                allowed_side="long",
                exploratory=True,
                ret_20d_min=0.10,
                ret_60d_min=0.30,
                min_upside_space_pct=0.0,
                min_vol_ratio_20d=0.0,
            )
            strict = LangLangEnhancedVariant(
                variant_id="langlang_plus_01_entry",
                allowed_side="long",
                exploratory=False,
                min_historical_match_score=0.99,
                ret_20d_min=0.10,
                ret_60d_min=0.30,
                min_upside_space_pct=0.0,
                min_vol_ratio_20d=0.0,
                max_turnover_rank_24h=999,
            )
            config = FleetConfig(
                run_id="unit-heartbeat-run",
                strategy_version="rules_langlang_native_final",
                execution=ExecutionConfig(mode="paper", exchange="okx", executor="paper_okx"),
                paper=PaperConfig(initial_equity_usdt=50_000, fee_bps=5, slippage_bps=10),
                risk=RiskConfig(max_position_usdt=1_000, max_daily_loss_usdt=500, default_leverage=5),
                market_data=MarketDataConfig(symbols=["BTC-USDT-SWAP"]),
                ledger_path=os.path.join(tmp, "fleet.sqlite3"),
                bots=[
                    BotConfig(bot_id="bot-loose", variant=loose, strategy_version="rules_langlang_native_final"),
                    BotConfig(bot_id="bot-strict", variant=strict, strategy_version="rules_langlang_enhanced_final"),
                ],
            )
            ledger = Ledger(config.ledger_path)
            runner = FleetRunner(
                config=config,
                market_data=StaticMarketData({"BTC-USDT-SWAP": multi_timeframe_candles()}),
                ledger=ledger,
            )

            runner.run_once()

            equity_bots = {
                row["bot_id"]
                for row in ledger.list_rows("equity_snapshots", run_id="unit-heartbeat-run")
                if row["variant_id"] in {"langlang_01", "langlang_plus_01_entry"}
            }
            self.assertEqual(equity_bots, {"bot-loose", "bot-strict"})
            versions_by_bot = {
                row["bot_id"]: row["strategy_version"]
                for row in ledger.list_rows("equity_snapshots", run_id="unit-heartbeat-run")
                if row["variant_id"] in {"langlang_01", "langlang_plus_01_entry"}
            }
            self.assertEqual(
                versions_by_bot,
                {
                    "bot-loose": "rules_langlang_native_final",
                    "bot-strict": "rules_langlang_enhanced_final",
                },
            )

    def test_multi_exchange_heartbeat_uses_multi_exchange_scope(self):
        from langlang_trader.models import utc_now_iso
        from langlang_trader.universe import UniverseSnapshot, UniverseSymbol

        class StaticUniverseProvider:
            def list_symbols(self):
                return UniverseSnapshot(
                    mode="test",
                    generated_at=utc_now_iso(),
                    symbols=["BTC-USDT-SWAP"],
                    reference_symbols=[],
                    rows=[
                        UniverseSymbol(
                            symbol="BTC-USDT-SWAP",
                            base_ccy="BTC",
                            quote_ccy="USDT",
                            inst_type="SWAP",
                            state="live",
                            is_reference=False,
                            tradable=True,
                            filter_reason="",
                            raw_payload={},
                            source_exchange="okx",
                            exchange_symbol="BTC-USDT-SWAP",
                            execution_symbol="BTC-USDT-SWAP",
                        ),
                        UniverseSymbol(
                            symbol="BTC-USDT-SWAP",
                            base_ccy="BTC",
                            quote_ccy="USDT",
                            inst_type="SWAP",
                            state="TRADING",
                            is_reference=False,
                            tradable=True,
                            filter_reason="",
                            raw_payload={},
                            source_exchange="binance",
                            exchange_symbol="BTCUSDT",
                            execution_symbol="BTCUSDT",
                        ),
                    ],
                    raw_payload={"summary": {}},
                    observed_symbols=["BTC-USDT-SWAP"],
                )

        with tempfile.TemporaryDirectory() as tmp:
            variant = LangLangNativeVariant(
                variant_id="langlang_01",
                allowed_side="long",
                exploratory=True,
                ret_20d_min=0.10,
                ret_60d_min=0.30,
                min_upside_space_pct=0.0,
                min_vol_ratio_20d=0.0,
            )
            config = FleetConfig(
                run_id="unit-multi-heartbeat-run",
                strategy_version="rules_langlang_native_final",
                execution=ExecutionConfig(mode="paper", exchange="multi", executor="paper_multi"),
                paper=PaperConfig(initial_equity_usdt=50_000, fee_bps=5, slippage_bps=10),
                risk=RiskConfig(max_position_usdt=1_000, max_daily_loss_usdt=500, default_leverage=5),
                market_data=MarketDataConfig(symbols=[]),
                universe=UniverseConfig(
                    mode="okx_binance_usdt_swap_observe",
                    provider="okx_binance",
                    reference_symbols=[],
                ),
                ledger_path=os.path.join(tmp, "fleet.sqlite3"),
                bots=[BotConfig(bot_id="bot-multi", variant=variant, strategy_version="rules_langlang_native_final")],
            )
            ledger = Ledger(config.ledger_path)
            runner = FleetRunner(
                config=config,
                market_data=StaticMarketData({"BTC-USDT-SWAP": multi_timeframe_candles()}),
                ledger=ledger,
                universe_provider=StaticUniverseProvider(),
            )

            runner.run_once()

            heartbeat_rows = [
                row for row in ledger.list_rows("equity_snapshots", run_id="unit-multi-heartbeat-run", bot_id="bot-multi")
                if '"source": "fleet_bot_tick"' in row["raw_json"]
            ]
            self.assertEqual({row["exchange"] for row in heartbeat_rows}, {"multi"})
            self.assertEqual({row["strategy_version"] for row in heartbeat_rows}, {"rules_langlang_native_final"})

    def test_fleet_level_events_use_config_run_and_multi_exchange_scope(self):
        from langlang_trader.models import utc_now_iso
        from langlang_trader.universe import UniverseSnapshot, UniverseSymbol

        class StaticUniverseProvider:
            def list_symbols(self):
                return UniverseSnapshot(
                    mode="test",
                    generated_at=utc_now_iso(),
                    symbols=["BTC-USDT-SWAP"],
                    reference_symbols=[],
                    rows=[
                        UniverseSymbol(
                            symbol="BTC-USDT-SWAP",
                            base_ccy="BTC",
                            quote_ccy="USDT",
                            inst_type="SWAP",
                            state="live",
                            is_reference=False,
                            tradable=True,
                            filter_reason="",
                            raw_payload={},
                            source_exchange="okx",
                            exchange_symbol="BTC-USDT-SWAP",
                            execution_symbol="BTC-USDT-SWAP",
                        )
                    ],
                    raw_payload={"summary": {}},
                    observed_symbols=["BTC-USDT-SWAP"],
                )

        with tempfile.TemporaryDirectory() as tmp:
            variant = LangLangNativeVariant(
                variant_id="langlang_01",
                allowed_side="long",
                exploratory=True,
                ret_20d_min=0.10,
                ret_60d_min=0.30,
                min_upside_space_pct=0.0,
                min_vol_ratio_20d=0.0,
            )
            config = FleetConfig(
                run_id="unit-fleet-event-run",
                strategy_version="rules_langlang_native_final",
                execution=ExecutionConfig(mode="paper", exchange="multi", executor="paper_multi"),
                paper=PaperConfig(initial_equity_usdt=50_000, fee_bps=5, slippage_bps=10),
                risk=RiskConfig(max_position_usdt=1_000, max_daily_loss_usdt=500, default_leverage=5),
                market_data=MarketDataConfig(symbols=[]),
                universe=UniverseConfig(
                    mode="okx_binance_usdt_swap_observe",
                    provider="okx_binance",
                    reference_symbols=[],
                ),
                selection=SymbolSelectionConfig(enabled=True, style="dual_board", scoring_profile="native", long_top_n=1),
                ledger_path=os.path.join(tmp, "fleet.sqlite3"),
                bots=[BotConfig(bot_id="bot-multi", variant=variant, strategy_version="rules_langlang_native_final")],
            )
            ledger = Ledger(config.ledger_path)
            runner = FleetRunner(
                config=config,
                market_data=StaticMarketData({"BTC-USDT-SWAP": multi_timeframe_candles()}),
                ledger=ledger,
                universe_provider=StaticUniverseProvider(),
            )

            runner.run_once()

            rows = [
                (row["run_id"], row["bot_id"], row["variant_id"], row["exchange"], row["reason"])
                for row in ledger.list_rows("risk_events")
                if row["reason"] in {"universe_snapshot", "symbol_selection"}
            ]
            self.assertEqual(
                set(rows),
                {
                    ("unit-fleet-event-run", "fleet", "fleet", "multi", "universe_snapshot"),
                    ("unit-fleet-event-run", "fleet", "fleet", "multi", "symbol_selection"),
                },
            )

    def test_runtime_symbols_falls_back_to_cached_snapshot_when_provider_fails(self):
        from langlang_trader.models import utc_now_iso
        from langlang_trader.universe import UniverseSnapshot, UniverseSymbol, write_universe_snapshot

        class FailingUniverseProvider:
            def list_symbols(self):
                raise RuntimeError("binance exchangeInfo unavailable")

        with tempfile.TemporaryDirectory() as tmp:
            snapshot_path = os.path.join(tmp, "universe_snapshot.json")
            snapshot = UniverseSnapshot(
                mode="okx_binance_usdt_swap_observe",
                generated_at=utc_now_iso(),
                symbols=["BTC-USDT-SWAP"],
                reference_symbols=[],
                rows=[
                    UniverseSymbol(
                        symbol="BTC-USDT-SWAP",
                        base_ccy="BTC",
                        quote_ccy="USDT",
                        inst_type="SWAP",
                        state="live",
                        is_reference=False,
                        tradable=True,
                        filter_reason="",
                        raw_payload={},
                        source_exchange="okx",
                        exchange_symbol="BTC-USDT-SWAP",
                        execution_symbol="BTC-USDT-SWAP",
                    )
                ],
                raw_payload={"summary": {"source": "cached"}},
                observed_symbols=["BTC-USDT-SWAP"],
            )
            write_universe_snapshot(snapshot_path, snapshot)
            variant = LangLangNativeVariant(
                variant_id="langlang_01",
                allowed_side="long",
                exploratory=True,
                ret_20d_min=0.10,
                ret_60d_min=0.30,
                min_upside_space_pct=0.0,
                min_vol_ratio_20d=0.0,
            )
            config = FleetConfig(
                run_id="unit-universe-fallback-run",
                strategy_version="rules_langlang_native_final",
                execution=ExecutionConfig(mode="paper", exchange="multi", executor="paper_multi"),
                paper=PaperConfig(initial_equity_usdt=50_000, fee_bps=5, slippage_bps=10),
                risk=RiskConfig(max_position_usdt=1_000, max_daily_loss_usdt=500, default_leverage=5),
                market_data=MarketDataConfig(symbols=[]),
                universe=UniverseConfig(
                    mode="okx_binance_usdt_swap_observe",
                    provider="okx_binance",
                    reference_symbols=[],
                    snapshot_path=snapshot_path,
                ),
                ledger_path=os.path.join(tmp, "fleet.sqlite3"),
                bots=[BotConfig(bot_id="bot-multi", variant=variant, strategy_version="rules_langlang_native_final")],
            )
            ledger = Ledger(config.ledger_path)
            runner = FleetRunner(
                config=config,
                market_data=StaticMarketData({"BTC-USDT-SWAP": multi_timeframe_candles()}),
                ledger=ledger,
                universe_provider=FailingUniverseProvider(),
            )

            cycle = runner.run_once()

            self.assertEqual(cycle["market_data_errors"], 0)
            fallback_events = [
                row
                for row in ledger.list_rows("risk_events", run_id="unit-universe-fallback-run", bot_id="fleet")
                if row["reason"] == "universe_snapshot_fallback"
            ]
            self.assertEqual(len(fallback_events), 1)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import time

from langlang_trader.config import UniverseConfig
from langlang_trader.fleet import FleetConfig, FleetRunner, load_fleet_config
from langlang_trader.ledger import Ledger
from langlang_trader.market_data import BinanceRestMarketData, FallbackMarketData, OkxRestMarketData


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a LangLang paper fleet")
    parser.add_argument("--config", required=True)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--interval-seconds", type=float, default=60.0)
    parser.add_argument("--symbols", help="Comma-separated startup universe override for paper smoke runs")
    args = parser.parse_args(argv)
    if not args.once and not args.loop:
        parser.error("choose --once or --loop")

    config = config_with_symbol_override(load_fleet_config(args.config), args.symbols)
    ledger = Ledger(config.ledger_path)
    runner = FleetRunner(config=config, market_data=market_data_for_config(config), ledger=ledger)
    if args.once:
        print(json.dumps(runner.run_once(), ensure_ascii=False, sort_keys=True))
        return 0
    while True:
        print(json.dumps(runner.run_once(), ensure_ascii=False, sort_keys=True))
        time.sleep(args.interval_seconds)


def config_with_symbol_override(config: FleetConfig, symbols: str | None) -> FleetConfig:
    if not symbols:
        return config
    parsed = [symbol.strip() for symbol in symbols.split(",") if symbol.strip()]
    if not parsed:
        return config
    return replace(
        config,
        market_data=replace(config.market_data, symbols=parsed),
        universe=UniverseConfig(mode="static", reference_symbols=config.universe.reference_symbols),
    )


def market_data_for_config(config: FleetConfig):
    okx = OkxRestMarketData()
    if config.universe.mode == "okx_binance_usdt_swap_observe" or config.universe.provider == "okx_binance":
        return FallbackMarketData(okx, BinanceRestMarketData())
    return okx


if __name__ == "__main__":
    raise SystemExit(main())

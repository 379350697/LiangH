from __future__ import annotations

import argparse
import json

from langlang_trader.config import load_config
from langlang_trader.ledger import Ledger
from langlang_trader.market_data import OkxRestMarketData
from langlang_trader.runner import TradingRunner


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LangLang lightweight paper/live trading runner v0.1")
    parser.add_argument("--config", required=True, help="Path to JSON config")
    parser.add_argument("--once", action="store_true", help="Run one polling cycle")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    ledger = Ledger(config.ledger_path)
    runner = TradingRunner(config=config, market_data=OkxRestMarketData(), ledger=ledger)
    result = runner.run_once()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

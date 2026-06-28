import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path

from langlang_trader.kline_backfill import BAR_MS, CSV_COLUMNS
from langlang_trader.market_cache import (
    MarketSnapshotCache,
    PublicMarketDataCooldownError,
    PublicMarketDataGuard,
    RollingKlineCacheMarketData,
)
from langlang_trader.models import Candle, Ticker


def candle(symbol="BTC-USDT-SWAP", bar="1H", ts=1_700_000_000_000, close=100.0, source="unit"):
    return Candle(
        symbol=symbol,
        bar=bar,
        ts=ts,
        open=close * 0.99,
        high=close * 1.01,
        low=close * 0.98,
        close=close,
        volume=1000.0,
        source=source,
    )


def write_cache_rows(cache_dir: Path, symbol: str, bar: str, rows: list[Candle]) -> None:
    path = cache_dir / bar
    path.mkdir(parents=True)
    with (path / f"{symbol}_merged.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "ts": str(row.ts),
                    "open": str(row.open),
                    "high": str(row.high),
                    "low": str(row.low),
                    "close": str(row.close),
                    "vol": str(row.volume),
                    "vol_ccy": "",
                    "vol_quote": "",
                    "confirm": "1",
                }
            )


class StubMarketData:
    def __init__(self, candles=None, error=None):
        self.candles = candles or []
        self.error = error
        self.calls = []

    def get_candles(self, symbol: str, bar: str = "1D", limit: int = 120):
        self.calls.append(("candles", symbol, bar, limit))
        if self.error:
            raise self.error
        return self.candles[-limit:]

    def latest_price(self, symbol: str) -> float:
        return 123.0

    def get_ticker(self, symbol: str):
        return Ticker(symbol=symbol, ts=1, last=123.0)

    def get_order_book(self, symbol: str, depth: int = 20):
        raise NotImplementedError


class RuntimeMarketCacheTest(unittest.TestCase):
    def test_shared_kline_cache_is_reused_by_independent_wrappers(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "shared_public_market_cache" / "kline_cache"
            now = 1_700_000_000_000
            upstream_rows = [candle(ts=now - idx * BAR_MS["1H"], close=100 + idx) for idx in range(3, 0, -1)]
            first_upstream = StubMarketData(candles=upstream_rows)
            first = RollingKlineCacheMarketData(cache, first_upstream, now_ms=lambda: now)

            first.get_candles("BTC-USDT-SWAP", bar="1H", limit=3)

            second_upstream = StubMarketData(candles=[candle(ts=now, close=999)])
            second = RollingKlineCacheMarketData(cache, second_upstream, now_ms=lambda: now)
            result = second.get_candles("BTC-USDT-SWAP", bar="1H", limit=3)

            self.assertEqual([row.close for row in result], [103.0, 102.0, 101.0])
            self.assertEqual(second_upstream.calls, [])

    def test_public_market_guard_cools_down_only_the_failed_endpoint(self):
        now = [1_700_000_000_000]
        upstream = StubMarketData(error=TimeoutError("ssl eof"))
        guard = PublicMarketDataGuard(
            upstream,
            provider_name="binance",
            now_ms=lambda: now[0],
            cooldown_base_seconds=60,
            cooldown_max_seconds=300,
        )

        with self.assertRaises(RuntimeError) as first_error:
            guard.get_candles("BTC-USDT-SWAP", bar="1H", limit=3)
        self.assertIn("public_market_cooldown_opened", str(first_error.exception))

        with self.assertRaises(PublicMarketDataCooldownError) as cooldown_error:
            guard.get_candles("BTC-USDT-SWAP", bar="1H", limit=3)
        self.assertIn("public_market_cooldown_skipped", str(cooldown_error.exception))
        self.assertEqual(upstream.calls, [("candles", "BTC-USDT-SWAP", "1H", 3)])

        with self.assertRaises(RuntimeError):
            guard.get_candles("ETH-USDT-SWAP", bar="1H", limit=3)
        self.assertEqual(upstream.calls[-1], ("candles", "ETH-USDT-SWAP", "1H", 3))

        now[0] += 61_000
        upstream.error = None
        upstream.candles = [candle(ts=now[0], close=123.0)]
        result = guard.get_candles("BTC-USDT-SWAP", bar="1H", limit=1)

        self.assertEqual(result[0].close, 123.0)
        self.assertEqual(guard.stats["public_market_recovered"], 1)

    def test_kline_cache_serves_stale_rows_when_guard_cools_down_tail_fetch(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "kline_cache"
            now = [1_700_000_000_000]
            cached = [candle(ts=now[0] - 10 * BAR_MS["1H"] + idx * BAR_MS["1H"], close=100 + idx) for idx in range(3)]
            write_cache_rows(cache, "BTC-USDT-SWAP", "1H", cached)
            upstream = StubMarketData(error=TimeoutError("ssl eof"))
            guard = PublicMarketDataGuard(
                upstream,
                provider_name="binance",
                now_ms=lambda: now[0],
                cooldown_base_seconds=60,
            )
            market = RollingKlineCacheMarketData(cache, guard, now_ms=lambda: now[0])

            first = market.get_candles("BTC-USDT-SWAP", bar="1H", limit=3)
            second = market.get_candles("BTC-USDT-SWAP", bar="1H", limit=3)

            self.assertEqual([row.close for row in first], [100.0, 101.0, 102.0])
            self.assertEqual([row.close for row in second], [100.0, 101.0, 102.0])
            self.assertEqual(upstream.calls, [("candles", "BTC-USDT-SWAP", "1H", 3)])
            self.assertEqual(market.stats["cache_stale_served"], 2)
            self.assertEqual(guard.stats["public_market_cooldown_skipped"], 1)

    def test_fresh_cache_serves_candles_without_public_fetch(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "kline_cache"
            now = 1_700_000_000_000 + 120 * BAR_MS["1H"]
            rows = [candle(ts=now - idx * BAR_MS["1H"], close=100 + idx) for idx in range(4, 0, -1)]
            write_cache_rows(cache, "BTC-USDT-SWAP", "1H", rows)
            upstream = StubMarketData(candles=[candle(ts=now + BAR_MS["1H"], close=999)])
            market = RollingKlineCacheMarketData(cache, upstream, now_ms=lambda: now)

            result = market.get_candles("BTC-USDT-SWAP", bar="1H", limit=4)

            self.assertEqual([row.close for row in result], [104.0, 103.0, 102.0, 101.0])
            self.assertEqual(upstream.calls, [])

    def test_stale_cache_fetches_only_missing_tail_and_dedupes_by_timestamp(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "kline_cache"
            start = 1_700_000_000_000
            cached = [candle(ts=start + idx * BAR_MS["1H"], close=100 + idx) for idx in range(118)]
            write_cache_rows(cache, "BTC-USDT-SWAP", "1H", cached)
            upstream_rows = [
                candle(ts=start + 117 * BAR_MS["1H"], close=217, source="public"),
                candle(ts=start + 118 * BAR_MS["1H"], close=218, source="public"),
                candle(ts=start + 119 * BAR_MS["1H"], close=219, source="public"),
            ]
            upstream = StubMarketData(candles=upstream_rows)
            market = RollingKlineCacheMarketData(
                cache,
                upstream,
                now_ms=lambda: start + 120 * BAR_MS["1H"],
                freshness_multiplier=1,
                fetch_buffer_bars=1,
            )

            result = market.get_candles("BTC-USDT-SWAP", bar="1H", limit=120)

            self.assertEqual(upstream.calls, [("candles", "BTC-USDT-SWAP", "1H", 4)])
            self.assertEqual(len(result), 120)
            self.assertEqual(result[-1].close, 219.0)
            self.assertEqual(len({row.ts for row in result}), 120)

    def test_exchange_failure_returns_available_stale_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "kline_cache"
            start = 1_700_000_000_000
            cached = [candle(ts=start + idx * BAR_MS["1H"], close=100 + idx) for idx in range(3)]
            write_cache_rows(cache, "BTC-USDT-SWAP", "1H", cached)
            market = RollingKlineCacheMarketData(
                cache,
                StubMarketData(error=RuntimeError("public eof")),
                now_ms=lambda: start + 10 * BAR_MS["1H"],
            )

            result = market.get_candles("BTC-USDT-SWAP", bar="1H", limit=3)

            self.assertEqual([row.close for row in result], [100.0, 101.0, 102.0])
            self.assertEqual(market.stats["tail_fetch_error"], 1)
            self.assertEqual(market.stats["cache_stale_served"], 1)

    def test_cache_miss_with_empty_exchange_response_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            market = RollingKlineCacheMarketData(
                Path(tmp) / "kline_cache",
                StubMarketData(candles=[]),
                now_ms=lambda: 1_700_000_000_000,
            )

            with self.assertRaises(RuntimeError) as raised:
                market.get_candles("BTC-USDT-SWAP", bar="1H", limit=3)

            self.assertIn("empty market data response", str(raised.exception))
            self.assertEqual(market.stats["tail_fetch_empty"], 1)

    def test_empty_exchange_response_returns_available_stale_cache_with_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "kline_cache"
            start = 1_700_000_000_000
            cached = [candle(ts=start + idx * BAR_MS["1H"], close=100 + idx) for idx in range(3)]
            write_cache_rows(cache, "BTC-USDT-SWAP", "1H", cached)
            market = RollingKlineCacheMarketData(
                cache,
                StubMarketData(candles=[]),
                now_ms=lambda: start + 10 * BAR_MS["1H"],
            )

            result = market.get_candles("BTC-USDT-SWAP", bar="1H", limit=3)

            self.assertEqual([row.close for row in result], [100.0, 101.0, 102.0])
            self.assertEqual(market.stats["tail_fetch_empty"], 1)
            self.assertEqual(market.stats["cache_stale_served"], 1)

    def test_market_snapshot_cache_writes_lightweight_summary_full_and_latest(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = MarketSnapshotCache(
                Path(tmp) / "market_snapshots",
                now_ms=lambda: 1_700_000_000_000,
            )

            paths = cache.write_snapshot(
                run_id="paper-run",
                phase="selection",
                symbol="BTC-USDT-SWAP",
                bar="multi",
                last_ts=1_700_000_000_000,
                features={
                    "selection_reason_codes": ["long_main_wave"],
                    "strong_pattern_tag": "golden_pit_reclaim",
                    "strong_pattern_score": 0.72,
                    "wyckoff_long_setup_tag": "spring_reclaim",
                    "wyckoff_long_score": 0.74,
                    "ema_12": 100.0,
                },
                full=True,
            )

            summary_rows = [json.loads(line) for line in paths["summary"].read_text(encoding="utf-8").splitlines()]
            full_rows = [json.loads(line) for line in paths["full"].read_text(encoding="utf-8").splitlines()]
            latest = json.loads(paths["latest"].read_text(encoding="utf-8"))
            self.assertEqual(paths["summary"].name, "2023-11-14.jsonl")
            self.assertIn("summary", paths["summary"].parts)
            self.assertIn("full", paths["full"].parts)
            self.assertEqual(summary_rows[0]["run_id"], "paper-run")
            self.assertEqual(summary_rows[0]["symbol"], "BTC-USDT-SWAP")
            self.assertEqual(summary_rows[0]["features"]["strong_pattern_tag"], "golden_pit_reclaim")
            self.assertEqual(summary_rows[0]["features"]["wyckoff_long_setup_tag"], "spring_reclaim")
            self.assertNotIn("ema_12", summary_rows[0]["features"])
            self.assertEqual(full_rows[0]["features"]["ema_12"], 100.0)
            self.assertEqual(latest["symbol"], "BTC-USDT-SWAP")
            self.assertEqual(latest["features"]["strong_pattern_tag"], "golden_pit_reclaim")

    def test_market_snapshot_cache_skips_full_for_ordinary_symbols(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = MarketSnapshotCache(
                Path(tmp) / "market_snapshots",
                now_ms=lambda: 1_700_000_000_000,
            )

            paths = cache.write_snapshot(
                run_id="paper-run",
                phase="selection",
                symbol="QUIET-USDT-SWAP",
                bar="multi",
                last_ts=1_700_000_000_000,
                features={"ret_20d": 0.02, "ema_12": 10.0},
                full=False,
            )

            self.assertIn("summary", paths)
            self.assertIn("latest", paths)
            self.assertNotIn("full", paths)
            self.assertFalse((Path(tmp) / "market_snapshots" / "full").exists())

    def test_market_snapshot_cache_compresses_old_daily_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "market_snapshots"
            old_summary = root / "summary" / "2023-11-13.jsonl"
            old_summary.parent.mkdir(parents=True)
            old_summary.write_text('{"old": true}\n', encoding="utf-8")
            cache = MarketSnapshotCache(
                root,
                now_ms=lambda: 1_700_000_000_000,
            )

            cache.write_snapshot(
                run_id="paper-run",
                phase="selection",
                symbol="BTC-USDT-SWAP",
                bar="multi",
                last_ts=1_700_000_000_000,
                features={"ret_20d": 0.1},
            )

            compressed = root / "summary" / "2023-11-13.jsonl.gz"
            self.assertFalse(old_summary.exists())
            self.assertTrue(compressed.exists())
            with gzip.open(compressed, "rt", encoding="utf-8") as handle:
                self.assertEqual(handle.read(), '{"old": true}\n')


if __name__ == "__main__":
    unittest.main()

import csv
import os
import tempfile
import unittest
from pathlib import Path

from langlang_trader.kline_backfill import (
    BAR_MS,
    BinanceHistoryClient,
    CSV_COLUMNS,
    FetchResult,
    RequiredWindow,
    _read_digest_required_windows,
    _process_one_window,
    _read_cache_stats,
    _window_cache_status,
    _write_cache_file,
    backfill_trade_kline_cache,
    build_required_windows,
)


class KlineBackfillTest(unittest.TestCase):
    def test_binance_history_client_uses_multiplier_symbol_mapping(self):
        seen_urls = []

        def fake_getter(url, *, timeout):
            seen_urls.append(url)
            return [], ""

        client = BinanceHistoryClient(
            sleep_seconds=0,
            timeout=1,
            symbol_map={"AIDOGE-USDT-SWAP": "1000000AIDOGEUSDT"},
            json_getter=fake_getter,
        )

        client.fetch("AIDOGE-USDT-SWAP", "1D", 1_700_000_000_000, 1_700_086_400_000)

        self.assertTrue(seen_urls)
        self.assertIn("symbol=1000000AIDOGEUSDT", seen_urls[0])

    def test_required_windows_merge_by_symbol_bar_without_unknown_status(self):
        trades = [
            {
                "trade_id": "1",
                "symbol": "BTC-USDT-SWAP",
                "entry_ts": 1_700_000_000_000,
                "exit_ts": 1_700_003_600_000,
            },
            {
                "trade_id": "2",
                "symbol": "BTC-USDT-SWAP",
                "entry_ts": 1_700_003_600_000,
                "exit_ts": 1_700_007_200_000,
            },
        ]

        windows = build_required_windows(trades, bars=["1m"])

        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].trade_ids, ("1", "2"))

    def test_no_network_backfill_writes_explicit_coverage_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            trades = os.path.join(tmp, "trades.csv")
            cache = os.path.join(tmp, "cache")
            out = os.path.join(tmp, "out")
            os.makedirs(os.path.join(cache, "1D"))
            with open(trades, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["trade_id", "entry_time", "exit_time", "symbol"])
                writer.writeheader()
                writer.writerow(
                    {
                        "trade_id": "1",
                        "entry_time": "2024-01-02 08:00:00",
                        "exit_time": "2024-01-02 09:00:00",
                        "symbol": "BTC-USDT-SWAP",
                    }
                )

            result = backfill_trade_kline_cache(
                trades_csv=trades,
                cache_dir=cache,
                out_dir=out,
                bars=["1D"],
                allow_network=False,
                time_offset_hours=-8,
            )

            self.assertEqual(result["trades"], 1)
            self.assertEqual(result["coverage_status"], "partial_feature_coverage")
            with open(result["csv"], encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["coverage_status"], "missing")
            self.assertEqual(rows[0]["unavailable_reason"], "cache_window_missing")

    def test_no_network_backfill_classifies_partial_cache_below_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            trades = os.path.join(tmp, "trades.csv")
            cache = Path(tmp) / "cache"
            out = os.path.join(tmp, "out")
            (cache / "1m").mkdir(parents=True)
            with open(trades, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["trade_id", "entry_time", "exit_time", "symbol"])
                writer.writeheader()
                writer.writerow(
                    {
                        "trade_id": "1",
                        "entry_time": "2024-01-02 08:00:00",
                        "exit_time": "2024-01-02 09:00:00",
                        "symbol": "BTC-USDT-SWAP",
                    }
                )
            cache_file = cache / "1m" / "BTC-USDT-SWAP_partial.csv"
            with cache_file.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
                writer.writeheader()
                writer.writerow(
                    {
                        "ts": "1704074400000",
                        "open": "1",
                        "high": "1",
                        "low": "1",
                        "close": "1",
                        "vol": "1",
                        "vol_ccy": "",
                        "vol_quote": "",
                        "confirm": "1",
                    }
                )

            result = backfill_trade_kline_cache(
                trades_csv=trades,
                cache_dir=cache,
                out_dir=out,
                bars=["1m"],
                allow_network=False,
                time_offset_hours=-8,
            )

            with open(result["csv"], encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["coverage_status"], "missing")
            self.assertEqual(rows[0]["unavailable_reason"], "partial_cache_below_threshold")

    def test_cache_window_requires_start_and_end_boundary_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache"
            (cache / "1D").mkdir(parents=True)
            cache_file = cache / "1D" / "BTC-USDT-SWAP_partial.csv"
            with cache_file.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
                writer.writeheader()
                for idx in range(9):
                    writer.writerow(
                        {
                            "ts": str(1000 + (idx + 1) * BAR_MS["1D"]),
                            "open": "1",
                            "high": "1",
                            "low": "1",
                            "close": "1",
                            "vol": "1",
                            "vol_ccy": "",
                            "vol_quote": "",
                            "confirm": "1",
                        }
                    )

            status, evidence = _window_cache_status(
                cache,
                RequiredWindow("BTC-USDT-SWAP", "1D", 1000, 1000 + 9 * BAR_MS["1D"], ("1",)),
            )

            self.assertEqual(status, "missing")
            self.assertIn("edge_covered=false", evidence)

    def test_partial_okx_fetch_falls_back_to_binance_until_window_available(self):
        class FakeClient:
            def __init__(self, result):
                self.result = result
                self.calls = 0

            def fetch(self, symbol, bar, start_ms, end_ms):
                self.calls += 1
                return self.result

        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache"
            window = RequiredWindow("BTC-USDT-SWAP", "1m", 1_000, 1_000 + 2 * BAR_MS["1m"], ("1",))
            okx = FakeClient(
                FetchResult(
                    "okx",
                    [
                        {
                            "ts": 1_000,
                            "open": "1",
                            "high": "1",
                            "low": "1",
                            "close": "1",
                            "vol": "1",
                            "vol_ccy": "",
                            "vol_quote": "",
                            "confirm": "1",
                        }
                    ],
                    "transient eof",
                )
            )
            binance = FakeClient(
                FetchResult(
                    "binance",
                    [
                        {
                            "ts": 1_000 + idx * BAR_MS["1m"],
                            "open": "1",
                            "high": "1",
                            "low": "1",
                            "close": "1",
                            "vol": "1",
                            "vol_ccy": "",
                            "vol_quote": "",
                            "confirm": "1",
                        }
                        for idx in range(3)
                    ],
                )
            )

            row, did_fetch = _process_one_window(
                window,
                cache_path=cache,
                allow_network=True,
                okx=okx,
                binance=binance,
            )

            self.assertEqual(did_fetch, 1)
            self.assertEqual(okx.calls, 1)
            self.assertEqual(binance.calls, 1)
            self.assertEqual(row["coverage_status"], "available")
            self.assertEqual(row["source"], "binance")

    def test_write_cache_file_merges_existing_rows_by_timestamp(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = os.path.join(tmp, "cache")
            path = os.path.join(cache, "1m")
            os.makedirs(path)
            existing = os.path.join(path, "BTC-USDT-SWAP_old.csv")
            with open(existing, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
                writer.writeheader()
                writer.writerow(
                    {
                        "ts": "1000",
                        "open": "1",
                        "high": "2",
                        "low": "1",
                        "close": "2",
                        "vol": "10",
                        "vol_ccy": "",
                        "vol_quote": "",
                        "confirm": "1",
                    }
                )

            written = _write_cache_file(
                cache_dir=Path(cache),
                window=RequiredWindow("BTC-USDT-SWAP", "1m", 1000, 2000, ("1",)),
                rows=[
                    {
                        "ts": 1000,
                        "open": "3",
                        "high": "4",
                        "low": "3",
                        "close": "4",
                        "vol": "11",
                        "vol_ccy": "",
                        "vol_quote": "",
                        "confirm": "1",
                    },
                    {
                        "ts": 2000,
                        "open": "4",
                        "high": "5",
                        "low": "4",
                        "close": "5",
                        "vol": "12",
                        "vol_ccy": "",
                        "vol_quote": "",
                        "confirm": "1",
                    },
                ],
                source="okx",
            )

            self.assertTrue(str(written).endswith("BTC-USDT-SWAP_merged.csv"))
            first, last, count, timestamps = _read_cache_stats(written)
            self.assertEqual((first, last, count, timestamps), (1000, 2000, 2, [1000, 2000]))

    def test_digest_coverage_reader_defaults_to_missing_rows_and_merges_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            coverage = Path(tmp) / "kline_window_coverage.csv"
            with coverage.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "trade_id",
                        "symbol",
                        "timeframe",
                        "window_profile",
                        "start_ms",
                        "end_ms",
                        "coverage_status",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "trade_id": "available-skip",
                        "symbol": "BTC-USDT-SWAP",
                        "timeframe": "1D",
                        "window_profile": "standard",
                        "start_ms": "1000",
                        "end_ms": str(1000 + BAR_MS["1D"]),
                        "coverage_status": "available",
                    }
                )
                writer.writerow(
                    {
                        "trade_id": "1",
                        "symbol": "BTC-USDT-SWAP",
                        "timeframe": "1D",
                        "window_profile": "expanded_manual_review",
                        "start_ms": "1000",
                        "end_ms": str(1000 + BAR_MS["1D"]),
                        "coverage_status": "insufficient_history",
                    }
                )
                writer.writerow(
                    {
                        "trade_id": "2",
                        "symbol": "BTC-USDT-SWAP",
                        "timeframe": "1D",
                        "window_profile": "standard",
                        "start_ms": str(1000 + BAR_MS["1D"]),
                        "end_ms": str(1000 + 2 * BAR_MS["1D"]),
                        "coverage_status": "insufficient_history",
                    }
                )

            windows = _read_digest_required_windows(coverage, bars=["1D"], missing_only=True)

            self.assertEqual(len(windows), 1)
            self.assertEqual(windows[0].symbol, "BTC-USDT-SWAP")
            self.assertEqual(windows[0].bar, "1D")
            self.assertEqual(windows[0].start_ms, 0)
            self.assertEqual(windows[0].end_ms, 3 * BAR_MS["1D"])
            self.assertEqual(windows[0].trade_ids, ("1", "2"))

    def test_digest_coverage_reader_skips_terminal_unavailable_rows_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            coverage = Path(tmp) / "kline_window_coverage.csv"
            coverage.write_text(
                "trade_id,symbol,timeframe,window_profile,start_ms,end_ms,coverage_status\n"
                "1,GONE-USDT-SWAP,1D,standard,1000,2000,instrument_unavailable\n"
                "2,MISS-USDT-SWAP,1D,standard,1000,2000,insufficient_history\n",
                encoding="utf-8",
            )

            windows = _read_digest_required_windows(coverage, bars=["1D"], missing_only=True)

            self.assertEqual([window.symbol for window in windows], ["MISS-USDT-SWAP"])

    def test_backfill_can_use_digest_coverage_without_trade_csv_and_only_fetch_missing_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            coverage = root / "coverage.csv"
            cache = root / "cache"
            out = root / "out"
            coverage.write_text(
                "trade_id,symbol,timeframe,window_profile,start_ms,end_ms,coverage_status\n"
                f"1,BTC-USDT-SWAP,1D,standard,1000,{1000 + BAR_MS['1D']},available\n"
                f"2,ETH-USDT-SWAP,1D,standard,1000,{1000 + BAR_MS['1D']},insufficient_history\n",
                encoding="utf-8",
            )

            result = backfill_trade_kline_cache(
                trades_csv=None,
                required_windows_csv=None,
                coverage_input_csv=coverage,
                cache_dir=cache,
                out_dir=out,
                bars=["1D"],
                allow_network=False,
            )

            self.assertEqual(result["trades"], 0)
            self.assertEqual(result["required_windows"], 1)
            with open(result["csv"], encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["symbol"], "ETH-USDT-SWAP")

    def test_network_empty_exchange_response_becomes_terminal_unavailable_status(self):
        class EmptyClient:
            def __init__(self, error):
                self.error = error
                self.calls = 0

            def fetch(self, symbol, bar, start_ms, end_ms):
                self.calls += 1
                return FetchResult("mock", [], self.error)

        with tempfile.TemporaryDirectory() as tmp:
            window = RequiredWindow("GONE-USDT-SWAP", "1D", 1000, 1000 + BAR_MS["1D"], ("1",))

            row, did_fetch = _process_one_window(
                window,
                cache_path=Path(tmp) / "cache",
                allow_network=True,
                okx=EmptyClient("okx:51001:Instrument ID does not exist"),
                binance=EmptyClient("binance:-1121:Invalid symbol."),
            )

            self.assertEqual(did_fetch, 1)
            self.assertEqual(row["coverage_status"], "instrument_unavailable")
            self.assertIn("Invalid symbol", row["evidence"])


if __name__ == "__main__":
    unittest.main()
